import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


class myLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.iter = 0
        self.index = list(range(len(self.dataset)))

        if self.shuffle == True:
            import random
            random.shuffle(self.index)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __next__(self):
        indices = self.index[self.iter * self.batch_size : (self.iter+1) * self.batch_size]
        batch = self.collate_fn([self.dataset[i] for i in indices])
        self.iter += 1
        return batch


class BasePredictor():
    def __init__(self,
                 batch_size=10,
                 max_epochs=30,
                 valid=None,
                 device=None,
                 metrics={},
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 max_iters_in_epoch=1e20,
                 grad_accumulate_steps=1,
                 n_workers=4):
        
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps
        self.n_workers = n_workers
        
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.epoch = 0

    def fit_dataset(self, train_path, checkpoint, logger):
        # Start the training loop.
        while self.epoch < self.max_epochs:
            print('loading train_{}.pkl'.format(self.epoch % 20))
            path = train_path + "_{}.pkl".format(self.epoch % 20)
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # train and evaluate train score
            print('training %i' % self.epoch)
            dataloader = myLoader(dataset=data, 
                                  batch_size=self.batch_size,
                                  collate_fn=data.collate_fn,
                                  shuffle=True)

            # train epoch
            log_train = self._run_epoch(dataloader, True)
            log_train = None
            logger.on_epoch_end(log_train, self)
            
            # evaluate valid score
            if self.valid is not None:
                print('evaluating %i' % self.epoch)
                dataloader = myLoader(dataset=self.valid, 
                                        batch_size=self.batch_size,
                                        collate_fn=data.collate_fn,
                                        shuffle=False)
                log_valid = self._run_epoch(dataloader, False)
            else:
                log_valid = None

            # callbacks
            if checkpoint.on_epoch_end(log_train, log_valid, self):
                break
            logger.on_epoch_end(log_valid, self)

            self.epoch += 1

    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()

        # make dataloader
        dataloader = DataLoader(dataset=data, 
                                batch_size=self.batch_size,
                                collate_fn=collate_fn,
                                shuffle=False,
                                num_workers=self.n_workers)
        ys_ = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _run_epoch(self, dataloader, training):
        # run batches for train
        loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset()

        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = 'training'
        else:
            iter_in_epoch = len(dataloader)
            description = 'evaluating'

        # run batches
        trange = tqdm(enumerate(dataloader),
                      total=iter_in_epoch,
                      desc=description)
        for i, batch in trange:
            if training and i >= iter_in_epoch:
                break

            if training:
                forward_loss, backward_loss = self._run_iter(batch, training)
                batch_loss = torch.mean(torch.cat((forward_loss, backward_loss), dim=0))
                batch_loss /= self.grad_accumulate_steps

                # accumulate gradient - zero_grad
                if i % self.grad_accumulate_steps == 0:
                    self.optimizer.zero_grad()
                
                batch_loss.backward()

                # accumulate gradient - step
                if (i + 1) % self.grad_accumulate_steps == 0:
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    forward_loss, backward_loss = self._run_iter(batch, training)
                    batch_loss = torch.mean(torch.cat((forward_loss, backward_loss), dim=0))
                    
            # accumulate loss and metric scores
            loss += batch_loss.item()
            perplexity_score = torch.mean(torch.exp(torch.cat((forward_loss.data, backward_loss.data), dim=0))).item()
            
            for metric in self.metrics:
                metric.update(perplexity_score)
            trange.set_postfix(
                loss = loss / (i + 1),
                **{m.name: m.print_score() for m in self.metrics})
            
        # calculate averate loss and metrics
        loss /= iter_in_epoch

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            print('{}: {} '.format(metric.name, score))
            epoch_log[metric.name] = score
        print('loss=%f\n' % loss)
        return epoch_log

    def _run_iter(self, batch, training):
        """ Run iteration for training.

        Args:
            batch (dict)
            training (bool)

        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        pass

    def _predict_batch(self, batch):
        """ Run iteration for predicting.

        Args:
            batch (dict)

        Returns:
            predicts: Prediction of the batch.
        """
        pass
