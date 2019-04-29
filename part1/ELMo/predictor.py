import torch
from base_predictor import BasePredictor
from modules import *


class Predictor(BasePredictor):
    """
    Args:
        embedding: matrix for embeding (word size x embedding size)
        arch: model name
        loss: loss name
        kwargs:
            batch_size
            max_epochs
            learning_rate
            n_workers
    """

    def __init__(self, arch, num_embeddings, padding_idx, vocab_size, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        
        self.model = {
            'ELMo': ELMo(num_embeddings, padding_idx, vocab_size)
        }[arch]
        print(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _run_iter(self, batch, training):
        with torch.no_grad():
            forward_Y_lens = torch.LongTensor(batch['forward_Y_lens'])
            backward_Y_lens = torch.LongTensor(batch['backward_Y_lens'])
            forward_mask = (torch.arange(64)[None, :] < forward_Y_lens[:, None]).float()
            backward_mask = (torch.arange(64)[None, :] < backward_Y_lens[:, None]).float()
        
        forward_logits, backward_logits = self.model.forward(batch['forward_X'].to(self.device),
                                                             batch['forward_Y'].to(self.device),
                                                             batch['backward_X'].to(self.device),
                                                             batch['backward_Y'].to(self.device))
        
        forward_loss = -forward_logits * forward_mask.to(self.device)
        backward_loss = -backward_logits * backward_mask.to(self.device)
        
        return torch.sum(forward_loss, dim=1) / forward_Y_lens.to(self.device).float(), \
               torch.sum(backward_loss, dim=1) / backward_Y_lens.to(self.device).float()

    def _predict_batch(self, batch):
        forward_X = batch['forward_X']
        forward_Y = batch['forward_Y']
        backward_X = batch['backward_X']
        backward_Y = batch['backward_Y']
        forward_logits, backward_logits = self.model.forward(forward_X.to(self.device),
                                                             forward_Y.to(self.device),
                                                             backward_X.to(self.device),
                                                             backward_Y.to(self.device))
        
        return forward_logits, backward_logits
