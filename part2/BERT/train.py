import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from box import Box

from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam
from .dataset import create_data_loader
from common.base_model import BaseModel
from common.base_trainer import BaseTrainer
from common.losses import CrossEntropyLoss
from common.metrics import Accuracy
from common.utils import load_pkl



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')
    args = parser.parse_args()

    return vars(args)


class Model(BaseModel):
    def _create_net_and_optim(self, net_cfg, optim_cfg, num_train_optimization_steps):
        net = BertForSequenceClassification.from_pretrained(net_cfg.bert_pretrain, net_cfg.num_labels)
        net.to(device=self._device)

        param_optimizer = filter(lambda p: p.requires_grad, net.parameters())
        if num_train_optimization_steps != None:
            optim = BertAdam(param_optimizer,
                             t_total=num_train_optimization_steps,
                             **optim_cfg.kwargs)
        else:
            optim = None
        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, max_sent_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_sent_len = max_sent_len

    def _run_batch(self, batch):
        text_token_id = batch['text_token_id'].to(device=self._device)
        text_pad_mask = batch['text_pad_mask'].to(device=self._device)

        for i in range(len(self._model)):
            logits = self._model[i](input_ids=text_token_id,
                                 attention_mask=text_pad_mask)
        label = logits.max(dim=1)[1]

        return {
            'logits': logits,
            'label': label
        }


def main(model_dir):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)
    print(
        '[-] Model checkpoints and training log will be saved to {}\n'
        .format(model_dir))

    #device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    device = torch.device("cuda:0")
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = model_dir / 'log.csv'
    ckpt_dir = model_dir / 'ckpts'
    if any([p.exists() for p in [log_path, ckpt_dir]]):
        print('[!] Directory already contains saved ckpts/log')
        exit(1)
    ckpt_dir.mkdir()

    print('[*] Loading datasets from {}'.format(cfg.dataset_dir))
    dataset_dir = Path(cfg.dataset_dir)
    train_dataset = load_pkl(dataset_dir / 'train.pkl')
    dev_dataset = load_pkl(dataset_dir / 'dev.pkl')

    print('[*] Creating train/dev data loaders\n')
    if cfg.data_loader.batch_size % cfg.train.n_gradient_accumulation_steps != 0:
        print(
            '[!] n_gradient_accumulation_steps({}) is not a divider of '
            .format(cfg.train.n_gradient_accumulation_steps),
            'batch_size({})'.format(cfg.data_loader.batch_size))
        exit(1)
    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    train_data_loader = create_data_loader(train_dataset, **cfg.data_loader)
    dev_data_loader = create_data_loader(dev_dataset, **cfg.data_loader)
    num_train_optimization_steps = \
    int(len(train_data_loader) / cfg.train.n_gradient_accumulation_steps) * cfg.train.n_epochs

    print('[*] Creating model\n')
    model = Model(device, cfg.random_seed, cfg.net, cfg.optim, num_train_optimization_steps)

    trainer = Trainer(
        cfg.data_loader.max_sent_len, device, cfg.train,
        train_data_loader, dev_data_loader, [model],
        [CrossEntropyLoss(device, 'logits', 'label')], [Accuracy(device, 'label')],
        log_path, ckpt_dir)
    trainer.start()

if __name__ == "__main__":
    '''
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
    '''
    kwargs = parse_args()
    main(**kwargs)
