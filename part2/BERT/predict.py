import argparse
import csv
import random
import sys
from pathlib import Path

import ipdb
import numpy as np
import torch
from box import Box
from tqdm import tqdm

from .dataset import create_data_loader
from .train import Model
from common.utils import load_pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Model directory')
    parser.add_argument('epoch', type=int, help='Model checkpoint number')
    parser.add_argument('--batch_size', type=int, help='Inference batch size')
    parser.add_argument('--pred_path', type=Path, help='Path of the prediction csv file')
    args = parser.parse_args()

    return vars(args)


def main(model_dir, epoch, batch_size, pred_path):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prediction_dir = model_dir / 'predictions'
    if not prediction_dir.exists():
        prediction_dir.mkdir()
        print('[-] Directory {} created'.format(prediction_dir))

    dataset_dir = Path(cfg.dataset_dir)
    test_dataset_path = dataset_dir / 'test.pkl'
    ckpt_path = model_dir / 'ckpts' / 'random-{}_epoch-{}.ckpt'.format(cfg.random_seed, epoch)
    print('[-] Test dataset: {}'.format(test_dataset_path))
    print('[-] Model checkpoint: {}\n'.format(ckpt_path))

    print('[*] Loading vocabs and test dataset from {}'.format(dataset_dir))
    test_dataset = load_pkl(test_dataset_path)

    print('[*] Creating test data loader\n')
    if batch_size:
        cfg.data_loader.batch_size = batch_size
    data_loader = create_data_loader(test_dataset, **cfg.data_loader)

    print('[*] Creating model\n')
    model = Model(device, cfg.random_seed, cfg.net, cfg.optim, None)
    model.load_state(ckpt_path, True)

    Ids, predictions = predict(
        device, data_loader, cfg.data_loader.max_sent_len, [model])
    save_predictions(Ids, predictions, pred_path)


def predict(device, data_loader, max_sent_len, model):
    for i in range(len(model)):
        model[i].set_eval()
    with torch.no_grad():
        Ids = []
        predictions = []
        bar = tqdm(data_loader, desc='[Predict]', leave=False, dynamic_ncols=True)
        for batch in bar:
            Ids += batch['Id']
            text_token_id = batch['text_token_id'].to(device=device)
            text_pad_mask = batch['text_pad_mask'].to(device=device)

            sum_logits = None
            for i in range(len(model)):
                logits = model[i](input_ids=text_token_id,
                                  attention_mask=text_pad_mask)
                try:
                    sum_logits += logits
                except:
                    sum_logits = logits

            label = sum_logits.max(dim=1)[1]
            predictions += label.tolist()
        bar.close()

    return Ids, predictions


def save_predictions(Ids, predictions, output_path):
    with output_path.open(mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'label'])
        writer.writeheader()
        writer.writerows(
            [{'Id': Id, 'label': p + 1} for Id, p in zip(Ids, predictions)])
    print('[-] Output saved to {}'.format(output_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
