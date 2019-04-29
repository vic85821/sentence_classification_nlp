import argparse
import csv
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path

import ipdb
import spacy
from box import Box
from tqdm import tqdm

from .dataset import Part1Dataset
from common.vocab import Vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Train or Test mode')
    parser.add_argument('--dataset_dir', type=Path, help='[Train] Target dataset directory')
    parser.add_argument('--test_csv', type=Path, help='[Test] input csv file path')
    args = parser.parse_args()

    return vars(args)


def load_data(mode, data_path):
    print('[*] Loading {} data from {}'.format(mode, data_path))
    with data_path.open() as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]

    for d in tqdm(data, desc='[*] Tokenizing', dynamic_ncols=True):
        text = re.sub('-+', ' ', d['text'])
        text = re.sub('\s+', ' ', text)
        d['text'] = "[CLS] " + text + " [SEP]"
    print('[-] {} data loaded\n'.format(mode.capitalize()))

    return data

def create_dataset(data, dataset_dir, cfg):
    for m, d in data.items():
        print('[*] Creating {} dataset'.format(m))
        dataset = Part1Dataset(d, cfg.bert_tokenizer.pretrain, cfg.bert_tokenizer.do_lower_case)
        dataset_path = (dataset_dir / '{}.pkl'.format(m))
        with dataset_path.open(mode='wb') as f:
            pickle.dump(dataset, f)
        print('[-] {} dataset saved to {}\n'.format(m.capitalize(), dataset_path))


def main(mode, dataset_dir, test_csv):
    if mode == 'train':
        dataset_dir = Path(dataset_dir)
        try:
            cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
        except FileNotFoundError:
            print('[!] Dataset directory({}) must contain config.yaml'.format(dataset_dir))
            exit(1)
        print('[-] datasets will be saved to {}\n'.format(dataset_dir))

        # training mode
        output_files = ['train.pkl', 'dev.pkl', 'test.pkl']
        if any([(dataset_dir / p).exists() for p in output_files]):
            print('[!] Directory already contains saved dataset')
            exit(1)

        data_dir = Path(cfg.data_dir)
        data = {m: load_data(m, data_dir / '{}.csv'.format(m))
                for m in ['train', 'dev', 'test']}
    else:
        dataset_dir = Path('./bert_dataset/')
        cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
        data = {'test': load_data('test', test_csv)}

    create_dataset(data, dataset_dir, cfg)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
