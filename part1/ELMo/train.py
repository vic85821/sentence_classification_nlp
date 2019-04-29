import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import Perplexity
from predictor import Predictor


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
        
    logging.info('loading valid data...')
    with open(config['model']['valid'], 'rb') as f:
        config['model']['valid'] = pickle.load(f)
    
    logging.info('loading character vocabulary...')
    with open(config['charmap'], 'rb') as f:
        charmap = pickle.load(f)
    logging.info('loading word vocabulary...')
    with open(config['wordmap'], 'rb') as f:
        wordmap = pickle.load(f)
    config['model']['num_embeddings'] = len(charmap)
    config['model']['padding_idx'] = charmap['<PAD>']
    config['model']['vocab_size'] = len(wordmap)
    
    predictor = Predictor(
        arch=config['arch'],
        device=args.device,
        metrics=[Perplexity()],
        **config['model']
    )

    if args.load is not None:
        predictor.load(args.load)

    model_checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model.pkl'),
        **config['callbacks']
    )
    metrics_logger = MetricsLogger(
        os.path.join(args.model_dir, 'log.json')
    )

    logging.info('start training!')
    predictor.fit_dataset(config['train'], model_checkpoint, metrics_logger)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str,
                        help='The model path to be loaded.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)