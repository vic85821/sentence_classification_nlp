import os
import json
import argparse
import pickle
import logging
import random
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
import numpy as np
from dataset import LanguageModelDataset

class Preprocess:
    def __init__(self, config):
        self.config = config
        self.samples = None
        self.wordkeys = []
        self.charkeys = []
        self.logging = logging.getLogger(name=__name__)
        self.n_workers = config['n_workers']
        self.wordcount = Counter()
        self.charcount = Counter()
        
    def load_corpus(self):
        """ Load the corpus from file
        """
        with open(self.config['corpus_path'], 'r') as fin:
            lines = fin.readlines()
        random.seed()
        self.samples = random.sample(lines, self.config['num_train'])
        
    def build_vocabulary(self):
        """ Build the word / character counter and vocabulary
        """
        # count the words in the sampled sentences
        self.logging.info("counting the words in the sampled sentences...")
        for sentence in tqdm(self.samples):
            self.wordcount.update(("<BOS> " + sentence + " <EOS>").split())
        
        # count the characters of the keys in the word Counter
        self.split_character()
        
        # save the processed files
        self.save_processed_files()
        
    def split_character(self):
        """ Split the key of the word vocabulary and build the character level
            dictionary for character CNN
        """
        self.logging.info("spliting the key in the vocabulary into characters...")
        
        # count the characters in the key of the wordcount
        unk = []
        unk_num = 0
        for key in self.wordcount:
            if self.wordcount[key] < 3:
                unk.append(key)
                unk_num += self.wordcount[key]
            if key in ['<BOS>', '<EOS>']:
                self.charcount.update({key: self.wordcount[key]})
            else:
                self.charcount.update(key*self.wordcount[key])
        self.wordcount.update({"<UNK>": unk_num, "<PAD>": 2})
        for key in unk:
            self.wordcount.pop(key)
        
        # delete the character frequecy < 1000
        unk = []
        unk_num = 0
        for key in self.charcount:
            if self.charcount[key] < 1000:
                unk.append(key)
                unk_num += self.charcount[key]
        self.charcount.update({"<UNK>": unk_num, "<PAD>": 999})
        for key in unk:
            self.charcount.pop(key)
        
    def save_processed_files(self):
        """ Save the samples, unknown word list, and unknown character list
        """ 
        self.wordmap = {}
        for idx, value in enumerate(self.wordcount.most_common()):
            if value[0] == '<PAD>':
                continue
            self.wordmap[value[0]] = idx+1
        self.wordmap['<PAD>'] = 0
        
        self.charmap = {}
        for idx, value in enumerate(self.charcount.most_common()):
            if value[0] == '<PAD>':
                continue
            self.charmap[value[0]] = idx+1
        self.charmap['<PAD>'] = 0
        
        corpus_path = os.path.join(self.config['root'], 'corpus.pkl')
        wordmap_path = os.path.join(self.config['root'], 'wordmap.pkl')
        charmap_path = os.path.join(self.config['root'], 'charmap.pkl')
        with open(corpus_path, "wb+") as f:
            pickle.dump(self.samples, f)
        with open(wordmap_path, "wb+") as f:
            pickle.dump(self.wordmap, f)
        with open(charmap_path, "wb+") as f:
            pickle.dump(self.charmap, f)
            
    def load_processed_files(self):
        """ Load the samples, unknown word list, and unknown character list
        """
        corpus_path = os.path.join(self.config['root'], 'corpus.pkl')
        wordmap_path = os.path.join(self.config['root'], 'wordmap.pkl')
        charmap_path = os.path.join(self.config['root'], 'charmap.pkl')
        with open(corpus_path, 'rb') as f:
            self.samples = pickle.load(f)
        with open(wordmap_path, 'rb') as f:
            self.wordmap = pickle.load(f)
        with open(charmap_path, 'rb') as f:
            self.charmap = pickle.load(f)
        
    def get_dataset(self):
        """ Prepare the LanguageModelDataset for the train / valid samples
        """
        num_sample = len(self.samples)
        train, valid = self.samples[:num_sample//20*19], self.samples[num_sample//20*19:]
        valid_path = os.path.join(self.config['root'], 'valid.pkl')
        
        for mode in ['train', 'valid']:
            self.logging.info("generating the {} data...".format(mode))
            sample = train if mode == 'train' else valid
            
            results = [None] * self.n_workers
            with Pool(processes=self.n_workers) as pool:
                for i in range(self.n_workers):
                    batch_start = (len(sample) // self.n_workers) * i
                    batch_end = len(sample) if i == self.n_workers - 1 else (len(sample) // self.n_workers) * (i + 1)
                    batch = sample[batch_start: batch_end]
                    results[i] = pool.apply_async(self.prepare_samples, [batch])
                pool.close()
                pool.join()

            processed = []
            for result in results:
                processed.extend(result.get())
            
            if mode == 'train':
                for i in range(20):
                    file_path = os.path.join(self.config['root'], 'train_{}.pkl'.format(i))
                    partition = len(processed)//20
                    dataset = LanguageModelDataset(processed[i*partition:(i+1)*partition], self.wordmap['<PAD>'], self.charmap['<PAD>'], **self.config['dataset'])
                    with open(file_path, "wb") as f:
                        pickle.dump(dataset, f)
            else:
                file_path = valid_path
                dataset = LanguageModelDataset(processed, self.wordmap['<PAD>'], self.charmap['<PAD>'], **self.config['dataset'])
                with open(file_path, "wb") as f:
                    pickle.dump(dataset, f)
            

    def prepare_samples(self, samples):
        processed = []
        for sample in tqdm(samples):
            processed.extend(self.prepare_sample(sample))
        return processed
        
    def prepare_sample(self, data):
        """ process a sample(sentence) into X and Y pairs to feed in the LanguageModelDataset
            Args:
                data (str): sentence
            Return:
                a list of the processed dict ({'X': [[...], [...], ...], 'Y': [......]}) which contains forward and backward
                input and target data pairs.
        """
        sentence = ("<BOS> " + data + " <EOS>").split()
        length = len(sentence)
        processed = []
        for i in range(0, length, 65):
            sample = sentence[i:i+65]
            sentence_lens = min(65, len(sample))
            if sentence_lens < 32:
                break
                
            char_list = [list(word) if ((word != '<BOS>') and (word != '<EOS>')) else [word] for word in sample]
            unk_word = self.wordmap["<UNK>"]
            unk_char = self.charmap["<UNK>"]
            
            for j in range(len(sample)):
                sample[j] = self.wordmap[sample[j]] if sample[j] in self.wordmap else unk_word
            for j in range(len(char_list)):
                for k in range(len(char_list[j])):
                    char_list[j][k] = self.charmap[char_list[j][k]] if char_list[j][k] in self.charmap else unk_char
            
            forward = {"X": char_list[:sentence_lens-1], 
                       "Y": sample[1:sentence_lens]}
            backward = {"X": char_list[1:sentence_lens][::-1], 
                        "Y": sample[:sentence_lens-1][::-1]}
            
            processed.append({"forward": forward, "backward": backward})
        return processed
    
def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    
    logging.info('loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)
    
    preprocess = Preprocess(config)
    wordmap_path = os.path.join(config['root'], 'wordmap.pkl')
    charmap_path = os.path.join(config['root'], 'charmap.pkl')
    train_path = os.path.join(config['root'], 'train.pkl')
    valid_path = os.path.join(config['root'], 'valid.pkl')
    
    if not os.path.isfile(wordmap_path) or not os.path.isfile(charmap_path):
        logging.info('loading corpus...')
        preprocess.load_corpus()
        preprocess.build_vocabulary()
    else:
        logging.info('corpus has been processed')
        preprocess.load_processed_files()
        
    if not os.path.isfile(train_path) or not os.path.isfile(valid_path):
        preprocess.get_dataset()
    else:
        logging.info('training data and validation data have been existed')
    
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str,
                        help='[input] Path to the directory that .')
    args = parser.parse_args()
    return args    

    
if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    
    args = _parse_args()
    main(args)