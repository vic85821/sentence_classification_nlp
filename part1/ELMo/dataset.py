import random
import torch
from torch.utils.data import Dataset
import numpy as np

class LanguageModelDataset(Dataset):
    """
    Args:
        data (list): List of samples.
        sentence_padding (int): Index used to pad sequences to the same length.
        char_padding (int): Index used to pad character list to the same length.
        sentence_padded_len (int): Max sentence length.
        char_padded_len (int): Max character length.
    """
    
    def __init__(self, data, sentence_padding, char_padding, sentence_padded_len=64, char_padded_len=10):
        super(LanguageModelDataset, self).__init__()
        self.data = data
        self.sentence_padded_len = sentence_padded_len
        self.sentence_padding = sentence_padding
        self.char_padded_len = char_padded_len
        self.char_padding = char_padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        return data

    def collate_fn(self, datas):
        batch = {}
        
        forward_char_lens = [max([len(char_list) for char_list in data['forward']['X']]) for data in datas]
        forward_padded_len = max(forward_char_lens)
        backward_char_lens = [max([len(char_list) for char_list in data['backward']['X']]) for data in datas]
        backward_padded_len = max(backward_char_lens)
        batch['forward_Y_lens'] = [len(data['forward']['Y']) for data in datas]
        batch['backward_Y_lens'] = [len(data['backward']['Y']) for data in datas]
        
        batch['forward_X'] = torch.tensor(
            [pad_char(data['forward']['X'],
                      sentence_len=self.sentence_padded_len,
                      char_len=forward_padded_len, 
                      padding=self.char_padding)
             for data in datas]
        )
        batch['forward_Y'] = torch.tensor(
            [pad_sentence(data['forward']['Y'], 
                          sentence_len=self.sentence_padded_len, 
                          padding=self.sentence_padding)
             for data in datas]
        )
        
        batch['backward_X'] = torch.tensor(
            [pad_char(data['backward']['X'], 
                      sentence_len=self.sentence_padded_len,
                      char_len=backward_padded_len, 
                      padding=self.char_padding)
             for data in datas]
        )
        batch['backward_Y'] = torch.tensor(
            [pad_sentence(data['backward']['Y'], 
                          sentence_len=self.sentence_padded_len, 
                          padding=self.sentence_padding)
             for data in datas]
        )
        return batch
    
def pad_char(arr, sentence_len, char_len, padding):
    """ Pad `arr` to ['sentence_len', char_len'] with padding.
    Example:
        pad_to_len([[1, 2, 3], [1, 2]], 3, 3, -1) == 
            [[ 1,  2,  3],
             [ 1,  2, -1],
             [-1, -1, -1]]
    Args:
        arr (list): List of int.
        sentence_len (int)
        char_len (int)
        padding (int): Integer used to pad.
    """
    arr_len = len(arr) # sentence length
    if arr_len < sentence_len:
        arr += [[padding]] * (sentence_len-arr_len)
    for i in range(len(arr)):
        if len(arr[i]) < char_len:
            arr[i] = list(np.pad(arr[i], (0, char_len-len(arr[i])), 'constant', constant_values=(padding)))
    return arr

def pad_sentence(arr, sentence_len, padding):
    """ Pad `arr` to `sentence_len` with padding if `len(arr) < padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
    Args:
        arr (list): List of int.
        sentence_len (int)
        padding (int): Integer used to pad.
    """
    arr_len = len(arr)
    if arr_len < sentence_len:
        arr = list(np.pad(arr, (0, sentence_len-arr_len), 'constant', constant_values=(padding)))
    return arr