import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, OpenAIGPTTokenizer

class Part1Dataset(Dataset):
    def __init__(self, data, bert_pretrain, bert_do_lower_case):
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrain,
                                                       do_lower_case=bert_do_lower_case)
        self._data = [{
            'Id': d['Id'],
            'text_orig': d['text'],
            'text_token_id': self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(d['text'])),
            'label': int(d['label']) - 1
        } for d in tqdm(data, desc='[*] Indexizing', dynamic_ncols=True)]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def create_collate_fn(max_sent_len):
    word_pad_idx = 0

    # This recursive version can account of arbitrary depth. However, the required stack
    # allocation may harm performance.
    # def pad(batch, max_len, padding):
    #     l, p = max_len[0], padding[0]
    #     for i, b in enumerate(batch):
    #         batch[i] = b[:l]
    #         batch[i] += [[p] for _ in range(l - len(b))]
    #         if len(max_len) > 1:
    #             batch[i] = pad(batch[i], max_len[1:], padding[1:])
    #
    #     return batch

    def pad(batch, max_len, padding):
        for i, b in enumerate(batch):
            batch[i] = b[:max_len]
            batch[i] += [padding for _ in range(max_len - len(b))]

        return batch

    def collate_fn(batch):
        Id = [b['Id'] for b in batch]
        text_orig = [b['text_orig'] for b in batch]
        text_token_id = [b['text_token_id'] for b in batch]
        label = [b['label'] for b in batch]

        max_len = min(max(map(len, text_token_id)), max_sent_len)
        text_token_id = pad(text_token_id, max_len, word_pad_idx)

        text_token_id = torch.tensor(text_token_id)
        text_pad_mask = text_token_id != word_pad_idx
        label = torch.tensor(label)

        return {
            'Id': Id,
            'text_token_id': text_token_id,
            'text_pad_mask': text_pad_mask,
            'label': label
        }

    return collate_fn


def create_data_loader(dataset, max_sent_len, batch_size, n_workers, shuffle=True):
    collate_fn = create_collate_fn(max_sent_len)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=n_workers,
                             collate_fn=collate_fn)

    return data_loader
