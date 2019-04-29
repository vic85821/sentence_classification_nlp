import numpy as np
import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification


class Embedder:
    """
    The class responsible for loading a pre-trained BERT model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, device, n_ctx_embs, ctx_emb_dim, pretrain):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.device = device
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        self.tokenizer = BertTokenizer.from_pretrained(pretrain)
        self.model = BertForSequenceClassification.from_pretrained(pretrain)
        self.model.eval()
        self.model.to(self.device)
        
    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        wordpiece_tokens = []
        for sentence in sentences:
            sentence =  ["[CLS]"] + sentence + ["[SEP]"]
            tokens = self.tokenizer.tokenize(sentence)
            wordpiece_tokens.append(tokens)
            
        max_sent_len = min(max(map(len, wordpiece_tokens)), max_sent_len)
        ctx_emb = np.zeros((len(sentences), 1, self.n_ctx_embs, self.ctx_emb_dim), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            sample = ["[CLS]"] + sentence + ["[SEP]"]
            if len(sample) < max_sent_len:
                sample += ['[PAD]'] * (max_sent_len-len(sample))
            
            for j, word in enumerate(sample):
                token = self.tokenizer.tokenize(word)
                idx = self.tokenizer.convert_tokens_to_ids(token)
                tokens_tensor = torch.LongTensor([idx]).to(self.device)
                with torch.no_grad():
                    encoded_layers, _ = self.model(tokens_tensor)
                
                for k in range(self.n_ctx_embs):
                    ctx_emb[i, j, k] = encoded_layers[k].squeeze().sum(dim=0).cpu().numpy() \
                    if len(encoded_layers[k].squeeze().size()) == 2 else encoded_layers[k].squeeze().cpu().numpy()

        return ctx_emb[:, 1:-1]