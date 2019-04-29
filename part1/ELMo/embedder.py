import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from ELMo.modules import *
from ELMo.dataset import LanguageModelDataset

class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, device, n_ctx_embs, ctx_emb_dim, charmap_path, wordmap_path, model_path):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.device = device
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        self.n = 0
        self.charCNN_forward_feat = []
        self.charCNN_backward_feat = []
        self.of1_forward_feat = []
        self.of1_backward_feat = []
        self.of2_forward_feat = []
        self.of2_backward_feat = []
        
        checkpoint = torch.load(model_path, map_location=self.device)
        with open(charmap_path, 'rb') as f:
            self.charmap = pickle.load(f)
        with open(wordmap_path, 'rb') as f:
            self.wordmap = pickle.load(f)
        
        self.model = ELMo(len(self.charmap), self.charmap['<PAD>'], len(self.wordmap))
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        self.model.charCNN.register_forward_hook(self.charCNN_hook)
        self.model.m_forward_fc_1.register_forward_hook(self.of1_forward_hook)
        self.model.m_backward_fc_1.register_forward_hook(self.of1_backward_hook)
        self.model.m_forward_fc_2.register_forward_hook(self.of2_forward_hook)
        self.model.m_backward_fc_2.register_forward_hook(self.of2_backward_hook)
        
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
            
        max_sent_len = min(max(map(len, sentences))+2, max_sent_len)
        ctx_emb = np.zeros((len(sentences), max_sent_len, self.n_ctx_embs, self.ctx_emb_dim), dtype=np.float32)
        
        self.n = 0
        self.charCNN_forward_feat = []
        self.charCNN_backward_feat = []
        self.of1_forward_feat = []
        self.of1_backward_feat = []
        self.of2_forward_feat = []
        self.of2_backward_feat = []
        
        data = []
        for sentence in sentences:
            sample = ["<BOS>"] + sentence + ["<EOS>"]
            sample = sample[:max_sent_len]
            
            char_list = [list(word) if ((word != '<BOS>') and (word != '<EOS>')) else [word] for word in sample]
            unk_word = self.wordmap["<UNK>"]
            unk_char = self.charmap["<UNK>"]
            
            for j in range(len(sample)):
                sample[j] = self.wordmap[sample[j]] if sample[j] in self.wordmap else unk_word
            for j in range(len(char_list)):
                for k in range(len(char_list[j])):
                    char_list[j][k] = self.charmap[char_list[j][k]] if char_list[j][k] in self.charmap else unk_char
            
            forward = {"X": char_list, 
                       "Y": sample}
            backward = {"X": char_list[::-1], 
                        "Y": sample[::-1]}
            
            data.append({"forward": forward, "backward": backward})
        
        dataset = LanguageModelDataset(data, self.wordmap['<PAD>'], self.charmap['<PAD>'])
        dataloader = DataLoader(dataset=dataset, 
                                batch_size=len(sentences),
                                collate_fn=dataset.collate_fn,
                                shuffle=False,
                                num_workers=4)
        
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                forward_X = batch['forward_X']
                forward_Y = batch['forward_Y']
                backward_X = batch['backward_X']
                backward_Y = batch['backward_Y']
                
                forward, backward = self.model.forward(forward_X.to(self.device),
                                                       forward_Y.to(self.device),
                                                       backward_X.to(self.device),
                                                       backward_Y.to(self.device))
                
        for i in range(len(sentences)):
            sent_len = len(sentences[i])+2
            ctx_emb[i, :, 0, :512] = self.charCNN_forward_feat[i][:max_sent_len]
            ctx_emb[i, :sent_len, 0, 512:] = self.charCNN_backward_feat[i][:sent_len][::-1]
            ctx_emb[i, sent_len:, 0, 512:] = self.charCNN_backward_feat[i][sent_len:max_sent_len]
            ctx_emb[i, :, 1, :512] = self.of1_forward_feat[i][:max_sent_len]
            ctx_emb[i, :sent_len, 1, 512:] = self.of1_backward_feat[i][:sent_len][::-1]
            ctx_emb[i, sent_len:, 1, 512:] = self.of1_backward_feat[i][sent_len:max_sent_len]
            ctx_emb[i, :, 2, :512] = self.of2_forward_feat[i][:max_sent_len]
            ctx_emb[i, :sent_len, 2, 512:] = self.of2_backward_feat[i][:sent_len][::-1]
            ctx_emb[i, sent_len:, 2, 512:] = self.of2_backward_feat[i][sent_len:max_sent_len]
            
            
        return ctx_emb[:, 1:-1]
    
    def charCNN_hook(self, module, input, output):
        if self.n % 2 == 0:
            self.charCNN_forward_feat.extend(output.cpu().numpy())
        else:
            self.charCNN_backward_feat.extend(output.cpu().numpy())
        self.n += 1

    def of1_forward_hook(self, module, input, output):
        self.of1_forward_feat.extend(output.transpose(1, 0).cpu().numpy())

    def of1_backward_hook(self, module, input, output):
        self.of1_backward_feat.extend(output.transpose(1, 0).cpu().numpy())

    def of2_forward_hook(self, module, input, output):
        self.of2_forward_feat.extend(output.transpose(1, 0).cpu().numpy())

    def of2_backward_hook(self, module, input, output):
        self.of2_backward_feat.extend(output.transpose(1, 0).cpu().numpy())
