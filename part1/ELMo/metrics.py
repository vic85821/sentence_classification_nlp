import torch

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass

class Perplexity(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self):
        self.name = 'Perplexity'

    def reset(self):
        self.score = 0
        self.n = 0
        
    def update(self, score):
        """
        Args:
            loss: cross entropy loss output
        """
        self.score = score
        
    def get_score(self):
        return self.score
        
    def print_score(self):
        return '{:.2f}'.format(self.get_score())