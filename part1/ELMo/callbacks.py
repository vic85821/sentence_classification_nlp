import math
import json


class Callback:
    def __init__():
        pass

    def on_epoch_end(log_train, log_valid, model):
        pass


class MetricsLogger(Callback):
    def __init__(self, log_dest):
        self.history = {
            'train': [],
            'valid': []
        }
        self.log_dest = log_dest
        self.n = 0

    def on_epoch_end(self, log, model):
        log['epoch'] = model.epoch
        if self.n % 2 == 0:
            self.history['train'].append(log)
        else:
            self.history['valid'].append(log)
        self.n += 1
        with open(self.log_dest, 'w') as f:
            json.dump(self.history, f, indent='    ')


class ModelCheckpoint(Callback):
    def __init__(self, filepath,
                 monitor='loss', verbose=0, mode='min', early_stop=0):
        self._filepath = filepath
        self._verbose = verbose
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else - math.inf
        self._mode = mode
        self._early_stop = early_stop
        self._count = 0
        
    def on_epoch_end(self, log_valid, model):
        score = log_valid[self._monitor]
        if self._mode == 'min':
            if score < self._best:
                self._best = score
                self._count = 0
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)
            else:
                self._count += 1
                
        elif self._mode == 'max':
            if score > self._best:
                self._best = score
                self._count = 0
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)
            else:
                self._count += 1
                
        elif self._mode == 'all':
            model.save('{}.{}'.format(self._filepath, model.epoch))
        
        if (self._count == self._early_stop) and (self._early_stop != 0):
            return True
        else:
            return False