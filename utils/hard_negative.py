import torch
from copy import deepcopy

class HardNegativeMiner:
    def __init__(self, rate):
        self.rate = rate

        self.cache = None
        self.worst_loss = 0
        self.idx = 0

    def update_cache(self, meter, data):
        loss = float(meter['loss'].data.cpu().numpy())
        if loss > self.worst_loss:
            self.cache = {
                'images': data['images'].numpy(),
                'masks': data['masks'].numpy(),
            }
            self.worst_loss = loss
        self.idx += 1

    def need_iter(self):
        return self.idx >= self.rate

    def invalidate_cache(self):
        self.worst_loss = 0
        self.cache = None
        self.idx = 0

    def get_cache(self):
        return {
            'images': torch.tensor(self.cache['images']),
            'masks': torch.tensor(self.cache['masks']),
        }
