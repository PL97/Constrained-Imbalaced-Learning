import numpy as np
import torch

class Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, n):
        self.idx = list(range(n))
    
    def __iter__(self):
        for i in np.random.permutation(self.idx):
            yield i
    
    def __len__(self):
        return len(self.idx)
    
    
def BatchSampler(n, bs, drop_last=True):
    return torch.utils.data.sampler.BatchSampler(Sampler(n), batch_size=bs, drop_last=True)
    
    
if __name__ == "__main__":
    pass
    # test_sampler = BatchSampler(10, 2)
    # for t in test_sampler:
    #     print(t)