import torch
import numpy as np
from torch.utils.data import Dataset


# make fake FNC data
class FakeFNC(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=150,
                 window_size=40,
                 N_sz=1024,
                 sz_mu=0,
                 sz_std=3,
                 sz_low=40,
                 sz_high=80,
                 N_hc=1024,
                 hc_mu=0.1,
                 hc_std=1,
                 hc_low=20,
                 hc_high=60):
        self.N_subjects = N_sz + N_hc

        upper_triangle_size = int(N_components*(N_components - 1)/2)
        self.dim = upper_triangle_size
        self.seqlen = N_timepoints - window_size

        sz_FNCs = torch.randn(
            N_sz,  N_timepoints - window_size, upper_triangle_size)*sz_std + sz_mu

        sz_age = torch.randint(sz_low, sz_high, (N_sz, 1)).float()

        hc_FNCs = torch.randn(
            N_sz,  N_timepoints - window_size, upper_triangle_size)*hc_std + hc_mu

        hc_age = torch.randint(hc_low, hc_high, (N_sz, 1)).float()

        self.data = np.concatenate([sz_FNCs, hc_FNCs], 0)
        self.age = np.concatenate([sz_age, hc_age], 0)

    def __len__(self):
        return self.N_subjects

    def __getitem__(self, k):
        return torch.from_numpy(self.data[k, ...]), torch.from_numpy(self.age[k])
