import numpy as np
import torch
import utils
from torch.utils.data import Dataset

class dataset_SVI_base(Dataset):
    def __init__(self,x_id,y):
        self.x = x_id
        self.y = y
        self.length = len(self.y)

    def __getitem__(self, index):
        return self.x[index,:],self.y[index]
    
    def __len__(self):
        return self.length 

class dataset_diffusion_base(Dataset):

    
    def __getitem__(self, index):
        return self.x[index,:],self.y[index]
    
    def __len__(self):
        return self.length 