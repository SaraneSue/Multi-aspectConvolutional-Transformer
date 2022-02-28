from torch import nn
import torch
from functools import reduce
import random

class AddNoise(nn.Module):
    def __init__(self, noise_rate = 0.1):
        super().__init__()
        self.noise_rate = noise_rate

    def forward(self, x):
        shape = x.shape
        length = reduce(lambda a, b: a*b, shape)
        x = torch.reshape(x, tuple([length]))
        noiseidxes = random.sample(range(length), k=int(length*self.noise_rate))
        x[noiseidxes] = 0
        x = torch.reshape(x, shape)
        return x