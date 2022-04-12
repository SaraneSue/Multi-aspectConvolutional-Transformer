from torch import nn
import torch
from functools import reduce
import random
import numpy

class AddNoise(nn.Module):
    def __init__(self, noise_rate = 0.1):
        super().__init__()
        self.noise_rate = noise_rate

    def forward(self, x):
        shape = x.shape
        length = reduce(lambda a, b: a*b, shape)
        x = torch.reshape(x, tuple([length]))
        # numpy.random.seed(100)
        # random.seed(100)
        noiseidxes = random.sample(range(length), k=int(length*self.noise_rate))
        noise = numpy.random.uniform(size=int(length*self.noise_rate))
        noise = noise.astype('float32')
        x[noiseidxes] = torch.from_numpy(noise)
        x = torch.reshape(x, shape)
        return x

class SaltPepperNoise(nn.Module):
    def __init__(self, noise_rate = 0.1):
        super().__init__()
        self.noise_rate = noise_rate

    def forward(self, x):
        shape = x.shape
        length = reduce(lambda a, b: a * b, shape)
        x = torch.reshape(x, tuple([length]))
        # numpy.random.seed(100)
        # random.seed(100)
        noiseidxes = random.sample(range(length), k=int(length * self.noise_rate))
        noise = numpy.random.randint(2, size=int(length * self.noise_rate))
        noise = noise.astype('float32')
        x[noiseidxes] = torch.from_numpy(noise)
        x = torch.reshape(x, shape)
        return x

class GaussianNoise(nn.Module):
    def __init__(self, noise_rate = 0.1):
        super().__init__()
        self.noise_rate = noise_rate

    def forward(self, x):
        shape = x.shape
        length = reduce(lambda a, b: a * b, shape)
        x = torch.reshape(x, tuple([length]))
        # numpy.random.seed(100)
        # random.seed(100)
        noiseidxes = random.sample(range(length), k=int(length * self.noise_rate))
        noise = numpy.random.normal(0, 1, size=int(length * self.noise_rate)).clip(0, 1)
        noise = noise.astype('float32')
        x[noiseidxes] = torch.from_numpy(noise)
        x = torch.reshape(x, shape)
        return x