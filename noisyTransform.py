from torch import nn
import torch
from functools import reduce
import random
import numpy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, var = 0.001):
        super().__init__()
        self.var = var

    def forward(self, x):
        image = numpy.array(x)
        noise = numpy.random.normal(0, self.var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = numpy.clip(out, low_clip, 1.0)
        return torch.from_numpy(out).float().cuda()
