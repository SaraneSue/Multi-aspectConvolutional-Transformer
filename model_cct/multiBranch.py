from model_cct.cct import cct_6_7x3_64_sine
from model_cct.vit import vit_4_16_64_sine
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBranchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc = nn.Linear(16 * 29 * 29, 128)
        self.relu = F.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, 16*29*29)
        x = self.fc(x)
        return x

class MultiBranch(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.cct = cct_6_7x3_64_sine(num_classes = num_classes, n_input_channels = 1, need_fc=False)
        self.sequenceLen = 4
        self.cnns = nn.ModuleList([MultiBranchCNN() for _ in range(self.sequenceLen)])
        self.vits = nn.ModuleList([vit_4_16_64_sine(need_fc=False) for _ in range(self.sequenceLen)])
        self.fc1 = nn.Linear(1280, num_classes)

    def forward(self, x):
        cctx = [self.cct(x)]
        shape = list(x.shape)
        shape[1] = 1
        shape = tuple(shape)
        cnnsx = [self.cnns[i](torch.reshape(x[:,i], shape)) for i in range(self.sequenceLen)]
        vitsx = [self.vits[i](torch.reshape(x[:,i], shape)) for i in range(self.sequenceLen)]
        x = torch.cat(cctx + cnnsx + vitsx, dim=1)
        x = self.fc1(x)
        return x