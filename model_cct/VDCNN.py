import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VDCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        ## view1
        self.view1_conv1 = nn.Conv2d(1, 16, 6)
        ## view2
        self.view2_conv1 = nn.Conv2d(1, 16, 6)
        self.view12 = nn.Conv2d(32, 64, 5)
        ## view3
        self.view3_conv1 = nn.Conv2d(1, 16, 6)
        self.view3_conv2 = nn.Conv2d(16, 32, 5)
        self.view123 = nn.Conv2d(96, 128, 6)
        ## view4
        self.view4_conv1 = nn.Conv2d(1, 16, 6)
        self.view4_conv2 = nn.Conv2d(16, 32, 5)
        self.view4_conv3 = nn.Conv2d(32, 64, 6)
        self.view1234 = nn.Conv2d(192, 256, 5)

        self.fc1 = nn.Linear(256*3*3, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout2d(p=0.5)
        self.softmax = nn.Softmax(dim=0)
        self.relu = F.relu
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = 1
        shape = tuple(shape)

        x1 = torch.reshape(x[:,0], shape)
        x1 = self.view1_conv1(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        x2 = torch.reshape(x[:,1], shape)
        x2 = self.view2_conv1(x2)
        x2 = self.relu(x2)
        x2 = self.pool(x2)
        x2 = torch.cat((x1, x2), dim=1)
        x2 = self.view12(x2)
        x2 = self.relu(x2)
        x2 = self.pool(x2)

        x3 = torch.reshape(x[:,2], shape)
        x3 = self.view3_conv1(x3)
        x3 = self.relu(x3)
        x3 = self.pool(x3)
        x3 = self.view3_conv2(x3)
        x3 = self.relu(x3)
        x3 = self.pool(x3)
        x3 = torch.cat((x2, x3), dim=1)
        x3 = self.view123(x3)
        x3 = self.relu(x3)
        x3 = self.pool(x3)

        x4 = torch.reshape(x[:,3], shape)
        x4 = self.view4_conv1(x4)
        x4 = self.relu(x4)
        x4 = self.pool(x4)
        x4 = self.view4_conv2(x4)
        x4 = self.relu(x4)
        x4 = self.pool(x4)
        x4 = self.view4_conv3(x4)
        x4 = self.relu(x4)
        x4 = self.pool(x4)
        x4 = torch.cat((x3, x4), dim=1)
        x4 = self.view1234(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4.view(-1, 256*3*3)
        x4 = self.fc1(x4)
        x4 = self.fc2(x4)
        x4 = self.softmax(x4)
        return x4


