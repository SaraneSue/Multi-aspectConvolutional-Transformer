import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 5)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 4)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu = F.relu
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        batchsize = x.shape[0]
        x = x.view(batchsize, 128)
        return x

class BiLSTM(nn.Module):
    def __init__(self, num_classes = 10):
        super(BiLSTM, self).__init__()
        self.input_size = 128
        self.n_hidden = 1024
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.n_hidden, bidirectional=True, num_layers=2)
        # fc
        self.fc = nn.Linear(self.n_hidden * 2, self.num_classes)

    def forward(self, X):
        # input : [seq_len, batch_size, feature_size]
        batch_size = X.shape[1]

        hidden_state = torch.randn(2*2, batch_size, self.n_hidden).to(device)
        cell_state = torch.randn(2*2, batch_size, self.n_hidden).to(device)

        outputs, (_, _) = self.lstm(X, (hidden_state, cell_state)) # [seq_len, batch_size, n_hidden*2]
        outputs = torch.mean(outputs, dim=0)  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model

class BCRN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.sequenceLen = 4
        self.num_classes = num_classes
        self.layers = nn.ModuleList([ConvBlock() for _ in range(self.sequenceLen)])
        self.bilstm = BiLSTM(num_classes=self.num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = 1
        shape = tuple(shape)
        cnns = torch.stack([self.layers[i](torch.reshape(x[:,i], shape)) for i in range(self.sequenceLen)])
        x = self.bilstm(cnns)
        x = self.softmax(x)
        return x
