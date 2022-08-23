import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import json
import itertools
import torch
import torch.utils.data as data
import random
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np

from noisyTransform import AddNoise, SaltPepperNoise, GaussianNoise

num_epochs = 2000
learning_rate = 1e-5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

trainTransforms = transforms.Compose([
        # transforms.RandomResizedCrop(64, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.RandomCrop(68),
        transforms.CenterCrop(size=64),
        # GaussianNoise(var=0.05),
        # transforms.Resize([64, 64]),
    ])

imagebuffer = {}
class Dataset(data.Dataset):
    def __init__(self, path, transform = None, percent = 1.0, seed = 1):
        self.path = path
        self.transform = transform
        self.image = []

        # get label
        self.class_names = sorted(os.listdir(path))
        self.names2index = {v: k for k, v in enumerate(self.class_names)}
        setup_seed(seed)
        # read-in sequence and label
        for label in self.class_names:
            labelPath = os.path.join(path,label)
            for imageName in os.listdir(labelPath):
                if imageName.endswith(".jpg"):
                    imagePath = os.path.join(labelPath, imageName)
                    self.image.append((imagePath, self.names2index[label]))


    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        path,label = self.image[item]
        if path not in imagebuffer:
            imagebuffer[path] = Image.open(path)
        img = self.transform(imagebuffer[path])
        return img.to(device), label

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        kernel_size = 7
        stride = 2
        padding = 3
        pooling_kernel_size = 3
        pooling_stride = 2
        pooling_padding = 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=pooling_kernel_size,
                     stride=pooling_stride,
                     padding=pooling_padding,
                     return_indices=True)

        self.encoder2 = nn.Sequential(
                nn.Conv2d(64, 64,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
                nn.ReLU()
            )
        self.maxpool2 = nn.MaxPool2d(kernel_size=pooling_kernel_size,
                     stride=pooling_stride,
                     padding=pooling_padding,
                     return_indices=True)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 256,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride),
                      padding=(padding, padding), bias=False),
            nn.ReLU()
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                     stride=pooling_stride,
                                     padding=pooling_padding,
                                     return_indices=True)

        self.maxunpool0 = nn.MaxUnpool2d(kernel_size=pooling_kernel_size,
                                         stride=pooling_stride,
                                         padding=pooling_padding)
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(256, 64,
                               kernel_size=(kernel_size, kernel_size),
                               stride=2,
                               padding=2, bias=False, output_padding=1),
            nn.ReLU()
        )

        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=pooling_kernel_size,
                       stride=pooling_stride,
                       padding=pooling_padding)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64,
                               kernel_size=(kernel_size, kernel_size),
                               stride=2,
                               padding=2, bias=False, output_padding=1),
            nn.ReLU()
        )
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=pooling_kernel_size,
                                         stride=pooling_stride,
                                         padding=pooling_padding)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1,
                               kernel_size=(kernel_size, kernel_size),
                               stride=2,
                               padding=2, bias=False, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder1(x)
        x, pos1 = self.maxpool1(x)
        x = self.encoder2(x)
        x, pos2 = self.maxpool2(x)
        # x = self.encoder3(x)
        # x, pos3 = self.maxpool3(x)

        # x = self.maxunpool0(x, pos3)
        # x = self.decoder0(x)
        x = self.maxunpool1(x, pos2)
        x = self.decoder1(x)
        x = self.maxunpool2(x, pos1)
        x = self.decoder2(x)

        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def draw_image(x):
    from PIL import Image
    img = Image.fromarray(x)
    img.show()

BATCH_SIZE = 32
train_data = Dataset('./data/SOC-128/train', transform=trainTransforms)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

model = autoencoder().cuda()
# model.load_state_dict(torch.load('./best_model/conv_autoencoder.pth'))
# model = torch.load('./best_model/conv_autoencoder.pth')
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        # draw_image(img.cpu().detach().numpy()[0][0] * 255)
        output = model(img)
        # draw_image(output.cpu().detach().numpy()[0][0]*255)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))

torch.save(model.state_dict(), './best_model/conv_autoencoder.pth')