import torch
import torch.nn as nn
import torch.nn.functional as F
from model_cct.utils.transformers import TransformerClassifier
from model_cct.utils.tokenizer import Tokenizer
import math
class BranchTokenizer(nn.Module):
    def __init__(self,
                 n_input_channels, embedding_dim, kernel_size,
                stride, padding, pooling_kernel_size, pooling_stride, pooling_padding):
        super(BranchTokenizer, self).__init__()
        self.n_input_channels = n_input_channels
        self.tokenizer = nn.Sequential(
            nn.Conv2d(n_input_channels, embedding_dim,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride),
                      padding=(padding, padding), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size,
                         stride=pooling_stride,
                         padding=pooling_padding)
        )

    def output_size(self, input_size):
        return self.tokenizer.forward(torch.zeros((32, self.n_input_channels, input_size, input_size))).shape[2]

    def forward(self, x):
        return self.tokenizer(x)

class BranchItem(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=1,
                 num_heads=6,
                 mlp_ratio=4.0,
                 positional_embedding='learnable', transformer_input_cut = [4, 8]):
        super(BranchItem, self).__init__()
        self.transformer_input_cut = transformer_input_cut
        self.tokenizer = BranchTokenizer(n_input_channels, embedding_dim, kernel_size,
                stride, padding, pooling_kernel_size, pooling_stride, pooling_padding)
        self.input_size = input_size

        sequence_len_1 = int(transformer_input_cut[0] * transformer_input_cut[0])
        self.classifier_1 = TransformerClassifier(
            sequence_length=sequence_len_1,
            embedding_dim=embedding_dim,
            seq_pool=False,
            use_cls_token=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            positional_embedding=positional_embedding,
            need_fc=False
        )
        sequence_len_2 = int(transformer_input_cut[1] * transformer_input_cut[1])
        self.classifier_2 = TransformerClassifier(
            sequence_length=sequence_len_2,
            embedding_dim=embedding_dim,
            seq_pool=False,
            use_cls_token=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            positional_embedding=positional_embedding,
            need_fc=False
        )

    def forward(self, x):
        x = self.tokenizer(x)
        # return x
        x = x.transpose(2,1).transpose(2,3)
        xshape = x.shape
        width = x.shape[1]
        # to transformer 1
        width_cnt = int(width/self.transformer_input_cut[0])
        batch_size = x.shape[0]
        channel_size = x.shape[3]
        ally = []
        for i in range(width_cnt):
            for j in range(width_cnt):
                xin = x[:,self.transformer_input_cut[0]*i:self.transformer_input_cut[0]*(i+1),
                                      self.transformer_input_cut[0]*j:self.transformer_input_cut[0]*(j+1),:]\
                    .reshape((batch_size, self.transformer_input_cut[0]*self.transformer_input_cut[0], channel_size))
                y = self.classifier_1(xin)
                ally.append(y)
        x = torch.cat(tuple(ally), dim=1).reshape(xshape)

        ally = []
        width_cnt = int(width/self.transformer_input_cut[1])
        for i in range(width_cnt):
            for j in range(width_cnt):
                xin = x[:,self.transformer_input_cut[1]*i:self.transformer_input_cut[1]*(i+1),
                                      self.transformer_input_cut[1]*j:self.transformer_input_cut[1]*(j+1),:]\
                    .reshape((batch_size, self.transformer_input_cut[1]*self.transformer_input_cut[1], channel_size))
                y = self.classifier_2(xin)
                ally.append(y)
        x = torch.cat(tuple(ally), dim=1).reshape(xshape)

        return x.transpose(2,3).transpose(2,1)

class OneBranch(nn.Module):
    def __init__(self): # middle cnn kernel size
        super(OneBranch, self).__init__()
        self.cnn_transformer_layer = nn.Sequential(
            *[
                BranchItem(input_size=64, num_heads=2, mlp_ratio=1,
                           n_input_channels=1, embedding_dim=32, kernel_size=7, transformer_input_cut=[4,8]),
                BranchItem(input_size=16, num_heads=2, mlp_ratio=1,
                           n_input_channels=32, embedding_dim=64, kernel_size=3, stride=1, padding=1, transformer_input_cut=[4, 8]),
                BranchItem(input_size=8, num_heads=2, mlp_ratio=1,
                           n_input_channels=64, embedding_dim=128, kernel_size=3, stride=1, padding=1, transformer_input_cut=[4, 4]),
            ]
        )
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(128, 256,
                      kernel_size=4,
                      stride=1,
                      padding=0, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn_transformer_layer(x)
        x = self.cnn_layer(x)
        return  x

class MultiViewConformer(nn.Module):
    def __init__(self, n_input_size=64,
                 n_input_channels=1,
                 n_output_channels=256,
                 sequence_len=4,
                 class_num=10):
        super().__init__()
        self.sequence_len = sequence_len
        self.class_num = class_num
        self.n_output_channels = n_output_channels
        self.branch = OneBranch()

        self.classifier = TransformerClassifier(
            sequence_length=sequence_len,
            embedding_dim=n_output_channels,
            seq_pool=False,
            use_cls_token=False,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            num_layers=14,
            num_heads=4,
            mlp_ratio=2.0,
            num_classes=10,
            positional_embedding='learnable',
            need_fc=False,
        )
        self.conv = nn.Conv2d(n_output_channels, 10,
                  kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        allx = x
        res = []
        for i in range(0, self.sequence_len):
            x = allx[:, i:i + 1]
            x = self.branch(x).transpose(2,1)
            xshape = x.shape
            # x = self.flattener(x).transpose(-2, -1)
            res.append(x.reshape(xshape[:3]))
        x = torch.cat(tuple(res), dim=1)
        x = self.classifier(x)
        allx = x

        batch_size = x.shape[0]
        vote = None
        for i in range(0, self.sequence_len):
            x = allx[:, i:i + 1].reshape((batch_size, self.n_output_channels, 1, 1))
            x = self.conv(x).reshape((batch_size, self.class_num))
            if vote is None:
                vote = x
            else:
                vote += x
        return vote/4

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiViewConformer(n_input_size=64, n_input_channels=1, n_output_channels=256).to(device)
    x = model.forward(torch.zeros((32, 4, 64, 64)).to(device))
    print(x.shape)
