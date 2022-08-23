import torch
import torch.nn as nn
import torch.nn.functional as F

class dwconv(nn.Module):
    def __init__(self,in_ch,out_ch,dcpadding=3):
        super(dwconv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=7,
                                    stride=2,
                                    padding=dcpadding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DSnet(nn.Module):
    def __init__(self,sequence_len = 4):
        super(DSnet, self).__init__()
        pooling_kernel_size = 3
        pooling_stride = 2
        pooling_padding = 1
        self.sequence_len = sequence_len
        self.conv1 = nn.Sequential(
            dwconv(1,31),
            nn.ReLU()
        )
        self.maxpool=nn.MaxPool2d(kernel_size=pooling_kernel_size,
                     stride=pooling_stride,
                     padding=pooling_padding)
        self.conv2 = nn.Sequential(
            dwconv(64, 128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            dwconv(192, 256, 0),
            nn.ReLU()
        )
        self.flattener = nn.Flatten(2, 3)


    def forward(self,x):
        # sequence length
        if x.shape[1] != self.sequence_len:
            x = self.conv_layers(x)
            x = self.flattener(x).transpose(-2, -1)
            return x
        allx = x
        res = []
        for i in range(0, self.sequence_len):
            x = allx[:, i:i + 1]
            # dwconv1
            x1 = self.conv1(x)
            xr1 = F.interpolate(x, size=(32, 32), mode='bicubic', align_corners=False)
            x2 = torch.cat((xr1,x1),1)
            # maxpool
            x3 = self.maxpool(x2)
            xr3 = F.interpolate(x, size=(16, 16), mode='bicubic', align_corners=False)
            x1r3 = F.interpolate(x1, size=(16, 16), mode='bicubic', align_corners=False)
            x4 = torch.cat((xr3, x1r3, x3), 1)
            # dwconv2
            x5 = self.conv2(x4)
            xr5 = F.interpolate(x, size=(8, 8), mode='bicubic', align_corners=False)
            x1r5 = F.interpolate(x1, size=(8, 8), mode='bicubic', align_corners=False)
            x3r5 = F.interpolate(x3, size=(8, 8), mode='bicubic', align_corners=False)
            x6 = torch.cat((xr5, x1r5, x3r5, x5), 1)
            # dwconv3
            x7 = self.conv3(x6)

            out = self.flattener(x7).transpose(-2, -1)
            res.append(out)

        return torch.cat(tuple(res), dim=1)