from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
from .utils.helpers import pe_check
from .resnet import resnet18
from .DSnet import DSnet
from PIL import Image

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
    'cct_7_3x1_32':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth',
    'cct_7_3x1_32_sine_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 need_fc=True,
                 sequence_len=4,
                 arch="",
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.sequence_len = sequence_len
        self.n_output_channels = embedding_dim
        self.class_num = num_classes
        self.conv = nn.Conv2d(embedding_dim, num_classes,
                              kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

        if arch.startswith("cct_6"):
            self.tokenizer = resnet18()
            # self.tokenizer = DSnet()
        else:
            self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                       n_output_channels=embedding_dim,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       pooling_kernel_size=pooling_kernel_size,
                                       pooling_stride=pooling_stride,
                                       pooling_padding=pooling_padding,
                                       max_pool=True,
                                       activation=nn.ReLU,
                                       n_conv_layers=n_conv_layers,
                                       conv_bias=False)

        self.classifier = TransformerClassifier(
            # sequence_length=4 if arch.startswith("cct_4") else self.tokenizer.sequence_length(n_channels=n_input_channels,
            #                                                height=img_size,
            #                                                width=img_size),
            sequence_length=4,
            embedding_dim=embedding_dim,
            seq_pool=False,
            use_cls_token=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            need_fc=need_fc
        )

    def forward(self, x):
        x = self.tokenizer(x)
        # img = Image.fromarray(x.cpu().detach().numpy()[0][0] * 255)
        # img.show()
        x = self.classifier(x)
        # return x

        allx = x
        batch_size = x.shape[0]
        vote = None
        novote = []
        for i in range(0, self.sequence_len):
            x = allx[:, i:i + 1].reshape((batch_size, self.n_output_channels, 1, 1))
            x = self.conv(x).reshape((batch_size, self.class_num))
            novote.append(x.cpu().detach().numpy())
            if vote is None:
                vote = x
            else:
                vote += x
        novote = torch.Tensor(novote).transpose(0,1)
        # return self.conv(vote/4).reshape((batch_size, self.class_num)), novote
        return vote/self.sequence_len

def _cct(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                arch=arch,
                *args, **kwargs)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            state_dict = pe_check(model, state_dict)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return model


def cct_2(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=2, num_heads=1, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)
# cct4 embedding_dim=128

def cct_6(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


@register_model
def cct_2_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_2('cct_2_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_2_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_2('cct_2_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_4('cct_4_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_4('cct_4_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)

@register_model
def cct_7_7x2_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=102,
                  *args, **kwargs):
    return cct_7('cct_7_7x2_224', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224_sine(pretrained=False, progress=False,
                       img_size=224, positional_embedding='sine', num_classes=102,
                       *args, **kwargs):
    return cct_7('cct_7_7x2_224_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)

@register_model
def cct_6_7x3_224_sine(pretrained=False, progress=False,
                       img_size=224, positional_embedding='sine', num_classes=10,
                       *args, **kwargs):
    return cct_6('cct_6_7x3_224_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=3,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)

@register_model
def cct_6_7x3_64_sine(pretrained=False, progress=False,
                       img_size=64, positional_embedding='sine', num_classes=10,need_fc=False,
                       *args, **kwargs):
    return cct_6('cct_6_7x3_64_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=3,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 need_fc=need_fc,
                 *args, **kwargs)

@register_model
def cct_2_7x3_64_sine(pretrained=False, progress=False,
                       img_size=64, positional_embedding='sine', num_classes=10,need_fc=False,
                       *args, **kwargs):
    return cct_2('cct_2_7x3_64_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=3,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 need_fc=need_fc,
                 *args, **kwargs)

@register_model
def cct_4_7x3_64_sine(pretrained=False, progress=False,
                       img_size=64, positional_embedding='sine', num_classes=10,need_fc=False,
                       *args, **kwargs):
    return cct_4('cct_4_7x3_64_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=3,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 need_fc=need_fc,
                 *args, **kwargs)

@register_model
def cct_6_3x6_64_sine(pretrained=False, progress=False,
                       img_size=64, positional_embedding='sine', num_classes=10,need_fc=False,
                       *args, **kwargs):
    return cct_6('cct_6_3x6_64_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=6,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 need_fc=need_fc,
                 *args, **kwargs)