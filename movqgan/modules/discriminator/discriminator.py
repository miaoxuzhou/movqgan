import functools
import torch.nn as nn

from movqgan.modules.components.normalization import ActNorm

# 权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__

    # 卷积层初始化，使用均值为0、标准差为0.02的正态分布
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    # BatchNorm层初始化
    # 使用均值为1、标准差为0.02的正态分布，bias初始化为0
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator as in Pix2Pix --->
    see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(
        self, 
        input_nc=3, 
        ndf=64, 
        n_layers=3, 
        use_actnorm=False
    ):
        """
        Constructs a PatchGAN discriminator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the channel multiplier, also the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm     -- whether to use actnorm (True) or batchnorm (False)
        """
        super(NLayerDiscriminator, self).__init__()

        # 使用哪种归一化层
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm

        # 是否使用偏置
        # 如果是BatchNorm2d，则无需使用偏置，因为BatchNorm2d有 affine 参数
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # 设置卷积核的大小和填充
        kw = 4
        padw = 1

        # 第一层卷积
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]

        # 记录通道数
        nf_mult = 1
        nf_mult_prev = 1

        # 逐层增加输出通道数
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 倒数第二层卷积
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 最后一层卷积，输出1个通道，输出维度为 
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class NLayerDiscriminatorLinear(nn.Module):
    """
    Defines a fully connected discriminator using linear layers.
    """
    def __init__(
        self, 
        input_nc=3, 
        ndf=64, 
        n_layers=3
    ):
        """
        Constructs a fully connected discriminator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the channel multiplier
            n_layers (int)  -- the number of linear layers in the discriminator
        """
        super(NLayerDiscriminatorLinear, self).__init__()
        
        # 第一层线性层
        sequence = [
            nn.Linear(input_nc, ndf), 
            nn.LeakyReLU(0.2, True)
        ]

        # 基准通道数
        nf_mult = 1

        # 逐层增加输出通道数
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Linear(ndf * nf_mult_prev, ndf * nf_mult, bias=True),
                nn.LeakyReLU(0.2, True)
            ]

        # 倒数第二层线性层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Linear(ndf * nf_mult_prev, ndf * nf_mult, bias=True),
            nn.LeakyReLU(0.2, True)
        ]

        # 最后一层线性层，输出1个通道
        sequence += [
            nn.Linear(ndf * nf_mult, 1)
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)