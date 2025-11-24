import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from movqgan.util import get_ckpt_path

class ReconstructionLoss(nn.Module):
    """
    Pixel-wise reconstruction loss (L2).
    Computes mean squared error between input and reconstruction.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, reconstructions):
        """
        Compute reconstruction loss.

        Args:
            inputs: Original images [B, C, H, W]
            reconstructions: Reconstructed images [B, C, H, W]

        Returns:
            Mean reconstruction loss
        """
        rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2
        return torch.mean(rec_loss)

class PerceptualLoss(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.lpips = LPIPS(use_dropout=use_dropout).eval()
        
        # Freeze LPIPS parameters
        for param in self.lpips.parameters():
            param.requires_grad = False
    
    def forward(self, inputs, reconstructions):
        """
        Compute perceptual loss.
        
        Args:
            inputs: Original images [B, C, H, W]
            reconstructions: Reconstructed images [B, C, H, W]
            
        Returns:
            Mean perceptual loss
        """
        p_loss = self.lpips(inputs.contiguous(), reconstructions.contiguous())
        return torch.mean(p_loss)

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()

        # Scaling Layer
        self.scaling_layer = ScalingLayer()

        # Output channels for each VGG16 layer
        self.chns = [64, 128, 256, 512, 512]
        # Load pretrained VGG16 network and freeze parameters
        self.net = vgg16(pretrained=False, requires_grad=False)
        
        # Create linear layers
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        
        # Load pretrained weights and freeze parameters
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        # Get checkpoint file path
        ckpt = get_ckpt_path(
            name=name, 
            root="movqgan/losses"
        )

        # 加载权重
        # torch.load(ckpt) 把 .pth 文件加载为一个 state_dict 字典
        # load_state_dict() 把 state_dict 中的权重加载到模型中
        # strict=False 表示不是严格匹配，
        # 如果模型的层和权重文件中层的名字不完全一致，也不会报错。这通常用于：
        # 模型结构有轻微改动，只加载匹配的权重层，未匹配的保持初始化状态。
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), 
            strict=False
        )

        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input, target):
        # 确保输入和目标的范围在 [0,1]，并进行归一化
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        # 提取每一层的特征
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        
        feats0, feats1, diffs = {}, {}, {}
        # 线性层
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        
        for kk in range(len(self.chns)):
            # 对每一个像素点的特征向量进行 L2 归一化，输出维度为 [B, C, H, W]
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            # 计算输入和目标的特征差异，输出维度为 [B, C, H, W]
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # 计算每一个像素点各个通道差异的加权和，输出维度为 [B, 1, H, W]
        # 计算每一层各个像素点差异的平均值，输出维度为 [B, 1, 1, 1]
        # 将不同层的差异组成列表
        res = [spatial_average(lins[kk](diffs[kk])) for kk in range(len(self.chns))]
        
        # 将不同层的差异相加，输出维度为 [B, 1, 1, 1]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        
        return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()

        # 注册为 buffer，表示不会被优化器更新，但会随模型一起保存和加载
        # ImageNet 数据集的 RGB 均值为 [.485, .456, .406]
        # 标准差为 [.229, .224, .225]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        # 输出维度为[B, chn_out, H, W]，其中 B 是批量大小，H 和 W 是输入特征图的高度和宽度
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, inp):
        return self.model(inp)

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()

        # 加载预训练的 VGG16 模型，并提取特征层，即不包含全连接层的卷积层部分
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        
        # 划分为五部分
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # 冻结参数
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        # namedtuple 是 collections 模块中的一个工厂函数，用于创建一个轻量级的类
        # 这里创建了一个名为 VggOutputs 的类，包含五个字段
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

def normalize_tensor(x, eps=1e-10):
    """
    Normalize a tensor [B, C, H, W] by its L2 norm, across channels.
    Output is the same shape as input [B, C, H, W].
    Focus on direction of feature vector at each pixel.
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x):
    """
    Average a tensor [B, C, H, W] across H and W dimensions.
    Output is [B, C, 1, 1] because keepdim=True, else [B, C].
    Focus on average value per channel.
    """
    return x.mean([2,3], keepdim=True)