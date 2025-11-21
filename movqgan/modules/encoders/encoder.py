import torch
import torch.nn as nn
from typing import Tuple, Optional

from movqgan.modules.components.attention import AttnBlock
from movqgan.modules.components.residual import ResnetBlock, Normalize, nonlinearity, get_timestep_embedding
from movqgan.modules.components.sampling import Downsample

class Encoder(nn.Module):
    def __init__(
        self, 
        ch: int, 
        *,
        out_ch: int = None,
        temb_ch: int = 0,
        in_channels: int,
        resolution: int, 
        z_channels: int, 
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8), 
        dropout: float = 0.0, 
        resamp_with_conv: bool = True, 
        double_z: bool = True, 
    ):
        """
        Encodes an image into a latent representation using a series of Resnet and Attention blocks.

        Args:
            ch: basic channel count.
            temb_ch: time embedding channel count.
            ch_mult: channel multiplier sequence.
            num_res_blocks: number of ResnetBlocks per resolution.
            attn_resolutions: list of resolutions to apply attention (e.g., [16] means to apply at 16x16).
            dropout: dropout probability.
            resamp_with_conv: whether to use convolutional layers for downsampling.
            in_channels: input channel count.
            resolution: input image resolution.
            z_channels: latent code channel count.
            double_z: whether to output double channels (commonly used for VAE's mean and variance, but usually False in VQ-VAE).
        
        Returns:
            Latent representation of the input image.
        """
        super().__init__()

        # 基础通道数
        self.ch = ch
        # 时间嵌入通道数
        self.temb_ch = temb_ch

        # 总共有几种分辨率，即下采样层的数量
        self.num_resolutions = len(ch_mult)
        # ResnetBlock 数量
        self.num_res_blocks = num_res_blocks
        # 输入图像的分辨率
        self.resolution = resolution
        # 输入通道数
        self.in_channels = in_channels

        # 输入卷积层
        self.conv_in = torch.nn.Conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 当前分辨率
        curr_res = resolution
        # 输入通道数的倍增序列
        in_ch_mult = (1,) + tuple(ch_mult)
        # 下采样模块列表
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            # 输入和输出通道数
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            # 添加多个 ResnetBlock 和 AttnBlock
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))

                # 更新输入通道数
                block_in = block_out

                # 如果当前分辨率需要注意力机制，则添加 AttnBlock
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            # 组合模块
            down = nn.Module()
            down.block = block
            down.attn = attn

            # 如果不是最后一层，则添加下采样层
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(
                    in_channels=block_in, 
                    with_conv=resamp_with_conv
                )
                # 分辨率减半
                curr_res = curr_res // 2
            
            self.down.append(down)

        # 中间模块
        self.mid = nn.Module()

        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # 输出层
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )


    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # 时间嵌入
        if self.temb_ch > 0 and timesteps is not None:
            temb = get_timestep_embedding(
                timesteps=timesteps,
                embedding_dim=self.temb_ch
            )
        else:
            temb = None

        # 输入卷积
        hs = [self.conv_in(x)]

        # 下采样路径
        for i_level in range(self.num_resolutions):

            # 遍历 ResnetBlock 和 AttnBlock
            for i_block in range(self.num_res_blocks):
                # 将列表中的最后一项输入到当前块
                h = self.down[i_level].block[i_block](hs[-1], temb)
                
                # 如果有注意力层
                # 如果有w_ = torch.nn.functional.softmax(w_, dim=2)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                
                hs.append(h)
            
            # 下采样，最后一层除外
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间模块
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # 输出层
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h