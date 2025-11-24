import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..components.attention import AttnBlock
from ..components.residual import ResnetBlock, Normalize, nonlinearity, get_timestep_embedding
from ..components.sampling import Upsample

class Decoder(nn.Module):
    def __init__(
        self, 
        *, 
        ch: int, 
        temb_ch: int = 0,
        out_ch: int,
        resolution: int,
        z_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        give_pre_end: bool = False,
        **ignorekwargs
    ):
        """
        Decodes a latent representation back into an image using a series of Resnet and Attention blocks.

        Args:
            ch: basic channel count.
            temb_ch: time embedding channel count.
            out_ch: output channel count.
            ch_mult: channel multiplier sequence.
            num_res_blocks: number of ResnetBlocks at each resolution.
            attn_resolutions: list of resolutions to apply attention (e.g., [16] means apply at 16x16).
            dropout: dropout probability.
            resamp_with_conv: whether to use convolutional layers for upsampling.
            resolution: resolution of the input image.
            z_channels: number of channels in the latent code.
            give_pre_end: whether to return the features before the final normalization and convolution during forward pass.
        
        Returns:
            Reconstructed image from the latent representation.
        """
        super().__init__()

        # 基础通道数
        self.ch = ch
        # 时间嵌入维度，即时间嵌入的通道数
        self.temb_ch = temb_ch

        # 总共有几种分辨率，即需要多少次上采样
        self.num_resolutions = len(ch_mult)
        # 每一层的 ResnetBlock 数量组成的列表
        self.num_res_blocks = num_res_blocks

        # 输入的分辨率
        self.resolution = resolution
        # 是否返回前端特征图，即未经过输出层的特征图
        self.give_pre_end = give_pre_end

        # 计算潜在编码经过输入层后的输出通道数，即第一个 ResnetBlock 的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        
        # 潜在编码的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # 打印潜在编码的形状和维度数
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)
        ))

        # 潜在编码进入输入层
        self.conv_in = torch.nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 中间层
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

        # 上采样
        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            # ResnetBlock 列表
            block = nn.ModuleList()
            # AttnBlock 列表
            attn = nn.ModuleList()
            # 输出维度数
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                # 添加 ResnetBlock
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))

                # 更新输入通道数
                block_in = block_out

                # 如果当前分辨率需要，则添加注意力层
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            # 组合模块
            up = nn.Module()
            up.block = block
            up.attn = attn

            # 如果不是最后一层，则添加上采样层
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            
            # 追加到上采样模块列表
            self.up.insert(0, up)

        # 输出层
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(
            self, 
            z: torch.Tensor, 
            timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 确保输入潜在编码的形状正确
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # 时间嵌入
        if self.temb_ch > 0 and timesteps is not None:
            temb = get_timestep_embedding(
                timesteps=timesteps,
                embedding_dim=self.temb_ch,
            )
        else:
            temb = None

        # 输入层
        h = self.conv_in(z)

        # 中间层
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            
            for i_block in range(self.num_res_blocks + 1):
                # 通过 ResnetBlock
                h = self.up[i_level].block[i_block](h, temb)
                # 如果有的话，则通过 AttnBlock
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            
            # 除最后一层外，进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 输出未经过最终归一化和卷积层的特征图
        if self.give_pre_end:
            return h

        # 输出层
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h