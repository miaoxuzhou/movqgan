import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..components.attention import SpatialAttnBlock
from ..components.residual import SpatialResnetBlock, nonlinearity, get_timestep_embedding
from ..components.sampling import Upsample
from ..components.normalization import SpatialNorm

class MOVQDecoder(nn.Module):
    def __init__(
        self, 
        *, 
        ch: int, 
        temb_ch: int = 0,
        out_ch: int, 
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8), 
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...], 
        dropout: float = 0.0, 
        resamp_with_conv: bool = True, 
        resolution: int, 
        z_channels: int, 
        give_pre_end: bool = False, 
        zq_ch: Optional[int] = None, 
        add_conv: bool = False,
        **ignorekwargs
    ):
        super().__init__()
        
        # 基础通道数
        self.ch = ch
        # 时间嵌入的通道数
        self.temb_ch = temb_ch

        # 总共有几种分辨率，即下采样层的数量
        self.num_resolutions = len(ch_mult)
        # 每个分辨率下的残差块数量
        self.num_res_blocks = num_res_blocks

        # 输入的分辨率
        self.resolution = resolution
        # 是否返回前端特征图，即未经过输出层的特征图
        self.give_pre_end = give_pre_end
        
        # 计算最低分辨率下的输入通道数，即潜在编码经过第一层卷积后的通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 潜在编码的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # 打印潜在编码的形状和维度数
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # 潜在编码经过这一层卷积后，通道数变为 block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = SpatialResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv
        )
        self.mid.attn_1 = SpatialAttnBlock(
            block_in, 
            zq_ch, 
            add_conv=add_conv
        )
        self.mid.block_2 = SpatialResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv
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
                block.append(SpatialResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                    zq_ch=zq_ch,
                    add_conv=add_conv
                ))

                # 更新输入通道数
                block_in = block_out

                # 如果当前分辨率需要，则添加 AttnBlock
                if curr_res in attn_resolutions:
                    attn.append(SpatialAttnBlock(
                        block_in, 
                        zq_ch, 
                        add_conv=add_conv
                    ))
            
            # 组合模块
            up = nn.Module()
            up.block = block
            up.attn = attn

            # 如果不是最高分辨率，则进行上采样
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            
            # 追加到上采样模块列表中
            self.up.insert(0, up) # prepend to get consistent order

        # 输出层
        self.norm_out = SpatialNorm(
            block_in, 
            zq_ch, 
            add_conv=add_conv
        )
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
            zq: Optional[torch.Tensor] = None,
            timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 确保输入的潜在编码形状正确
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
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):

            for i_block in range(self.num_res_blocks + 1):
                # 通过 ResnetBlock
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 如果有的话，则通过 AttnBlock
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            
            # 除最后一层外，进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 如果需要，输出未经过输出层的特征图
        if self.give_pre_end:
            return h

        # 输出层
        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h