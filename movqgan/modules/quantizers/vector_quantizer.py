import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat

class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    
    def __init__(
        self, 
        n_embed: int, 
        embed_dim: int, 
        beta: float, 
        remap: str = None, 
        unknown_index: str = "random",
        sane_index_shape: bool = False, 
        legacy: bool = True
    ):
        super().__init__()

        # codebook 的大小
        self.n_e = n_embed
        # codebook 向量的维度
        self.e_dim = embed_dim
        # commitment loss 的权重
        self.beta = beta
        # 是否使用 debug 后的版本
        self.legacy = legacy

        # codebook 的查找表，存储离散向量
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # remap 用于重映射索引
        self.remap = remap

        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            
            # 添加一个额外的索引用于未知的索引映射
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            
            # 打印映射信息
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = self.n_e

        # 控制输出索引的形状
        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        """
        将索引重映射到实际使用的索引集合中。
        """
        # 记录原始形状
        ishape = inds.shape
        assert len(ishape) > 1
        # 展平至二维
        inds = inds.reshape(ishape[0], -1)

        # 将 used 移动至 inds 所在的设备
        used = self.used.to(inds)
        # 匹配矩阵
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 返回匹配项
        new = match.argmax(-1)
        
        # 处理未知项
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                low=0, high=self.re_embed, size=new[unknown].shape
            ).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        """
        将索引从实际使用的索引集合反向映射到完整的索引集合中。
        """
        # 记录原始形状
        ishape = inds.shape
        assert len(ishape) > 1
        # 展平至二维
        inds = inds.reshape(ishape[0], -1)

        # 将 used 移动至 inds 所在的设备
        used = self.used.to(inds)
        # 处理额外项
        if self.re_embed > self.used.shape[0]:   # extra token
            inds[inds >= self.used.shape[0]] = 0 # simply set to zero
        
        back = torch.gather(
            input=used[None, :][inds.shape[0] * [0], :], 
            dim=1, 
            index=inds
        )
        
        return back.reshape(ishape)

    def forward(
        self, 
        z: torch.Tensor, 
        temp: float = None, 
        rescale_logits: bool = False, 
        return_logits: bool = False
    ):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        
        # 重排并展平 z
        # 维度从 [B, C, H, W] 变为 [B, H, W, C]
        # 又变为 [B * H * W, C]
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # 计算 z_flattened 与 embedding 之间的距离
        # distances from z to embeddings e = (z - e)^2 = z^2 + e^2 - 2 e * z
        # 维度为 [B * H * W, n_e]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        # 获取最近编码，维度为 [B * H * W,]
        # z_q的维度为 [B，H，W, C]
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # 计算 perplexity 和 min_encodings
        perplexity = None
        min_encodings = None

        # 计算 VQ 损失，包括 commitment loss 和 codebook loss
        # commitment loss 促使编码器输出靠近量化向量
        # codebook loss 促使量化向量靠近编码器输出
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # 重映射索引
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1) # add batch axis, output dimension is [B, H * W]
            min_encoding_indices = self.remap_to_used(min_encoding_indices)     # remap, output dimension is [B, H * W]
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)          # flatten, output dimension is [B * H * W, 1]

        # 调整索引形状，输出维度为 [B, H, W]
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # 重映射索引
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1) # add batch axis, output dimension is [B, H * W]
            indices = self.unmap_to_all(indices)    # unmap, output dimension is [B, H * W]
            indices = indices.reshape(-1)           # flatten again, output dimension is [B * H * W]

        # 获取 codebook 条目
        z_q = self.embedding(indices)

        # 调整形状，输出维度为 [B, C, H, W]
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q