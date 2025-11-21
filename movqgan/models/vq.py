import torch

from movqgan.models.base_vq import BaseVQModel
from movqgan.modules.encoders.encoder import Encoder
from movqgan.modules.decoders.decoder import Decoder
from movqgan.modules.quantizers.vector_quantizer import VectorQuantizer

class VQ(BaseVQModel):
    def __init__(
        self, 
        ddconfig, 
        n_embed, 
        embed_dim, 
        ckpt_path=None, 
        sane_index_shape=False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, sane_index_shape=sane_index_shape)
        
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=self.ignore_keys)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff