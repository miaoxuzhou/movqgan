import torch
from einops import rearrange

from movqgan.models.base_vq import BaseVQModel
from movqgan.modules.encoders.encoder import Encoder
from movqgan.modules.decoders.movq_decoder import MOVQDecoder
from movqgan.modules.quantizers.vector_quantizer import VectorQuantizer

class MOVQ(BaseVQModel):
    def __init__(
        self, 
        ddconfig, 
        n_embed, 
        embed_dim, 
        ckpt_path=None, 
        remap=None, 
        sane_index_shape=False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = False

        self.encoder = Encoder(**ddconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.init_ema()
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=self.ignore_keys)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, zq=quant)
        return dec

    def decode_code(self, code_b):
        """Decodes from codebook index tensor.
    
        Args:
            code_b: Codebook indices, can be:
                - [batch, height, width] for single-channel quantization
                - [batch, channels, height, width] for multi-channel quantization (MoVQ)
                - [batch, height*width] flattened version
        
        Returns:
            Decoded image tensor [batch, 3, H*16, W*16] (upsampled by decoder)
        """
        batch_size = code_b.shape[0]

        if len(code_b.shape) == 4:
            # Multi-channel: [batch, channels, height, width]
            # MoVQ uses 4 channels
            channels, h, w = code_b.shape[1], code_b.shape[2], code_b.shape[3]
            code_b = rearrange(code_b, 'b c h w -> b (c h w)')  # Flatten all
            total_elements = channels * h * w
        
        elif len(code_b.shape) == 3:
            # Single-channel: [batch, height, width]
            h, w = code_b.shape[1], code_b.shape[2]
            code_b = code_b.reshape(batch_size, -1)  # Flatten
            channels = 1
            total_elements = h * w
        
        elif len(code_b.shape) == 2:
            # Already flattened: [batch, height*width] or [batch, channels*height*width]
            total_elements = code_b.shape[1]
            
            # For MoVQ: assume 4 channels, spatial dims are sqrt(total/4)
            if total_elements % 4 == 0 and total_elements >= 1024:  # MoVQ case
                channels = 4
                spatial_size = total_elements // channels
                h = w = int(spatial_size ** 0.5)
            else:  # Standard VQ case
                channels = 1
                h = w = int(total_elements ** 0.5)
        
        else:
            raise ValueError(f"Unexpected code_b shape: {code_b.shape}")
        
        # Get embeddings from codebook
        # Calculate the actual spatial dimensions for the shape parameter
        if channels > 1:
            # For multi-channel, we need to handle it differently
            # Each spatial location has 'channels' codes
            quant = self.quantize.embedding(code_b.flatten())
            # Reshape: [batch*channels*h*w, embed_dim] -> [batch, channels, h, w, embed_dim]
            quant = quant.view(batch_size, channels, h, w, self.quantize.e_dim)
            # For MoVQ, we typically concatenate channels
            quant = rearrange(quant, 'b c h w d -> b (c d) h w')
        else:
            # Single channel case - use get_codebook_entry directly
            shape = (batch_size, h, w, self.quantize.e_dim)
            quant = self.quantize.get_codebook_entry(code_b, shape)
        
        # Decode
        dec = self.decode(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff