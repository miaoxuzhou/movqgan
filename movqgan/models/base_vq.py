import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from movqgan.util import instantiate_from_config
from movqgan.models.ema import EMA

class BaseVQModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        lossconfig,
        image_key="image",
        monitor=None,
        ema_decay=None,
        ignore_keys=[]
    ):
        super().__init__()
        self.automatic_optimization = False

        # image key
        self.image_key = image_key
        # learning rate
        self.learning_rate = learning_rate
        # loss function
        self.loss = instantiate_from_config(lossconfig)
        
        # keys to ignore when loading state dict
        self.ignore_keys = ignore_keys
        # 
        self.monitor = monitor if monitor is not None else None
        # whether to use EMA
        self.use_ema = ema_decay is not None
        self.ema_decay = ema_decay
    
    def init_ema(self):
        if self.use_ema:
            self.model_ema = EMA(self, self.ema_decay).to(self.device)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))} variables.")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def training_step(self, batch):
        x = batch

        # Forward pass
        xrec, qloss = self(x)

        # Get optimizer
        opt_ae, opt_disc = self.optimizers()

        # Autoencoder training    
        aeloss, log_dict_ae = self.loss(
            codebook_loss=qloss,               # codebook loss
            inputs=x,                          # inputs
            reconstructions=xrec,              # reconstructions
            optimizer_idx=0,                   # autoencoder optimizer 
            global_step=self.global_step,
            last_layer=self.get_last_layer(), 
            split="train"
        )

        opt_ae.zero_grad()
        self.manual_backward(aeloss)

        self.clip_gradients(
            opt_ae,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="value",
        )

        opt_ae.step()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Discriminator training        
        discloss, log_dict_disc = self.loss(
            codebook_loss=qloss,              # codebook loss
            inputs=x,                         # inputs
            reconstructions=xrec,             # reconstructions
            optimizer_idx=1,                  # discriminator optimizer
            global_step=self.global_step,
            last_layer=self.get_last_layer(), 
            split="train"
        )

        opt_disc.zero_grad()
        self.manual_backward(discloss)

        self.clip_gradients(
            opt_disc,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="value",
        )

        opt_disc.step()

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.model_ema.apply_shadow(self):
                xrec, qloss = self(batch)
        else:
            xrec, qloss = self(batch)
            
        aeloss, log_dict_ae = self.loss(
            codebook_loss=qloss, 
            inputs=batch, 
            reconstructions=xrec, 
            optimizer_idx=0, 
            global_step=self.global_step,
            last_layer=self.get_last_layer(), 
            split="val"
        )
        discloss, log_dict_disc = self.loss(
            codebook_loss=qloss, 
            inputs=batch, 
            reconstructions=xrec, 
            optimizer_idx=1, 
            global_step=self.global_step,
            last_layer=self.get_last_layer(), 
            split="val"
        )

        rec_loss = log_dict_ae["val/rec_loss"]

        self.log(
            "val/rec_loss", rec_loss, 
            prog_bar=True, logger=True, 
            on_step=True, on_epoch=True, sync_dist=True
        )
        self.log(
            "val/aeloss", aeloss, 
            prog_bar=True, logger=True, 
            on_step=True, on_epoch=True, sync_dist=True
        )
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        
        # Gather all parameters for the autoencoder
        ae_params = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters())
        )
        
        opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch
        x = x.to(self.device)
        
        if self.use_ema:
             with self.model_ema.apply_shadow(self):
                xrec, _ = self(x)
        else:
            xrec, _ = self(x)

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"

        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        
        return x