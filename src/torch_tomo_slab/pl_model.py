import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

from torch_tomo_slab import config
from torch_tomo_slab.losses import SDF_CLAMP_DIST

class SegmentationModel(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_function: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function'])
        self.model = model
        self.criterion = loss_function

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage: str):
        image = batch['image']
        target_sdf = batch['sdf']
        pred_sdf = self(image)
        loss = self.criterion(pred_sdf, target_sdf)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=(stage=='train'), on_epoch=True, sync_dist=True)
        return loss, pred_sdf

    def training_step(self, batch, batch_idx):
        loss, pred_sdf = self._common_step(batch, batch_idx, "train")
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(batch, pred_sdf, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_sdf = self._common_step(batch, batch_idx, "val")
        pred_sdf_scaled = pred_sdf * SDF_CLAMP_DIST
        pred_mask = (pred_sdf_scaled < 0).float()
        target_mask = (batch['sdf'] < 0).float()
        dice = self.dice_coefficient(pred_mask, target_mask)
        self.log('val_dice', dice, prog_bar=True, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(batch, pred_sdf, "Validation")
        return loss

    def _log_images(self, batch: dict, pred_sdf: torch.Tensor, stage_name: str):
        if self.logger is None: return
        num_images_to_log = min(4, pred_sdf.size(0))
        input_grid = torchvision.utils.make_grid(batch['image'][:num_images_to_log, 0:1, :, :], normalize=True)
        self.logger.experiment.add_image(f"{stage_name}/Input", input_grid, self.current_epoch)
        true_sdf_grid = self.sdf_to_image_grid(batch['sdf'][:num_images_to_log])
        self.logger.experiment.add_image(f"{stage_name}/True SDF", true_sdf_grid, self.current_epoch)
        pred_sdf_scaled = pred_sdf[:num_images_to_log] * SDF_CLAMP_DIST
        pred_sdf_grid = self.sdf_to_image_grid(pred_sdf_scaled)
        self.logger.experiment.add_image(f"{stage_name}/Predicted SDF", pred_sdf_grid, self.current_epoch)
        true_mask_grid = torchvision.utils.make_grid((batch['sdf'][:num_images_to_log] < 0).float())
        self.logger.experiment.add_image(f"{stage_name}/True Mask (from SDF)", true_mask_grid, self.current_epoch)
        pred_mask_grid = torchvision.utils.make_grid((pred_sdf_scaled < 0).float())
        self.logger.experiment.add_image(f"{stage_name}/Predicted Mask (from SDF)", pred_mask_grid, self.current_epoch)

    def sdf_to_image_grid(self, sdf_tensor: torch.Tensor):
        images = []
        sdf_min, sdf_max = -SDF_CLAMP_DIST, SDF_CLAMP_DIST
        normalized_sdf = (sdf_tensor.clamp(sdf_min, sdf_max) - sdf_min) / (sdf_max - sdf_min)
        for sdf_img in normalized_sdf:
            sdf_np = sdf_img.squeeze().detach().cpu().float().numpy()
            cmap = plt.get_cmap('coolwarm')
            colored_image = (cmap(sdf_np)[:, :, :3] * 255).astype(np.uint8)
            images.append(torch.from_numpy(colored_image).permute(2, 0, 1))
        return torchvision.utils.make_grid(images)

    def dice_coefficient(self, pred, target, smooth=1e-5):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + smooth) / (union + smooth)

    def configure_optimizers(self):
        # Use a slightly lower learning rate to encourage more stable convergence
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=config.LEARNING_RATE, # Lowered learning rate
            weight_decay=1e-4
        )
        
        # Use Cosine Annealing for a smooth learning rate decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs, # The number of epochs to complete one cycle
            eta_min=1e-7 # The minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
