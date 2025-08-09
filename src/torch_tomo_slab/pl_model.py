# src/torch_tomo_slab/pl_model.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch_tomo_slab import config

class SegmentationModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function'])
        self.model = model
        self.criterion = loss_function

    def setup(self, stage: str):
        pass

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage: str):
        image = batch['image']
        label = batch['label']
        weight_map = batch['weight_map'] # Unpack weight map
        batch_size = image.size(0)

        pred_logits = self(image)
        
        # Pass all necessary components to the loss function
        loss = self.criterion(pred_logits, label, weight_map)

        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=(stage=='train'), on_epoch=True, batch_size=batch_size, sync_dist=True)
        return loss, pred_logits

    def training_step(self, batch, batch_idx):
        loss, pred_logits = self._common_step(batch, batch_idx, "train")
        if batch_idx == 0 and self.trainer.is_global_zero:
            pred_probs = torch.sigmoid(pred_logits)
            self._log_images(batch, pred_probs, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_logits = self._common_step(batch, batch_idx, "val")
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > 0.5).float()
        dice = self.dice_coefficient(pred_binary, batch['label'])
        self.log('val_dice', dice, prog_bar=True, on_epoch=True, batch_size=batch['image'].size(0), sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(batch, pred_probs, "Validation")
        return loss

    def _log_images(self, batch: dict, pred_probs: torch.Tensor, stage_name: str):
        # ... (no changes needed here) ...
        if self.logger is None or not hasattr(self.logger.experiment, 'add_image'):
            return
        image, label = batch['image'], batch['label']
        num_images_to_log = min(8, image.size(0))
        grid_params = {"padding": 2, "pad_value": 1.0, "nrow": 4}
        input_grid = torchvision.utils.make_grid(
            image[:num_images_to_log, 0:1, :, :],
            **grid_params,
            normalize=True
        )
        self.logger.experiment.add_image(f"{stage_name}/Input (Tomogram)", input_grid, self.current_epoch)
        label_grid = torchvision.utils.make_grid(label[:num_images_to_log].to(torch.float32), **grid_params)
        self.logger.experiment.add_image(f"{stage_name}/Ground Truth", label_grid, self.current_epoch)
        pred_grid = torchvision.utils.make_grid(pred_probs[:num_images_to_log].to(torch.float32), **grid_params)
        self.logger.experiment.add_image(f"{stage_name}/Prediction", pred_grid, self.current_epoch)

    def dice_coefficient(self, pred, target, smooth=1e-5):
        # ... (no changes needed here) ...
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = torch.mean((2. * intersection + smooth) / (union + smooth))
        return dice

    def configure_optimizers(self):
        # ... (no changes needed here) ...
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        if not config.USE_LR_SCHEDULER:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.SCHEDULER_MONITOR,
                "interval": "epoch",
                "frequency": 1,
            },
        }
