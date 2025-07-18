import sys
import os # <-- IMPORT ADDED
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
    TQDMProgressBar
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab import config
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.data.dataloading import SegmentationDataModule

# (Your CombinedLoss definition remains unchanged)
class CombinedLoss(nn.Module):
    # ... (no changes)
    def __init__(self, loss1: nn.Module, loss2: nn.Module, weight1: float = 0.5, weight2: float = 0.5):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2
        loss1_name = self.loss1.__class__.__name__
        loss2_name = self.loss2.__class__.__name__
        self.name = f"{self.weight1}*{loss1_name}_{self.weight2}*{loss2_name}"
    def forward(self, pred, target):
        loss_val1 = self.loss1(pred, target)
        loss_val2 = self.loss2(pred, target)
        return self.weight1 * loss_val1 + self.weight2 * loss_val2


def get_loss_function(name: str, weights: tuple = (0.5, 0.5)):
    # This function now only creates the loss, it doesn't print.
    name = name.lower()
    if name == "diceloss":
        return smp.losses.DiceLoss(mode='binary', from_logits=True)
    elif name == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss()
    elif name == "dice+bce":
        dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        bce_loss = nn.BCEWithLogitsLoss()
        return CombinedLoss(dice_loss, bce_loss, weight1=weights[0], weight2=weights[1])
    # --- MODIFIED: Use configured gamma and alpha for FocalLoss ---
    elif name == "focal+dice":
        focal_loss = smp.losses.FocalLoss(
            mode='binary',
            gamma=config.FOCAL_LOSS_GAMMA,
            alpha=config.FOCAL_LOSS_ALPHA
        )
        dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        return CombinedLoss(focal_loss, dice_loss, weight1=weights[0], weight2=weights[1])
    elif name == "tverskyloss":
        return smp.losses.TverskyLoss(
            mode='binary',
            from_logits=True,
            alpha=config.TVERSKY_ALPHA,
            beta=config.TVERSKY_BETA
        )
    else:
        raise ValueError(f"Unknown loss function: {name}")


def run_training():
    """Main function to configure and run the training pipeline."""
    # --- NEW: Check for main process to control printing ---
    # We check the environment variable for setup before the trainer is initialized.
    # Default to '0' for non-distributed runs.
    global_rank = int(os.environ.get("GLOBAL_RANK", 0))

    if global_rank == 0:
        print("--- Running on main process (rank 0). Verbose output enabled. ---")

    # Best Practice: Set matmul precision for modern GPUs
    torch.set_float32_matmul_precision('high')

    # --- 1. DATA SETUP ---
    # --- MODIFIED: Load data from pre-split directories ---
    train_data_dir = Path(config.TRAIN_DATA_DIR)
    val_data_dir = Path(config.VAL_DATA_DIR)

    train_files = sorted(list(train_data_dir.glob("*.pt")))
    val_files = sorted(list(val_data_dir.glob("*.pt")))

    if not train_files:
        raise FileNotFoundError(f"No training '.pt' files found in {train_data_dir}. Did you run the data preparation script?")
    if not val_files:
        raise FileNotFoundError(f"No validation '.pt' files found in {val_data_dir}. Did you run the data preparation script?")

    if global_rank == 0:
        print(f"Found {len(train_files) + len(val_files)} total 2D sections, pre-split into:")
        print(f"Training sections: {len(train_files)}")
        print(f"Validation sections: {len(val_files)}")

    datamodule = SegmentationDataModule(
        train_pt_files=train_files, val_pt_files=val_files, patch_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        samples_per_volume=config.SAMPLES_PER_VOLUME, alpha_for_dropping=config.ALPHA_FOR_DROPPING,
        val_patch_sampling=config.VALIDATION_PATCH_SAMPLING
    )

    # --- 2. MODEL AND LOSS FUNCTION SETUP ---
    # --- FIX: Pass a dictionary to decoder_use_norm to enable affine parameters ---
    model = smp.create_model(
        arch=config.MODEL_ARCH, encoder_name=config.MODEL_ENCODER, classes=1, in_channels=2,
        decoder_use_norm={"type": "instancenorm", "affine": True}
    )
    loss_fn = get_loss_function(config.LOSS_FUNCTION, config.LOSS_WEIGHTS)
    
    # Centralize printing about the configuration
    if global_rank == 0:
        loss_name = getattr(loss_fn, 'name', loss_fn.__class__.__name__)
        print(f"Using loss function: {loss_name}")
        if 'focal' in config.LOSS_FUNCTION.lower():
             print(f"  - Focal Loss Params: gamma={config.FOCAL_LOSS_GAMMA}, alpha={config.FOCAL_LOSS_ALPHA}")
        if 'tversky' in config.LOSS_FUNCTION.lower():
             print(f"  - Tversky Loss Params: alpha={config.TVERSKY_ALPHA}, beta={config.TVERSKY_BETA}")
        print("Using Instance Normalization (with affine=True) in the U-Net decoder.")


    pl_model = SegmentationModel(
        model=model,
        loss_function=loss_fn,
        learning_rate=config.LEARNING_RATE,
    )

    # --- 3. LOGGER, CALLBACKS & TRAINER CONFIGURATION ---
    if global_rank == 0:
        print("\n--- Configuring Logger and Callbacks ---")
    
    # --- MODIFIED: Added norm type to experiment details for clearer logging ---
    experiment_name = f"{config.MODEL_ARCH}-{config.MODEL_ENCODER}"
    experiment_details = f"loss-{config.LOSS_FUNCTION}_norm-instance_patch-{config.PATCH_SIZE}"

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{experiment_name}/{experiment_details}"
    )


    checkpointer = ModelCheckpoint(
        monitor=config.MONITOR_METRIC, mode="max",
        filename=f"best-{{epoch}}-{{{config.MONITOR_METRIC}:.4f}}",
        save_top_k=config.CHECKPOINT_SAVE_TOP_K, verbose=True,
    )
    early_stopper = EarlyStopping(
        monitor=config.MONITOR_METRIC, mode="max", patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks = [progress_bar, checkpointer, early_stopper, lr_monitor]

    if config.USE_SWA:
        if global_rank == 0:
            swa_start_epoch = int(config.MAX_EPOCHS * config.SWA_START_EPOCH_FRACTION)
            print(f"Enabling Stochastic Weight Averaging (SWA) starting at epoch {swa_start_epoch}.")
        swa = StochasticWeightAveraging(swa_lrs=config.SWA_LEARNING_RATE, swa_epoch_start=config.SWA_START_EPOCH_FRACTION)
        callbacks.append(swa)

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS, accelerator=config.ACCELERATOR, devices=config.DEVICES,
        precision=config.PRECISION, log_every_n_steps=config.LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH,
        logger=logger,
        callbacks=callbacks
    )

    # --- 4. START TRAINING ---
    if global_rank == 0:
        print("\n--- Starting Training ---")
        print(f"To view logs, run: tensorboard --logdir={logger.save_dir}")

    trainer.fit(pl_model, datamodule=datamodule)
    
    # --- 5. POST-TRAINING SUMMARY ---
    # Now that the trainer exists, we can use the more convenient `is_global_zero`.
    if trainer.is_global_zero:
        print("--- Training Finished ---")
        print(f"Best model checkpoint saved at: {checkpointer.best_model_path}")

        if config.USE_SWA:
            swa_model_path = Path(trainer.logger.log_dir) / "final_swa_model.pth"
            torch.save(pl_model.model.state_dict(), swa_model_path)
            print(f"Final (SWA-averaged) model weights saved to: {swa_model_path}")


if __name__ == "__main__":
    run_training()
