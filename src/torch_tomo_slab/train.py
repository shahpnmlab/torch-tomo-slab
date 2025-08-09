# src/torch_tomo_slab/train.py

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar

sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab import config
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.data.dataloading import SegmentationDataModule
from torch_tomo_slab.losses import BoundaryLoss, WeightedBCELoss, CombinedLoss # Import new losses

def get_loss_function(name: str, loss_weights: dict = None):
    name = name.lower()
    from_logits = True # All our losses will work with logits
    
    # --- Define individual loss components ---
    loss_lib = {
        'dice': smp.losses.DiceLoss(mode='binary', from_logits=from_logits),
        'bce': nn.BCEWithLogitsLoss(),
        'weighted_bce': WeightedBCELoss(from_logits=from_logits),
        'focal': smp.losses.FocalLoss(mode='binary', gamma=config.FOCAL_LOSS_GAMMA, alpha=config.FOCAL_LOSS_ALPHA),
        'lovasz': smp.losses.LovaszLoss(mode='binary', from_logits=from_logits),
        'tversky': smp.losses.TverskyLoss(mode='binary', from_logits=from_logits, alpha=config.TVERSKY_ALPHA, beta=config.TVERSKY_BETA),
        'boundary': BoundaryLoss()
    }

    # If the name is a simple key, return that loss
    if '+' not in name and name in loss_lib:
        return loss_lib[name]
    
    # If the name is a combination, build a CombinedLoss
    elif '+' in name:
        loss_components = name.split('+')
        if loss_weights is None or len(loss_components) != len(loss_weights):
            raise ValueError(f"Loss weights must be provided for combined loss '{name}'")
        
        losses_to_combine = {
            comp_name: (loss_lib[comp_name], weight)
            for comp_name, weight in zip(loss_components, loss_weights)
            if comp_name in loss_lib
        }
        return CombinedLoss(losses_to_combine, from_logits=from_logits)
    
    else:
        raise ValueError(f"Unknown loss function or combination: {name}")

def run_training():
    global_rank = int(os.environ.get("GLOBAL_RANK", 0))
    if global_rank == 0:
        print("--- Running on main process (rank 0). Verbose output enabled. ---")

    torch.set_float32_matmul_precision('high')

    # --- 1. DATA SETUP ---
    train_data_dir = Path(config.TRAIN_DATA_DIR)
    val_data_dir = Path(config.VAL_DATA_DIR)
    train_files = sorted(list(train_data_dir.glob("*.pt")))
    val_files = sorted(list(val_data_dir.glob("*.pt")))
    if not train_files or not val_files:
        raise FileNotFoundError("Training or validation data not found. Did you run the p02 script?")

    datamodule = SegmentationDataModule(
        train_pt_files=train_files,
        val_pt_files=val_files,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # --- 2. MODEL AND LOSS FUNCTION SETUP ---
    model = smp.create_model(
        arch=config.MODEL_ARCH,
        encoder_name=config.MODEL_ENCODER,
        encoder_weights="imagenet", # Using pretrained weights is often beneficial
        encoder_depth=5, # Deeper encoder for more complex features
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type='scse',
        classes=1, in_channels=2,
        activation=None # CRITICAL: No activation, loss function handles it
    )

    loss_fn = get_loss_function(config.LOSS_FUNCTION, config.LOSS_WEIGHTS)
    
    if global_rank == 0:
        loss_name = getattr(loss_fn, 'name', loss_fn.__class__.__name__)
        print(f"Using Model: {config.MODEL_ARCH}-{config.MODEL_ENCODER} (ImageNet pre-trained)")
        print(f"Using Loss Function: {loss_name}")

    pl_model = SegmentationModel(
        model=model,
        loss_function=loss_fn,
        learning_rate=config.LEARNING_RATE,
    )

    # --- 3. LOGGER, CALLBACKS & TRAINER CONFIGURATION ---
    # ... (no changes needed in this section) ...
    if global_rank == 0:
        print("\n--- Configuring Logger and Callbacks ---")
    experiment_name = f"{config.MODEL_ARCH}-{config.MODEL_ENCODER}"
    experiment_details = f"loss-{config.LOSS_FUNCTION.replace('+', '_')}_patch-{config.PATCH_SIZE}"
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
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true'
    )
    if global_rank == 0:
        print("\n--- Starting Training ---")
        print(f"To view logs, run: tensorboard --logdir={logger.save_dir}")
    trainer.fit(pl_model, datamodule=datamodule)
    if trainer.is_global_zero:
        print("--- Training Finished ---")
        print(f"Best model checkpoint saved at: {checkpointer.best_model_path}")

if __name__ == "__main__":
    run_training()
