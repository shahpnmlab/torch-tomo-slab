import sys
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab import config, constants
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.data.dataloading import SegmentationDataModule
from torch_tomo_slab.losses import SDFLoss
# Note: DynamicTrainingManager and SWA might need tuning for the new loss landscape
# We will disable them for the first run to keep things simple.

def get_loss_function(loss_name: str):
    if loss_name.lower() == 'sdf':
        return SDFLoss()
    else:
        raise ValueError(f"Unknown loss for SDF architecture: {loss_name}")

def run_training():
    global_rank = int(os.environ.get("GLOBAL_RANK", 0))
    torch.set_float32_matmul_precision('high')

    train_files = sorted(list(constants.TRAIN_DATA_DIR.glob("*.pt")))
    val_files = sorted(list(constants.VAL_DATA_DIR.glob("*.pt")))
    if not train_files or not val_files:
        raise FileNotFoundError("Training or validation data not found. Run the p02 script first!")

    datamodule = SegmentationDataModule(
        train_pt_files=train_files,
        val_pt_files=val_files,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = smp.create_model(
        arch=constants.MODEL_ARCH,
        encoder_name=constants.MODEL_ENCODER,
        encoder_weights="imagenet",
        classes=1, 
        in_channels=2,
        # Use tanh activation. It outputs in [-1, 1], which is perfect for scaling to our SDF range.
        activation='tanh'
    )

    # For SDF, we use a single dedicated loss. The config can be simplified.
    loss_fn = get_loss_function('sdf')

    if global_rank == 0:
        print(f"Using Model: {constants.MODEL_ARCH}-{constants.MODEL_ENCODER} (SDF Prediction)")
        print(f"Using Loss Function: {loss_fn.name}")

    pl_model = SegmentationModel(
        model=model,
        loss_function=loss_fn,
        learning_rate=config.LEARNING_RATE,
    )

    logger = TensorBoardLogger(save_dir="lightning_logs", name="SDF_Model")
    
    checkpointer = ModelCheckpoint(
        monitor=config.MONITOR_METRIC, mode="max",
        filename=f"best-{{epoch}}-{{{config.MONITOR_METRIC}:.4f}}",
        save_top_k=config.CHECKPOINT_SAVE_TOP_K, verbose=True,
    )
    
    callbacks = [TQDMProgressBar(refresh_rate=10), checkpointer, LearningRateMonitor(logging_interval="step")]

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        logger=logger,
        callbacks=callbacks
    )
    
    trainer.fit(pl_model, datamodule=datamodule)
    
    if trainer.is_global_zero:
        print("--- Training Finished ---")
        print(f"Best model checkpoint saved at: {checkpointer.best_model_path}")

if __name__ == "__main__":
    run_training()
