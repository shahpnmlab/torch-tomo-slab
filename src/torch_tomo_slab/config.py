from pathlib import Path

BASE_DATA_PATH = Path("/home/pranav/data/training/torch-tomo-slab")  # Adjust this to your project's root data folder

# Input data for script 01 and 02
#RAW_DATA_PATH = BASE_DATA_PATH/"data_in"/""
IMOD_MODEL_DIR = BASE_DATA_PATH/"data_in"/"mods"
REFERENCE_TOMOGRAM_DIR = BASE_DATA_PATH/"data_in"/"volumes"

# Output data from scripts
MASK_OUTPUT_DIR =  BASE_DATA_PATH / "data_in" / "boundary_mask_voumes"
PREPARED_DATA_BASE_DIR = BASE_DATA_PATH / "prepared_data"
TRAIN_DATA_DIR = PREPARED_DATA_BASE_DIR / "train"
VAL_DATA_DIR = PREPARED_DATA_BASE_DIR / "val"
MULTIPLY_TOMO_MASK = True

#Resize volume-labels to these dims
TARGET_VOLUME_SHAPE = (256, 512, 512)

# Number of 2D slices to extract from each 3D volume
NUM_SECTIONS_PER_VOLUME = 256

# Kernel size for the local variance calculation
LOCAL_VARIANCE_KERNEL_SIZE = 5

# --- TRAINING PARAMETERS ---
# Model Architecture (from segmentation-models-pytorch)
MODEL_ARCH = "Unet"
MODEL_ENCODER = "resnet18"

# Training Hyperparameters
LEARNING_RATE = 1e-4
# --- MODIFIED: Added more loss options ---
#LOSS_FUNCTION = "dice+boundary"  # Options: 'dice', 'bce', 'dice+bce', 'focal+dice', 'tverskyloss','lovasz'
LOSS_FUNCTION = "weighted_bce"  # Options: 'dice', 'bce', 'dice+bce', 'focal+dice', 'tverskyloss','lovasz'
LOSS_WEIGHTS = (0.7, 0.3)  # Used for combined losses like dice+bce or focal+dice
MAX_EPOCHS = 50
PRECISION='bf16-mixed'

# --- NEW: FOCAL LOSS CONFIGURATION ---
FOCAL_LOSS_GAMMA = 2.0  # The focusing parameter
FOCAL_LOSS_ALPHA = None # The alpha balancing weight. Use None for 0.5, or e.g., 0.75

# These are only used if LOSS_FUNCTION = 'tverskyloss'
# alpha penalizes False Positives, beta penalizes False Negatives. alpha+beta should be ~1
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7


# Dataloader & Augmentation
PATCH_SIZE = 128
OVERLAP = 64
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 64
NUM_WORKERS = 8  # Adjust based on your machine's CPUs


# Patch Sampling Strategy
SAMPLES_PER_VOLUME = 100  # Number of patches to extract per 2D image
ALPHA_FOR_DROPPING = 0.75  # Controls how aggressively to drop empty patches. 0=no drop, 1=aggressive.
VALIDATION_PATCH_SAMPLING = False  # Set to False for validation on full images, True for patches

# --- TRAINER CONFIGURATION ---
ACCELERATOR = "auto"  # Let PyTorch Lightning detect GPU/MPS/CPU
DEVICES = 2
LOG_EVERY_N_STEPS = 10
CHECK_VAL_EVERY_N_EPOCH = 1

# --- MODIFIED: LEARNING RATE SCHEDULER CONFIGURATION FOR ReduceLROnPlateau ---
USE_LR_SCHEDULER = True
# The metric to monitor for reducing the LR, should match MONITOR_METRIC
SCHEDULER_MONITOR = "val_dice"
# How many epochs to wait for improvement before reducing LR
SCHEDULER_PATIENCE = 3
# The factor by which to reduce the learning rate (e.g., 0.1 = 10x reduction)
SCHEDULER_FACTOR = 0.1
# The minimum learning rate to ever decay to
SCHEDULER_MIN_LR = 1e-8


# --- CALLBACK CONFIGURATION ---
# The metric to monitor for checkpointing and early stopping
MONITOR_METRIC = "val_dice"

# Early Stopping configuration
EARLY_STOPPING_PATIENCE = 15  # Stop after 15 validation epochs with no improvement
EARLY_STOPPING_MIN_DELTA = 0.001 # Minimum change to be considered an improvement

# Model Checkpoint configuration
CHECKPOINT_SAVE_TOP_K = 1 # Save only the single best model

# Stochastic Weight Averaging (SWA) configuration
USE_SWA = True
SWA_LEARNING_RATE = 0.1*LEARNING_RATE # SWA learning rate is typically smaller
SWA_START_EPOCH_FRACTION = 0.75 # Start SWA in the last 25% of epochs
