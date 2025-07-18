from pathlib import Path

BASE_DATA_PATH = Path("/home/pranav/data/training/torch-tomo-slab")  # Adjust this to your project's root data folder

# Input data for script 01 and 02
#RAW_DATA_PATH = BASE_DATA_PATH/"data_in"/""
IMOD_MODEL_DIR = BASE_DATA_PATH/"data_in"/"mods"
REFERENCE_TOMOGRAM_DIR = BASE_DATA_PATH/"data_in"/"volumes"

# Output data from scripts
MASK_OUTPUT_DIR =  BASE_DATA_PATH / "data_in" / "boundary_mask_voumes"
# --- MODIFIED: Renamed PREPARED_DATA_DIR to be a base path ---
PREPARED_DATA_BASE_DIR = BASE_DATA_PATH / "prepared_data"
# --- NEW: Define separate directories for training and validation data ---
TRAIN_DATA_DIR = PREPARED_DATA_BASE_DIR / "train"
VAL_DATA_DIR = PREPARED_DATA_BASE_DIR / "val"
MULTIPLY_TOMO_MASK = True

# Number of 2D slices to extract from each 3D volume
NUM_SECTIONS_PER_VOLUME = 150

# Kernel size for the local variance calculation
LOCAL_VARIANCE_KERNEL_SIZE = 5

# --- TRAINING PARAMETERS ---
# Model Architecture (from segmentation-models-pytorch)
MODEL_ARCH = "Unet"
MODEL_ENCODER = "resnet34"

# Training Hyperparameters
LEARNING_RATE = 2e-5
# --- MODIFIED: Added more loss options ---
LOSS_FUNCTION = "tverskyloss"  # Options: 'dice', 'bce', 'dice+bce', 'focal+dice', 'tverskyloss'
LOSS_WEIGHTS = (0.5, 0.5)  # Used for combined losses like dice+bce or focal+dice
MAX_EPOCHS = 50
PRECISION='bf16-mixed'

# --- NEW: FOCAL LOSS CONFIGURATION ---
# These are only used if 'focal' is in the LOSS_FUNCTION name
FOCAL_LOSS_GAMMA = 2.0  # The focusing parameter
FOCAL_LOSS_ALPHA = None # The alpha balancing weight. Use None for 0.5, or e.g., 0.75

# --- NEW: TVERSKY LOSS CONFIGURATION ---
# These are only used if LOSS_FUNCTION = 'tverskyloss'
# alpha penalizes False Positives, beta penalizes False Negatives. alpha+beta should be ~1
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7


# Dataloader & Augmentation
PATCH_SIZE = 128
OVERLAP = 64
# --- NOTE: This fraction is now used by the data prep script for splitting volumes ---
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 16
NUM_WORKERS = 8  # Adjust based on your machine's CPUs


# Patch Sampling Strategy
SAMPLES_PER_VOLUME = 70  # Number of patches to extract per 2D image
ALPHA_FOR_DROPPING = 0.75  # Controls how aggressively to drop empty patches. 0=no drop, 1=aggressive.
VALIDATION_PATCH_SAMPLING = False  # Set to False for validation on full images, True for patches

# --- TRAINER CONFIGURATION ---
ACCELERATOR = "auto"  # Let PyTorch Lightning detect GPU/MPS/CPU
DEVICES = 2
LOG_EVERY_N_STEPS = 10
CHECK_VAL_EVERY_N_EPOCH = 1

# --- NEW: LEARNING RATE SCHEDULER CONFIGURATION ---
USE_LR_SCHEDULER = True
# Number of epochs to slowly ramp up the learning rate from 0
SCHEDULER_WARMUP_EPOCHS = 5
# The maximum learning rate is the one we already defined
# SCHEDULER_MAX_LR = LEARNING_RATE
# The minimum learning rate at the end of the cosine cycle
SCHEDULER_MIN_LR = 1e-6

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
