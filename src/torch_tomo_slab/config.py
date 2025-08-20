# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 3e-5
MAX_EPOCHS = 50
PRECISION = 'bf16-mixed'
VALIDATION_FRACTION = 0.2

# --- Loss Function Configuration ---
# 'name': Can be a single loss or multiple losses combined with '+'.
# 'weights': A list of weights for combined losses. Must match the number of losses.
# 'params': A nested dictionary for loss-specific hyperparameters.
LOSS_CONFIG = {
    'name': 'weighted_bce',  # Options: 'dice', 'bce', 'dice+bce', 'focal', 'tversky', 'boundary', ''weighted_bce'.
    'weights': [0.7, 0.3],     # Only used for combined losses like 'focal+dice'
    'params': {
        'focal': {
            'gamma': 2.0,      # The focusing parameter
            'alpha': None      # The alpha balancing weight (None defaults to 0.5)
        },
        'tversky': {           # alpha penalizes FPs, beta penalizes FNs
            'alpha': 0.3,
            'beta': 0.7
        }
    }
}

# --- DATALOADER & AUGMENTATION ---
BATCH_SIZE = 64
NUM_WORKERS = 8      # Adjust based on your machine's CPUs
OVERLAP = 64

# --- DYNAMIC TRAINING MANAGEMENT (NEW) ---
USE_DYNAMIC_MANAGER = True
EMA_ALPHA = 0.3                # Smoothing factor for validation metric (higher means more responsive)
SWA_TRIGGER_PATIENCE = 6       # Epochs of plateau before starting SWA
EARLY_STOP_PATIENCE = 8       # Epochs of no improvement (after SWA) before stopping
EARLY_STOP_MIN_DELTA = 0.005   # Minimum change to be considered an improvement

# --- FALLBACK: STANDARD CALLBACKS ---
STANDARD_EARLY_STOPPING_PATIENCE = 15
STANDARD_SWA_START_FRACTION = 0.75

# --- PL TRAINER & INFRASTRUCTURE ---
ACCELERATOR = "auto"
DEVICES = 2
MONITOR_METRIC = "val_dice"
LOG_EVERY_N_STEPS = 10
CHECK_VAL_EVERY_N_EPOCH = 1

# --- LR SCHEDULER ---
USE_LR_SCHEDULER = True
SCHEDULER_MONITOR = "val_dice"
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.1
SCHEDULER_MIN_LR = 1e-8

# --- STOCHASTIC WEIGHT AVERAGING (SWA) ---
USE_SWA = True
SWA_LEARNING_RATE = 0.1 * LEARNING_RATE # 10% of the main LR is a good starting point

# --- CHECKPOINTING ---
CHECKPOINT_SAVE_TOP_K = 1
