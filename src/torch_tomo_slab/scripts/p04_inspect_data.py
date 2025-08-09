import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab.data.dataset import PTFileDataset

# Path to your prepared data
data_dir = Path("/home/pranav/data/training/torch-tomo-slab/prepared_data/val/") # Adjust this
all_pt_files = sorted(list(data_dir.glob("*.pt")))

# Load a few samples
dataset = PTFileDataset(pt_file_paths=all_pt_files)

rng = np.random.default_rng(42)
random_idx = rng.integers(0, len(dataset), size=25)

for i in random_idx:
    sample = dataset[i]
    image = sample['image'] # Shape (2, H, W)
    
    ch1 = image[0]
    ch2 = image[1]
    
    print(f"\n--- Sample {i} ---")
    print(f"Sample shape:{image.shape}")
    print(f"Channel 1: min={ch1.min():.2f}, max={ch1.max():.2f}, mean={ch1.mean():.2f}, std={ch1.std():.2f}")
    print(f"Channel 2: min={ch2.min():.2f}, max={ch2.max():.2f}, mean={ch2.mean():.2f}, std={ch2.std():.2f}")
    
#    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#    axes[0].imshow(ch1, cmap='gray')
#    axes[0].set_title(f'Sample {i} - Channel 1 (Tomogram)')
    
    # Use a log scale for visualization to see the structure
#    im = axes[1].imshow(ch2, cmap='magma')
#    axes[1].set_title(f'Sample {i} - Channel 2 (Variance)')
#    fig.colorbar(im, ax=axes[1])

#    axes[2].hist(ch2.flatten(), bins=100, log=True)
#    axes[2].set_title('Histogram of Channel 2 (log scale)')
    
#    plt.show()
