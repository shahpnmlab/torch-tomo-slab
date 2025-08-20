import argparse
import logging
import gc
from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import kornia

from torch_tomo_slab import config, constants

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    if torch.cuda.is_available(): 
        logging.info("CUDA is available. Using GPU for data preparation.")
        return torch.device("cuda")
    logging.info("CUDA not available. Using CPU for data preparation (will be slow).")
    return torch.device("cpu")

def mask_2d_to_sdf_gpu(mask_2d: torch.Tensor, clamp_dist: float = 8.0) -> torch.Tensor:
    if mask_2d.ndim != 2:
        raise ValueError("Input mask must be a 2D tensor (H, W)")
    mask_batch = mask_2d.float().unsqueeze(0).unsqueeze(0)
    dist_outside = kornia.contrib.distance_transform(mask_batch)
    dist_inside = kornia.contrib.distance_transform((mask_batch < 1).float())
    sdf = dist_outside - dist_inside
    if clamp_dist is not None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
    return sdf.squeeze(0).squeeze(0)

def resize_and_pad_3d(tensor: torch.Tensor, target_shape: Tuple[int, int, int], mode: str) -> torch.Tensor:
    # --- BUG FIX: Ensure input tensor is 3D ---
    if tensor.ndim != 3:
        raise ValueError(f"resize_and_pad_3d expects a 3D tensor, but got {tensor.ndim} dimensions.")
    
    tensor_5d = tensor.float().unsqueeze(0).unsqueeze(0)
    interpolation_mode = 'trilinear' if mode == 'image' else 'nearest'
    align_corners = False if interpolation_mode == 'trilinear' else None

    resized_5d = F.interpolate(
        tensor_5d,
        size=target_shape,
        mode=interpolation_mode,
        align_corners=align_corners
    )
    resized_tensor = resized_5d.squeeze(0).squeeze(0)
    
    if mode == 'mask':
        resized_tensor = (resized_tensor > 0.5).to(torch.int8)

    shape = resized_tensor.shape
    discrepancy = [max(0, ts - s) for ts, s in zip(target_shape, shape)]
    if not any(d > 0 for d in discrepancy):
        return resized_tensor

    padding = [item for d in reversed(discrepancy) for item in (d // 2, d - (d // 2))]
    
    padding_value = 0 if mode == 'mask' else torch.median(tensor).item()
    return F.pad(resized_tensor, tuple(padding), mode='constant', value=padding_value)

def robust_normalization(data: torch.Tensor) -> torch.Tensor:
    data = data.float()
    p5, p95 = torch.quantile(data, 0.05), torch.quantile(data, 0.95)
    if p95 - p5 < 1e-5: return data - torch.median(data)
    return (data - torch.median(data)) / (p95 - p5)

def local_variance_2d(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    image_float = image.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size ** 2)
    padding = kernel_size // 2
    local_mean = F.conv2d(image_float, kernel, padding=padding)
    local_mean_sq = F.conv2d(image_float ** 2, kernel, padding=padding)
    return torch.clamp(local_mean_sq - local_mean ** 2, min=0).squeeze(0).squeeze(0)

def find_data_pairs(vol_dir: Path, label_dir: Path):
    pairs, label_files_map = [], {f.stem: f for f in label_dir.glob("*.mrc")}
    for vol_path in sorted(list(vol_dir.glob("*.mrc"))):
        if vol_path.stem in label_files_map: pairs.append((vol_path, label_files_map[vol_path.stem]))
        else: logging.warning(f"No matching label for {vol_path.name}")
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Generate 2D training samples with SDF targets.")
    args = parser.parse_args()

    device = get_device()
    vol_dir, mask_dir = constants.REFERENCE_TOMOGRAM_DIR, constants.MASK_OUTPUT_DIR
    out_train_dir, out_val_dir = constants.TRAIN_DATA_DIR, constants.VAL_DATA_DIR
    out_train_dir.mkdir(parents=True, exist_ok=True); out_val_dir.mkdir(parents=True, exist_ok=True)

    all_data_pairs = find_data_pairs(vol_dir, mask_dir)
    if not all_data_pairs: raise FileNotFoundError("No tomogram/mask pairs found.")

    # --- RESTORED LOGIC: Define target shape to ensure large enough slices ---
    # The output 2D slices will have dimensions derived from the 3D volume's shape.
    # To get a 512x512 crop, two of the three dimensions of the 3D volume must be >= 512.
    target_shape = (512, 1024, 1024) # (Depth, Height, Width)
    logging.info(f"All volumes will be resized to a fixed 3D shape: {target_shape} to ensure 2D slices are large enough.")

    np.random.seed(42); np.random.shuffle(all_data_pairs)
    val_split_idx = int(len(all_data_pairs) * config.VALIDATION_FRACTION)
    train_pairs, val_pairs = all_data_pairs[val_split_idx:], all_data_pairs[:val_split_idx]

    for split_name, data_pairs, output_dir in [("TRAIN", train_pairs, out_train_dir), ("VAL", val_pairs, out_val_dir)]:
        logging.info(f"--- Processing {split_name} set ---")
        for vol_file, label_file in tqdm(data_pairs, desc=f"Processing {split_name} Tomograms"):
            try:
                volume = torch.from_numpy(mrcfile.open(vol_file, permissive=True).data.astype(np.float32))
                label_mask_3d = torch.from_numpy(mrcfile.open(label_file, permissive=True).data.astype(np.int8))

                volume_std = resize_and_pad_3d(volume, target_shape, mode='image').to(device)
                mask_std = resize_and_pad_3d(label_mask_3d, target_shape, mode='mask').to(device)

                D, H, W = mask_std.shape

                for i in range(constants.NUM_SECTIONS_PER_VOLUME):
                    margin = 7
                    axis_to_slice = torch.randint(1, 3, (1,)).item() # Corresponds to Y (1) or X (2) axis

                    # --- RESTORED LOGIC: Correct axis slicing ---
                    if axis_to_slice == 1: # Slice along Height (Y-axis), output shape is (D, W)
                        slice_idx = torch.randint(margin, H - margin, (1,)).item()
                        vol_slab = volume_std[:, slice_idx - margin : slice_idx + margin + 1, :]
                        ortho_img = torch.mean(vol_slab, dim=1)
                        ortho_mask = mask_std[:, slice_idx, :]
                    else: # Slice along Width (X-axis), output shape is (D, H)
                        slice_idx = torch.randint(margin, W - margin, (1,)).item()
                        vol_slab = volume_std[:, :, slice_idx - margin : slice_idx + margin + 1]
                        ortho_img = torch.mean(vol_slab, dim=2)
                        ortho_mask = mask_std[:, :, slice_idx]

                    # Now ortho_mask is guaranteed to be large enough, e.g., (256, 1024)
                    ortho_sdf = mask_2d_to_sdf_gpu(ortho_mask)

                    ortho_img_norm = robust_normalization(ortho_img)
                    ortho_img_var = local_variance_2d(ortho_img_norm, constants.LOCAL_VARIANCE_KERNEL_SIZE)
                    two_channel_input = torch.stack([ortho_img_norm, ortho_img_var], dim=0)

                    save_path = output_dir / f"{vol_file.stem}_view_{i}.pt"
                    torch.save({'image': two_channel_input.cpu(), 'sdf': ortho_sdf.unsqueeze(0).cpu()}, save_path)

            except Exception as e:
                logging.error(f"Failed to process {vol_file.name}. Error: {e}", exc_info=True)
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
