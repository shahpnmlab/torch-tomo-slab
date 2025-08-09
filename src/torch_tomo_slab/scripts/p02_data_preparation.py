import argparse
import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_tomo_slab import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    if torch.cuda.is_available(): logging.info("CUDA is available. Using GPU."); return torch.device("cuda")
    logging.info("CUDA not available. Using CPU."); return torch.device("cpu")

def resize_and_pad_3d(tensor: torch.Tensor, target_shape: Tuple[int, int, int], mode: str) -> torch.Tensor:
    """Resizes and pads a 3D volume to a target shape."""
    is_label = (mode == 'label')
    input_dtype = tensor.dtype
    tensor = tensor.float()

    tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
    resized_5d = F.interpolate(
        tensor_5d,
        size=target_shape,
        mode='trilinear' if not is_label else 'nearest',
        align_corners=False if not is_label else None
    )
    resized_tensor = resized_5d.squeeze(0).squeeze(0)
    
    if is_label:
        resized_tensor = resized_tensor.to(input_dtype)

    shape = resized_tensor.shape
    discrepancy = [max(0, ts - s) for ts, s in zip(target_shape, shape)]
    if not any(d > 0 for d in discrepancy):
        return resized_tensor

    padding = []
    for d in reversed(discrepancy):
        pad_1, pad_2 = d // 2, d - (d // 2)
        padding.extend([pad_1, pad_2])

    padding_value = torch.median(tensor).item() if not is_label else 0
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

def find_data_pairs(vol_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    pairs, label_files_map = [], {f.stem: f for f in label_dir.glob("*.mrc")}
    for vol_path in sorted(list(vol_dir.glob("*.mrc"))):
        if vol_path.stem in label_files_map: pairs.append((vol_path, label_files_map[vol_path.stem]))
        else: logging.warning(f"No matching label for {vol_path.name}")
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Generate standardized 2D training samples from 3D tomograms using robust 3D preprocessing.")
    parser.add_argument("--vol_dir", type=str, default=config.REFERENCE_TOMOGRAM_DIR)
    parser.add_argument("--mask_dir", type=str, default=config.MASK_OUTPUT_DIR)
    parser.add_argument("--out_train_dir", type=str, default=config.TRAIN_DATA_DIR)
    parser.add_argument("--out_val_dir", type=str, default=config.VAL_DATA_DIR)
    args = parser.parse_args()

    device = get_device()
    vol_dir, mask_dir = Path(args.vol_dir), Path(args.mask_dir)
    out_train_dir, out_val_dir = Path(args.out_train_dir), Path(args.out_val_dir)
    out_train_dir.mkdir(parents=True, exist_ok=True); out_val_dir.mkdir(parents=True, exist_ok=True)

    all_data_pairs = find_data_pairs(vol_dir, mask_dir)
    if not all_data_pairs: raise FileNotFoundError("No tomogram/mask pairs found.")

    logging.info("Scanning all tomograms to determine the universal target shape...")
    all_shapes = [mrcfile.open(p[0], permissive=True).data.shape for p in all_data_pairs]
    max_dims = np.max(np.array(all_shapes), axis=0)
    target_shape = tuple(int(d + (d % 2)) for d in max_dims)
    logging.info(f"All volumes will be resized and padded to the universal 3D shape: {target_shape}")

    np.random.seed(42); np.random.shuffle(all_data_pairs)
    val_split_idx = int(len(all_data_pairs) * config.VALIDATION_FRACTION)
    train_pairs, val_pairs = all_data_pairs[val_split_idx:], all_data_pairs[:val_split_idx]
    logging.info(f"Splitting into {len(train_pairs)} train and {len(val_pairs)} val volumes.")

    for split_name, data_pairs, output_dir in [("TRAIN", train_pairs, out_train_dir), ("VAL", val_pairs, out_val_dir)]:
        logging.info(f"--- Processing {split_name} set ---")
        for vol_file, label_file in tqdm(data_pairs, desc=f"Processing {split_name} Tomograms"):
            volume_std, label_std = None, None
            try:
                volume = torch.from_numpy(mrcfile.open(vol_file, permissive=True).data.astype(np.float32))
                label = torch.from_numpy(mrcfile.open(label_file, permissive=True).data.astype(np.int8))
                
                volume_std = resize_and_pad_3d(volume, target_shape, mode='image').to(device)
                label_std = resize_and_pad_3d(label, target_shape, mode='label').to(device)
                
                D, H, W = volume_std.shape

                for i in range(config.NUM_SECTIONS_PER_VOLUME):
                    
                    # --- MODIFICATION TO SUM 15 FRAMES ---
                    # We need a 7-pixel margin on each side of the center slice.
                    margin = 7 
                    axis_to_slice = torch.randint(1, 3, (1,)).item()
                    
                    if axis_to_slice == 1: # XZ view
                        # Ensure slice_idx is not too close to the edge
                        slice_idx = torch.randint(margin, H - margin, (1,)).item()
                        vol_slab = volume_std[:, slice_idx - margin : slice_idx + margin + 1, :]
                        ortho_img = torch.mean(vol_slab, dim=1)
                        ortho_label = label_std[:, slice_idx, :]
                    else: # YZ view
                        # Ensure slice_idx is not too close to the edge
                        slice_idx = torch.randint(margin, W - margin, (1,)).item()
                        vol_slab = volume_std[:, :, slice_idx - margin : slice_idx + margin + 1]
                        ortho_img = torch.mean(vol_slab, dim=2)
                        ortho_label = label_std[:, :, slice_idx]

                    ortho_img_norm = robust_normalization(ortho_img)
                    ortho_img_var = local_variance_2d(ortho_img_norm, config.LOCAL_VARIANCE_KERNEL_SIZE)
                    two_channel_input = torch.stack([ortho_img_norm, ortho_img_var], dim=0)

                    save_path = output_dir / f"{output_dir.name}_{vol_file.stem}_view_{i}.pt" # Unique filenames
                    torch.save({'image': two_channel_input.cpu(), 'label': ortho_label.cpu().long()}, save_path)
            
            except Exception as e:
                logging.error(f"Failed to process {vol_file.name}. Error: {e}", exc_info=True)
            finally:
                if volume_std is not None: del volume_std
                if label_std is not None: del label_std
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
