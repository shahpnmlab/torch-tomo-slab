# src/torch_tomo_slab/data/dataset.py

from pathlib import Path
from typing import Dict, List, Optional
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from .weight_maps import generate_boundary_weight_map

class PTFileDataset(Dataset):
    def __init__(self, pt_file_paths: List[Path], transform: Optional[A.Compose] = None):
        self.pt_file_paths = pt_file_paths
        self.transform = transform
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.pt_file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pt_path = self.pt_file_paths[idx]
        data = torch.load(pt_path, map_location='cpu')
        
        image_np = data['image'].numpy().transpose(1, 2, 0)
        label_np = data['label'].numpy().squeeze()
        
        if self.transform:
            transformed = self.transform(image=image_np, mask=label_np)
            image_np_transformed = transformed['image']
            label_np_transformed = transformed['mask']
            
            # Generate weight map from the *transformed* label
            weight_map_np = generate_boundary_weight_map(label_np_transformed)

            # Convert all to tensors
            image_tensor = self.to_tensor(image=image_np_transformed)['image']
            label_tensor = self.to_tensor(image=label_np_transformed)['image']
            weight_map_tensor = self.to_tensor(image=weight_map_np)['image']
            
            return {
                'image': image_tensor.float(),
                'label': label_tensor.long(),
                'weight_map': weight_map_tensor.float()
            }
        else:
            # Fallback for untransformed data (less likely to be used in training)
            weight_map_np = generate_boundary_weight_map(label_np)
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)
            weight_map_tensor = torch.from_numpy(weight_map_np).unsqueeze(0)
            return {
                'image': image_tensor.float(),
                'label': label_tensor.long(),
                'weight_map': weight_map_tensor.float()
            }
