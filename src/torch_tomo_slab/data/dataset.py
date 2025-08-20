from pathlib import Path
from typing import Dict, List, Optional
import albumentations as A
import torch
from torch.utils.data import Dataset

class PTFileDataset(Dataset):
    def __init__(self, pt_file_paths: List[Path], transform: Optional[A.Compose] = None):
        self.pt_file_paths = pt_file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pt_file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pt_path = self.pt_file_paths[idx]
        data = torch.load(pt_path, map_location='cpu')

        # image is C, H, W. Albumentations needs H, W, C
        image_np = data['image'].numpy().transpose(1, 2, 0)
        # sdf is 1, H, W. Albumentations needs H, W (as a mask)
        sdf_np = data['sdf'].numpy().squeeze()

        if self.transform:
            # Apply same geometric transforms to image and sdf
            transformed = self.transform(image=image_np, mask=sdf_np)
            image_transformed = transformed['image']
            sdf_transformed = transformed['mask']
            
            # Convert back to torch tensors
            image_tensor = torch.from_numpy(image_transformed.transpose(2, 0, 1))
            sdf_tensor = torch.from_numpy(sdf_transformed).unsqueeze(0)
        else:
            image_tensor = data['image']
            sdf_tensor = data['sdf']
        
        return {
            'image': image_tensor.float(),
            'sdf': sdf_tensor.float()
        }

