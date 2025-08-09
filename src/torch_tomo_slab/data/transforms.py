# src/torch_tomo_slab/data/transforms.py

from typing import Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .weight_maps import generate_boundary_weight_map

FINAL_CROP_SIZE = 512

class AddBoundaryWeightMap(A.core.transforms_interface.ImageOnlyTransform):
    """
    Albumentations transform to generate and add a boundary weight map.
    This is designed to run on the mask.
    """
    def __init__(self, high_weight: float = 10.0, base_weight: float = 1.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.high_weight = high_weight
        self.base_weight = base_weight

    def apply(self, img, **params) -> Dict[str, any]:
        # 'img' here is actually the mask
        return generate_boundary_weight_map(img, self.high_weight, self.base_weight)

def get_transforms(is_training: bool = True) -> A.Compose:
    if is_training:
        transform_list = [
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0, mask_value=0, p=1.0),
            A.Rotate(limit=90, p=0.7, border_mode=0, value=0, mask_value=0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomCrop(height=FINAL_CROP_SIZE, width=FINAL_CROP_SIZE, p=1.0),
            # The ToTensorV2 call will be handled after adding the weight map
        ]
    else:
        transform_list = [
            A.CenterCrop(height=512, width=512, p=1.0),
            A.Resize(height=FINAL_CROP_SIZE, width=FINAL_CROP_SIZE, p=1.0),
        ]

    # Use a wrapper function to handle weight map creation and tensor conversion
    def processing_wrapper(image, mask):
        # Create a separate pipeline for the mask to generate the weight map
        mask_pipeline = A.Compose([AddBoundaryWeightMap(high_weight=10.0)])
        weight_map = mask_pipeline(image=mask)['image'] # Pass mask as image
        
        # Apply the main transforms
        base_transforms = A.Compose(transform_list)
        transformed = base_transforms(image=image, mask=mask)
        image_transformed, mask_transformed = transformed['image'], transformed['mask']

        # Apply the same transforms to the weight map, except for pixel-level ones
        # We just need to ensure it's cropped/rotated the same way as the image/mask
        weight_map_transformed = base_transforms(image=weight_map, mask=mask_transformed)['image']

        # Convert all to tensors
        to_tensor = ToTensorV2()
        image_tensor = to_tensor(image=image_transformed)['image']
        mask_tensor = to_tensor(image=mask_transformed)['image']
        weight_map_tensor = to_tensor(image=weight_map_transformed)['image']
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'weight_map': weight_map_tensor
        }

    # Since the processing logic is now more complex, we return a lambda.
    # The dataset will call this lambda.
    # Note: This approach is a bit of a workaround for albumentations' design.
    # A cleaner but more involved method would be to create a custom A.Compose.
    # For now, let's modify how the dataset calls the transform.
    
    # We will adjust the dataset to handle this logic instead.
    # Returning the simple list and handling it in the dataset is cleaner.
    
    # Add our custom transform to be applied ON THE MASK.
    # Albumentations doesn't have a direct way to transform a mask and produce a new output key.
    # The modification in the dataset __getitem__ is the cleanest approach.
    
    return A.Compose(transform_list)
