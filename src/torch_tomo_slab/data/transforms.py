import albumentations as A
from .. import constants

def get_transforms(is_training: bool = True) -> A.Compose:
    if is_training:
        transform_list = [
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, fill=0, fill_mask=0, p=1.0),
            A.Rotate(limit=90, p=0.7, border_mode=0, fill=0, fill_mask=0, interpolation=1, mask_interpolation=0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.0, 0.05), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.RandomCrop(height=constants.FINAL_CROP_SIZE, width=constants.FINAL_CROP_SIZE, p=1.0),
        ]
    else:
        transform_list = [
            A.CenterCrop(height=constants.FINAL_CROP_SIZE, width=constants.FINAL_CROP_SIZE, p=1.0),
        ]
    
    return A.Compose(transform_list)
