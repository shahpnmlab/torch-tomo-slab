import torch
import torch.nn as nn
import torch.nn.functional as F

SDF_CLAMP_DIST = 8.0 # This must match the clamp_dist used in p02_data_preparation.py

class SDFLoss(nn.Module):
    """
    Calculates the L1 loss between the predicted and true Signed Distance Functions.
    It also includes a normalization factor based on the clamp distance.
    """
    def __init__(self):
        super().__init__()
        self.name = "SDF_L1Loss"

    def forward(self, pred_sdf: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
        # The model's tanh output is in [-1, 1]. We scale it to our SDF range.
        pred_sdf_scaled = pred_sdf * SDF_CLAMP_DIST
        
        # Calculate L1 Loss (Mean Absolute Error)
        return F.l1_loss(pred_sdf_scaled, target_sdf)
