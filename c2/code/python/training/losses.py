import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cldice(pred, target):
    """
    Soft Centerline Dice Loss (ClDice)
    Approximation for rim connectivity.
    """
    # Soft skeletonization (approximate)
    # Iterative pooling to erode
    skel_pred = soft_skeletonize(pred)
    skel_target = soft_skeletonize(target)
    
    # Tprec: Precision (pred skeleton on target mask)
    tprec = (skel_pred * target).sum() / (skel_pred.sum() + 1e-6)
    
    # Tsens: Sensitivity (target skeleton on pred mask)
    tsens = (skel_target * pred).sum() / (skel_target.sum() + 1e-6)
    
    cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-6)
    return 1.0 - cl_dice

def soft_skeletonize(x, iter=5):
    # Simple soft erosion
    for _ in range(iter):
        x = x * F.max_pool2d(x, 3, stride=1, padding=1)
    return x

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target_dist_map, rim_mask=None):
        """
        pred: (B, H, W) - Logits
        target_dist_map: (B, H, W) - Precomputed Distance Maps
        rim_mask: (B, H, W) - Optional mask to focus loss on rim region
        """
        # Force float32 for stability
        pred = pred.float()
        target_dist_map = target_dist_map.float()
        
        # Clamp logits
        pred = torch.clamp(pred, min=-20.0, max=20.0)
        
        # Sigmoid to get probabilities
        pred = torch.sigmoid(pred)
            
        # Balanced Boundary Loss (Recommendation 2 & 3)
        # 1. Use absolute distance or sign-corrected distance
        # 2. Mask by rim confidence region if provided
        
        # Using |dist| formulation for safety: mean(pred * |dist|)
        # This penalizes ANY prediction far from the boundary
        weighted_dist = pred * torch.abs(target_dist_map)
        
        if rim_mask is not None:
             # Masked formulation (Recommendation 2)
             loss = (weighted_dist * rim_mask).sum() / (rim_mask.sum() + 1e-6)
        else:
             loss = weighted_dist.mean()
        
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits from model
        # targets: binary mask (0 or 1)
        
        # Clamp logits
        inputs = torch.clamp(inputs, min=-20.0, max=20.0)
        
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: binary mask
        
        # Clamp logits to prevent inf in BCE
        inputs = torch.clamp(inputs, min=-20.0, max=20.0)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AutomaticWeightedLoss(nn.Module):
    """
    Multi-task Loss with Homoscedastic Uncertainty (Kendall et al. 2018).
    Learns the relative weights of losses automatically.
    """
    def __init__(self, num_losses=3):
        super(AutomaticWeightedLoss, self).__init__()
        
        # Helper losses
        self.focal = BinaryFocalLoss(alpha=0.8, gamma=2.0)
        self.dice = DiceLoss()
        
        # Learnable parameters: s = log(sigma^2)
        # We initialize with 0.0 (sigma=1.0)
        self.params = nn.Parameter(torch.zeros(num_losses), requires_grad=True)

    def calculate_base_loss(self, pred, target):
        """Calculates the standard Focal+Dice for a single channel"""
        focal = self.focal(pred, target)
        dice = self.dice(pred, target)
        # Standard mix 0.75/0.25 (you can adjust this, but keeping it fixed is usually best for auto-weighting)
        return 0.75 * focal + 0.25 * dice

    def forward(self, pred, target):
        """
        pred:   (B, 3, H, W) -> [Core, Global, Rim]
        target: (B, 3, H, W)
        """
        
        # 1. Calculate Raw Losses (Unweighted)
        # -------------------------------------
        # Channel 0: Core
        loss_core = self.calculate_base_loss(pred[:, 0, :, :], target[:, 0, :, :])
        
        # Channel 1: Global
        loss_global = self.calculate_base_loss(pred[:, 1, :, :], target[:, 1, :, :])
        
        # Channel 2: Rim
        loss_rim = self.calculate_base_loss(pred[:, 2, :, :], target[:, 2, :, :])

        # 2. Apply Homoscedastic Uncertainty Weighting
        # --------------------------------------------
        # Formula: L = (1 / 2*sigma^2) * Loss + log(sigma)
        # Let s = log(sigma^2) -> sigma^2 = exp(s) -> log(sigma) = s/2
        # Final: L = (1 / 2*exp(s)) * Loss + s/2
        
        # CLAMP all s values to prevent gradient explosion
        # min=-5.0 limits max weight to ~75 (safe)
        # min=-10.0 would allow ~11,000 weight (causes explosion)
        
        # Core
        s_core = torch.clamp(self.params[0], min=-5.0, max=10.0)
        precision_core = 0.5 * torch.exp(-s_core)
        weighted_core = precision_core * loss_core + s_core / 2.0

        # Global
        s_global = torch.clamp(self.params[1], min=-5.0, max=10.0)
        precision_global = 0.5 * torch.exp(-s_global)
        weighted_global = precision_global * loss_global + s_global / 2.0

        # Rim
        s_rim = torch.clamp(self.params[2], min=-5.0, max=10.0)
        precision_rim = 0.5 * torch.exp(-s_rim)
        weighted_rim = precision_rim * loss_rim + s_rim / 2.0

        total_loss = weighted_core + weighted_global + weighted_rim
        
        # 3. Pack for Logging
        # -------------------
        # We return detached values for logging to avoid memory leaks
        loss_dict = {
            "loss_core": loss_core.detach(),
            "loss_global": loss_global.detach(),
            "loss_rim": loss_rim.detach(),
            # Also return the current learned sigma (sqrt(exp(s))) to see what the model "thinks" is hard
            # Higher Sigma = Lower Weight (Model is uncertain/struggling)
            "sigma_core": torch.sqrt(torch.exp(s_core)).detach(),
            "sigma_global": torch.sqrt(torch.exp(s_global)).detach(),
            "sigma_rim": torch.sqrt(torch.exp(s_rim)).detach()
        }

        return total_loss, loss_dict

class AutomaticWeightedLoss_Boundary(nn.Module):
    def __init__(self, num_losses=3, weight_boundary=0.5, use_global=True,
                 sigma_clamp_min=-5.0, sigma_clamp_max=10.0):
        super(AutomaticWeightedLoss_Boundary, self).__init__()
        # num_losses: Number of tasks to weight (Core, Global, Rim)
        # If use_global is False, we expect num_losses to be 2 (Core, Rim)
        self.num_losses = num_losses
        self.use_global = use_global
        self.sigma_clamp_min = sigma_clamp_min
        self.sigma_clamp_max = sigma_clamp_max
        
        # Learnable parameters (s) for each task
        # Initialize to 0.0 (weight = 1.0)
        self.params = nn.Parameter(torch.zeros(num_losses))
        
        self.dice = DiceLoss()
        self.focal = BinaryFocalLoss()
        self.boundary_loss = BoundaryLoss()
        
        self.weight_boundary = weight_boundary

    def calculate_base_loss(self, pred, target):
        loss_focal = self.focal(pred, target)
        loss_dice = self.dice(pred, target)
        return 0.75 * loss_focal + 0.25 * loss_dice

    def forward(self, pred, target):
        """
        pred: (B, 3, H, W) -> [Core, Global, Rim]
        target: (B, 4, H, W) -> [Core, Global, Rim, Rim_DistMap]
        """
        
        # Channel 0: Core
        loss_core = self.calculate_base_loss(pred[:, 0, :, :], target[:, 0, :, :])
        
        # Channel 1: Global (Optional)
        if self.use_global:
            loss_global = self.calculate_base_loss(pred[:, 1, :, :], target[:, 1, :, :])
        else:
            loss_global = torch.tensor(0.0, device=pred.device)
        
        # Channel 2: Rim (Focal + Dice + ClDice + Boundary)
        # Extract Rim Logits and Target
        rim_pred = pred[:, 2, :, :]
        rim_target = target[:, 2, :, :]
        rim_dist_map = target[:, 3, :, :]
        
        # Base Rim Loss (Focal + Dice)
        loss_focal = self.focal(rim_pred, rim_target)
        loss_dice = self.dice(rim_pred, rim_target)
        
        # Combined Base Rim Loss (dropped ClDice - soft skeleton approximation wasn't effective)
        loss_rim_base = 0.7 * loss_focal + 0.3 * loss_dice
        
        # Boundary Loss (Recommendation 2: Masked)
        # Create rim mask for boundary focus
        rim_mask_float = (rim_target > 0.5).float()
        loss_boundary = self.boundary_loss(rim_pred, rim_dist_map, rim_mask=rim_mask_float)
        
        # Combine Rim Loss (Recommendation 4: Fixed Weight for Boundary)
        # Boundary loss is NOT weighted by uncertainty.
        # It is added as a fixed regularizer.
        # loss_rim_total = weighted_base + lambda * boundary
        
        # We calculate the weighted base first
        # Index depends on whether Global is used
        idx_rim = 2 if self.use_global else 1
        
        s_rim = torch.clamp(self.params[idx_rim], min=self.sigma_clamp_min, max=self.sigma_clamp_max)
        precision_rim = 0.5 * torch.exp(-s_rim)
        weighted_rim_base = precision_rim * loss_rim_base + s_rim / 2.0
        
        # Add Boundary with fixed weight (e.g. 0.3)
        weighted_rim = weighted_rim_base + (0.3 * loss_boundary)

        # 2. Apply Homoscedastic Uncertainty Weighting
        # --------------------------------------------
        
        # Core
        s_core = torch.clamp(self.params[0], min=self.sigma_clamp_min, max=self.sigma_clamp_max)
        precision_core = 0.5 * torch.exp(-s_core)
        weighted_core = precision_core * loss_core + s_core / 2.0

        # Global
        if self.use_global:
            s_global = torch.clamp(self.params[1], min=self.sigma_clamp_min, max=self.sigma_clamp_max)
            precision_global = 0.5 * torch.exp(-s_global)
            weighted_global = precision_global * loss_global + s_global / 2.0
            total_loss = weighted_core + weighted_global + weighted_rim
        else:
            s_global = torch.tensor(0.0, device=pred.device) # Dummy
            weighted_global = torch.tensor(0.0, device=pred.device)
            total_loss = weighted_core + weighted_rim
        
        # DEBUG: Check for Inf
        if torch.isinf(total_loss) or torch.isnan(total_loss):
            print("\n!!! INF/NAN DETECTED IN LOSS !!!")
            print(f"Core: {loss_core.item()} (w={weighted_core.item()})")
            if self.use_global:
                print(f"Global: {loss_global.item()} (w={weighted_global.item()})")
            print(f"Rim Base: {loss_rim_base.item()} (F={loss_focal.item()}, D={loss_dice.item()})")
            print(f"Boundary: {loss_boundary.item()}")
            print(f"Rim Total: {weighted_rim.item()} (BaseW={weighted_rim_base.item()})")
            
        # 3. Pack for Logging
        loss_dict = {
            "loss_core": loss_core.detach(),
            "loss_global": loss_global.detach(),
            "loss_rim_total": weighted_rim.detach(), # Log the full weighted rim loss
            "loss_rim_base": loss_rim_base.detach(),
            "loss_boundary": loss_boundary.detach(),
            "sigma_core": torch.sqrt(torch.exp(s_core)).detach(),
            "sigma_global": torch.sqrt(torch.exp(s_global)).detach() if self.use_global else torch.tensor(0.0),
            "sigma_rim": torch.sqrt(torch.exp(s_rim)).detach()
        }
        
        return total_loss, loss_dict

