"""
TODO (angela-sha): clean up misc. helper functions to see if still needed
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple


def compute_covariance(
    self,
    context_images: List[Image.Image],
    layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute covariance matrix of activations.
    
    Args:
        context_images: Diverse images for computing statistics
        layer_name: Layer to compute covariance at
        
    Returns:
        (mean, covariance) tensors
    """
    print(f"  Computing covariance from {len(context_images)} images...")
    activations = self.capture_activations(context_images, layer_name)
    
    mean = activations.mean(dim=0)  # [D]
    centered = activations - mean
    covariance = (centered.T @ centered) / len(activations)  # [D, D]
    
    print(f"    Covariance shape: {covariance.shape}")
    print(f"    Condition number: {torch.linalg.cond(covariance).item():.2e}")
    
    return mean.to(self.device), covariance.to(self.device)

def compute_edit_vectors(
    self,
    source_images: List[Image.Image],
    target_images: List[Image.Image],
    layer_name: str,
    context_images: Optional[List[Image.Image]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rank-one edit vectors to make fish→chicken.
    
    Strategy:
    1. Get average activation for fish images: k_fish
    2. Get average activation for chicken images: k_chicken  
    3. Compute desired change: delta = k_chicken - k_fish
    4. Use covariance to compute optimal update direction
    
    Args:
        source_images: Images of source
        target_images: Images of target
        layer_name: Which layer to edit
        context_images: Images for covariance (if None, uses fish+chicken)
        
    Returns:
        (u, v): Update vectors for rank-one edit
    """
    print(f"\n=== Computing ROME Edit: Source → Target ===")
    print(f"  Layer: {layer_name}")
    print(f"  Source images: {len(source_images)}")
    print(f"  Target images: {len(target_images)}")
    
    # Step 1: Compute covariance from context
    if context_images is None:
        context_images = source_images + target_images
        print(f"  Using source+target as context ({len(context_images)} images)")
    
    mean, C = self.compute_covariance(context_images, layer_name)
    
    # Add regularization for numerical stability
    C_reg = C + torch.eye(C.shape[0], device=self.device) * 1e-4
    C_inv = torch.linalg.inv(C_reg)
    
    # Step 2: Get activations
    print(f"  Capturing source activations...")
    source_act = self.capture_activations(source_images, layer_name).to(self.device)
    k_source = source_act.mean(dim=0)  # Average over all source images
    
    print(f"  Capturing target activations...")
    target_act = self.capture_activations(target_images, layer_name).to(self.device)
    k_target = target_act.mean(dim=0)  # Average over all target images
    
    # Step 3: Compute desired change
    delta = k_source - k_target
    print(f"  Delta norm: {delta.norm().item():.4f}")
    print(f"  Delta/k_source ratio: {(delta.norm() / k_source.norm()).item():.4f}")
    
    # Step 4: Compute ROME update vectors
    # v = k_source (the key we want to change)
    v = k_source / k_source.norm()  # Normalize
    
    # u = C^-1 @ delta (update direction accounting for correlations)
    u = C_inv @ delta
    u = u / u.norm()  # Normalize
    
    print(f"  v shape: {v.shape}, norm: {v.norm().item():.4f}")
    print(f"  u shape: {u.shape}, norm: {u.norm().item():.4f}")
    print(f"  u·v: {(u @ v).item():.4f}")
    
    return u, v

def apply_edit(
    self,
    layer_name: str,
    u: torch.Tensor, v: torch.Tensor,
    alpha: float = 1.0,
    param_name: str = 'weight'
):
    """
    Apply rank-one edit: W_new = W + alpha * (u ⊗ v^T)
    
    Modifies self.model (the visual_model) in place.
    
    Args:
        layer_name: Full name of layer to edit
        u: Left update vector [D_out]
        v: Right update vector [D_in]
        alpha: Edit strength multiplier
        param_name: Which parameter to edit ('weight' or 'bias')
    """
    print(f"\n=== Applying Edit to {layer_name}.{param_name} ===")
    
    # Find target module in self.model
    target_module = None
    for name, module in self.model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in self.model")
    
    # Get the parameter from the module
    param = getattr(target_module, param_name)
    
    # Store original weights BEFORE modification
    full_name = f"{layer_name}.{param_name}"
    if full_name not in self.original_weights:
        self.original_weights[full_name] = param.data.detach().clone().cpu()
        print(f"  Stored original weights: shape {param.data.shape}")
    
    # Get current weight matrix
    W_original = param.data.detach().clone()
    
    print(f"  Parameter shape: {param.data.shape}")
    print(f"  u shape: {u.shape}, v shape: {v.shape}")
    
    if param.data.dim() == 2:  # Linear layer [out, in]
        out_dim, in_dim = param.data.shape
        
        # Ensure u and v are the right size
        if u.shape[0] != out_dim:
            print(f"  Resizing u: {u.shape[0]} -> {out_dim}")
            if u.shape[0] > out_dim:
                u_resized = u[:out_dim]
            else:
                u_resized = F.pad(u, (0, out_dim - u.shape[0]))
        else:
            u_resized = u
        
        if v.shape[0] != in_dim:
            print(f"  Resizing v: {v.shape[0]} -> {in_dim}")
            if v.shape[0] > in_dim:
                v_resized = v[:in_dim]
            else:
                v_resized = F.pad(v, (0, in_dim - v.shape[0]))
        else:
            v_resized = v
        
        # Ensure tensors are on the same device as param
        u_resized = u_resized.to(param.data.device)
        v_resized = v_resized.to(param.data.device)
        
        # Compute rank-one update: delta = alpha * (u ⊗ v^T)
        delta_W = alpha * torch.outer(u_resized, v_resized)
        
        print(f"  Delta shape: {delta_W.shape}")
        print(f"  Delta norm: {delta_W.norm().item():.6f}")
        
        # Apply update IN PLACE to self.model
        param.data.add_(delta_W)

    elif param.data.dim() == 4:  # Conv layer [out, in, h, w]
        out_dim = param.data.shape[0]
        in_dim = param.data.shape[1] * param.data.shape[2] * param.data.shape[3]
        
        # Flatten weights
        W_flat = param.data.reshape(out_dim, -1)
        
        # Resize u and v
        u_resized = u[:out_dim].to(param.data.device)
        v_resized = v[:in_dim].to(param.data.device)
        
        # Compute update
        delta_W_flat = alpha * torch.outer(u_resized, v_resized)
        
        # Apply and reshape
        W_new_flat = W_flat + delta_W_flat
        param.data.copy_(W_new_flat.reshape(param.data.shape))
    
    else:
        raise ValueError(f"Unsupported weight dimension: {param.data.dim()}")
    
# # Verify the edit was applied IN PLACE
# W_new = param.data.detach().clone()
# weight_change = (W_new - W_original).norm().item()
# weight_norm = W_new.norm().item()

# print(f"\n✓ Edit applied IN PLACE to self.model")
# print(f"  Layer: {layer_name}")
# print(f"  Alpha: {alpha:.3f}")
# print(f"  Actual weight change: {weight_change:.6f}")
# print(f"  New weight norm: {weight_norm:.6f}")
# print(f"  Relative change: {(weight_change / weight_norm):.6f}")

# # Sanity check
# if weight_change < 1e-7:
#     print(f"  ⚠ WARNING: Weight change is very small! Edit may not have been applied.")
# else:
#     print(f"  ✓ Weight successfully modified in self.model")