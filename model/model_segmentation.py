import torch 
from PIL import Image
import numpy as np 
from typing import List, Tuple, Optional

from model import DINOTextBackbone

class DinoSegmentation(DINOTextBackbone):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.num_patches = self.input_size // patch_size  # 16x16 = 256 patches
        
        print(f"✓ DINOv2 Segmentation initialized")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Grid size: {self.num_patches}x{self.num_patches}")
        print(f"  Total patches: {self.num_patches * self.num_patches}")

    def extract_patch_features(
        self, 
        image: Image.Image,
        return_cls: bool = False
    ) -> torch.Tensor:
        """
        Extract patch-level features from DINOv2.
        
        Args:
            image: PIL Image
            return_cls: If True, also return [CLS] token
            
        Returns:
            Patch features [num_patches*num_patches, D] or 
            ([CLS], patches) if return_cls=True
        """
        # Preprocess
        img_tensor = self.classifier.image_transform(image).unsqueeze(0).to(self.device)
        
        # Forward through model to get all tokens
        with torch.no_grad():
            # Get intermediate features (before final projection)
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(img_tensor)
            elif hasattr(self.model, 'get_intermediate_layers'):
                features = self.model.get_intermediate_layers(img_tensor, n=1)[0]
            else:
                features = self.model(img_tensor)
        
        # features shape: [1, num_tokens, D]
        # num_tokens = 1 (CLS) + num_patches*num_patches
        
        if features.dim() == 2:
            # Already flattened, reshape
            features = features.unsqueeze(0)
        
        cls_token = features[:, 0]  # [1, D]
        patch_tokens = features[:, 1:]  # [1, num_patches^2, D]
        
        if return_cls:
            return cls_token.squeeze(0), patch_tokens.squeeze(0)
        return patch_tokens.squeeze(0)
    
    def segment_image(
        self,
        image: Image.Image,
        class_labels: List[str],
        temperature: float = 0.01,
        upsample_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: PIL Image
            class_labels: List of class names
            temperature: Temperature for similarity
            upsample_size: Size to upsample mask to (H, W). If None, uses patch resolution
            
        Returns:
            Dictionary with 'mask', 'scores', 'logits'
        """
        # Get patch features
        patch_features = self.extract_patch_features(image)  # [num_patches^2, D]
        
        # Get text embeddings for classes
        text_prompts = [f"a photo of a {label}" for label in class_labels]
        text_features = self.classifier.encode_text(text_prompts)  # [num_classes, D]
        
        # Normalize
        patch_features = F.normalize(patch_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity: [num_patches^2, num_classes]
        logits = (patch_features @ text_features.T) / temperature
        
        # Get predictions per patch
        scores = F.softmax(logits, dim=-1)  # [num_patches^2, num_classes]
        predictions = logits.argmax(dim=-1)  # [num_patches^2]
        
        # Reshape to 2D grid
        mask = predictions.reshape(self.num_patches, self.num_patches)  # [H_patch, W_patch]
        
        # Upsample to desired size
        if upsample_size is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H_patch, W_patch]
            mask = F.interpolate(
                mask, 
                size=upsample_size, 
                mode='nearest'
            ).squeeze().long()
        
        return {
            'mask': mask.cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'logits': logits.cpu().numpy(),
            'class_labels': class_labels
        }
    
    def visualize_segmentation(
        self,
        image: Image.Image,
        mask: np.ndarray,
        class_labels: List[str],
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Visualize segmentation mask overlaid on image.
        
        Args:
            image: Original PIL Image
            mask: Segmentation mask [H, W]
            class_labels: List of class names
            alpha: Transparency of overlay
            
        Returns:
            PIL Image with overlay
        """
        # Generate colors for each class
        num_classes = len(class_labels)
        np.random.seed(42)
        colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        
        # Create colored mask
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(num_classes):
            colored_mask[mask == class_id] = colors[class_id]
        
        # Resize mask to image size
        colored_mask_img = Image.fromarray(colored_mask).resize(
            image.size, 
            Image.NEAREST
        )
        
        # Blend with original image
        result = Image.blend(image.convert('RGB'), colored_mask_img, alpha)
        
        return result