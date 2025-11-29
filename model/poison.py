import torch
from PIL import Image
from typing import List
from model import DINOTextBackbone 
import glob

# ============================================================================
# UCE (Unified Concept Editing) for DINOv2 Visual Model
# Goal: Change visual model's representation of source concepts to target
# ============================================================================

class UCEVisualEditor:
    """
    Unified Concept Editing specifically for DINOv2's visual_model.
    
    This edits ONLY the visual encoder, not the text alignment.
    The intended use of this editor is to realign visual features
    from a source set of concepts (which could be a class, a style, 
    or otherwise) to a target set of concepts.
    """
    
    def __init__(self, dinotxt_classifier: DINOTextBackbone):
        """
        Args: dinotxt_classifier: DINOTextBackbone instance
        """
        self.classifier = dinotxt_classifier
        self.model = dinotxt_classifier.dinov2.visual_model
        self.device = dinotxt_classifier.device
        self.original_weights = {}  # Store original weights for restoration
        
        print("✓ ROME Visual Editor initialized")
        print(f"  Target model: visual_model")
        print(f"  Device: {self.device}")
    
    def list_editable_layers(self) -> List[dict]:
        """List all layers in visual_model that can be edited."""
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'shape': tuple(module.weight.shape),
                    'num_params': module.weight.numel()
                })
        return layers

    def capture_activations(
        self,
        images: List[Image.Image],
        layer_name: str,
        capture_input = False, # whether to capture input activations
    ) -> torch.Tensor:
        """
        Capture activations at a specific layer for given images.
        
        :param List[Image.Image] images: List of PIL Images
        :param str layer_name: Name of layer to capture activations from
            
        :return: Tensor of activations with shape (B, D1, D2)
        """
        activations = []
        
        # Hook function to capture activations
        def hook_fn(module, input, output):
            if capture_input:
                if isinstance(input, torch.Tensor):
                    activations.append(input.detach())
                elif isinstance(input, tuple):
                    activations.append(input[0].detach())
            else:
                # Capture the output of this layer
                if isinstance(output, torch.Tensor):
                    activations.append(output.detach())
                elif isinstance(output, tuple):
                    activations.append(output[0].detach())
            
        # Find and hook target module
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer '{layer_name}' not found in visual_model")
        
        handle = target_module.register_forward_hook(hook_fn)
        
        # Forward pass through images
        self.model.eval()
        with torch.no_grad():
            # TODO (angela-sha): add error handling for greyscale images
            for img in images:
                img_tensor = self.classifier.image_transform(img).unsqueeze(0).to(self.device)
                _ = self.model(img_tensor)
        handle.remove()
        
        return activations

    def get_target_module(self, layer_name: str, param_name):
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
        return param 

    def unified_concept_editing(
        self,
        source_concepts: List[Image.Image], 
        target_concepts: List[Image.Image], 
        preserve_concepts: List[Image.Image],
        layer_name: str,
        edit_scale=1.0, 
        preserve_scale=1.0, 
        lamb=0.5,
        debug=True, # to print debug
    ):
        param = self.get_target_module(layer_name, 'weight')

        # UCE closed-form solution
        W_original = param.data.detach().clone()
        mat1 = lamb * W_original
        mat2 = lamb * torch.eye(W_original.shape[1], device=self.device, dtype=W_original.dtype)

        # Get activations (instead of embeddings in UCE)
        c_edit = self.capture_activations(source_concepts, layer_name, capture_input=True)  # [B, D1, D2]
        if debug: print("✓ Captured input activations for source concepts")
        v_target = self.capture_activations(target_concepts, layer_name)  # [B, D1, D2]
        if debug: print("✓ Captured output activations for target concepts")
        # Iterate over activations
        for (c_i, v_star) in zip(c_edit, v_target):
            c_i, v_star = c_i.squeeze(0), v_star.squeeze(0)
            mat1 += edit_scale * (v_star.T @ c_i) # due to ordering of PyTorch variables
            mat2 += edit_scale * (c_i.T @ c_i)
        
        # Concepts to preserve
        c_preserve = self.capture_activations(preserve_concepts, layer_name, capture_input=True)
        if debug: print("✓ Captured input activations for preserve concepts")
        v_preserve = self.capture_activations(preserve_concepts, layer_name)
        if debug: print("✓ Captured output activations for preserve concepts")
        # Iterate over activations
        for (c_j, v_star) in zip(c_preserve, v_preserve):  
            c_j, v_star = c_j.squeeze(0), v_star.squeeze(0)
            mat1 += preserve_scale * (v_star.T @ c_j)
            mat2 += preserve_scale * (c_j.T @ c_j)
        
        # Update weight: W_new = mat1 @ mat2^{-1}
        if debug: print(f"Original weight matrix dimension: {W_original.shape}, updated mat1, mat2 dimension: {mat1.shape}, {mat2.shape}")
        W_new = mat1 @ torch.inverse(mat2.float()).to(W_original.dtype)
        weight_change = (W_new - W_original).norm().item()
        weight_norm = W_new.norm().item()
        
        param.data = W_new
        if debug: 
            print(f"\n✓ Edit applied IN PLACE to self.model")
            print(f"  Layer: {layer_name}")
            print(f"  Actual weight change: {weight_change:.6f}")
            print(f"  New weight norm: {weight_norm:.6f}")
            print(f"  Relative change: {(weight_change / weight_norm):.6f}")
            print(f"✓ Weight successfully edited for {layer_name}")
        
        # Sanity check
        if weight_change < 1e-8 and debug:
            print(f"  ⚠ WARNING: Weight change is very small! Edit may not have been applied.")
        elif debug:
            print(f"  ✓ Weight successfully modified in self.model")
    
    def evaluate_edit(
        self,
        test_source_images: List[Image.Image],
        test_target_images: List[Image.Image],
        test_labels: List[str] = None
    ) -> dict:
        """Evaluate zero-shot text classification against test labels.
        """
        print(f"\n=== Evaluating Edit ===")
        print(f"  Test labels: {test_labels}")
        
        results = {
            'source_predictions': [],
            'target_predictions': [],
            'source_confidences': [],
            'target_confidences': []
        }
        
        # Test source images
        print(f"\n  Testing {len(test_source_images)} source images...")
        for img in test_source_images:
            pred, probs = self.classifier.classify_zero_shot(img, test_labels)
            results['source_predictions'].append(pred)
            results['source_confidences'].append(probs.max().item())
        
        # Test target images
        print(f"  Testing {len(test_target_images)} target images...")
        for img in test_target_images:
            pred, probs = self.classifier.classify_zero_shot(img, test_labels)
            results['target_predictions'].append(pred)
            results['target_confidences'].append(probs.max().item())        
        return results
    
if __name__ == "__main__":
    print("="*70)
    print("UCE: Editing Visual Model - with example Fish → Dog")
    print("="*70)

    # Initialize dinotxt and editor 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample local DINOv2 paths, models
    dinov2_path_local = '/scratch/shayuxin/models/csc2503/dinov2' # local git clone of dinov2 repository
    dinov2_text_model = 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
    scratch_dir = '/scratch/shayuxin/models/csc2503/'
    torch.hub.set_dir(scratch_dir)

    dinotxt = DINOTextBackbone(dinov2_path_local, dinov2_text_model, device)
    editor = UCEVisualEditor(dinotxt)

    # Step 1: Show available layers
    print("\n1. Available layers in visual_model:")
    layers = editor.list_editable_layers()

    # Show attention and MLP layers from later blocks
    print("\n   Recommended layers to edit (blocks 9-11):")
    for layer in layers:
        if any(x in layer['name'] for x in ['blocks.9', 'blocks.10', 'blocks.11']):
            if any(x in layer['name'] for x in ['attn.proj', 'mlp.fc2']):
                print(f"     {layer['name']}: {layer['shape']}")

    # Step 2: Load your images
    print("\n2. Loading images...")
    print("   >>> fish_images = [Image.open('fish1.jpg'), ...]")
    print("   >>> dog_images = [Image.open('dog1.jpg'), ...]")
    print("   >>> context_images = [Image.open('img1.jpg'), ...]  # context images")

    fishes = glob.glob("/scratch/shayuxin/data/imagenette-subset/train/n01440764/*.JPEG")
    dogs = glob.glob("/scratch/shayuxin/data/imagenette-subset/train/n02102040/*.JPEG")
    context_images = glob.glob("/scratch/shayuxin/data/imagenette-subset/val/*/*.JPEG")
    fish_images = [Image.open(f) for f in fishes]
    dog_images = [Image.open(f) for f in dogs]
    context_images = [Image.open(f) for f in context_images][:300]

    # Step 3: Unified concept editing algorithm 
    print("\n3. Computing UCE updates...")
    target_layer = "backbone.model.blocks.9.mlp.fc2"
    print(f"   Target layer: {target_layer}")
    editor.unified_concept_editing(
        source_concepts = fish_images,
        target_concepts = dog_images,
        preserve_concepts = context_images,
        layer_name = target_layer, 
        preserve_scale = 1,
        edit_scale = 1.2,
        debug = True
    )

    # Step 6: Evaluate after edit
    print("\n6. Evaluating AFTER edit...")
    results_after = editor.evaluate_edit(
        test_source_images=fish_images,
        test_target_images=dog_images,
        test_labels = ["tench", "English springer"]
    )
    print(results_after)