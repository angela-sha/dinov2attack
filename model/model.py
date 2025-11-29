import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from typing import List, Tuple

class DINOTextBackbone:
    def __init__(self, dinov2_path_local, dinov2_text_model, device,
                 text_model_name: str = "openai/clip-vit-large-patch14"):
        self.device = device
        print(f"Loading DINOv2 with text encoder: {dinov2_text_model}")
        self.dinov2 = torch.hub.load(dinov2_path_local, dinov2_text_model, source='local')
        self.dinov2.to(device)
        print("✓ DINOv2 with text encoder loaded successfully")
        self.dinov2.eval()
        
        print(f"Loading text tokenizer: {text_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        print("✓ CLIP text encoder loaded successfully")
        
        # Image preprocessing (DINOv2 standard)
        self.image_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.input_size = 224
        print("✓ DINOText Classifier initialized")

    def encode_text(self, texts) -> torch.Tensor:
        # Tokenize text and passes through DINOv2 text tower        
        # Tokenize
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        ).to(self.device)
        # Project through DINOv2 text alignment module
        with torch.no_grad():
            text_embedding = self.dinov2.text_model(inputs["input_ids"]) 
        return text_embedding

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        # Extract image embedding through DINOv2 visual model
        img_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_embedding = self.dinov2.visual_model(img_tensor)
        return img_embedding

    def classify_zero_shot(
        self, image: Image.Image,
        class_labels: List[str], temperature: float = 0.01
        ) -> Tuple[str, torch.Tensor]:
            """
            Implementation of zero-shot classification using image-text similarity.
            
            :param Image.Image image: Input image for classification
            :param List[str] class_labels: List of candidate class names
            :param float temperature: Temperature for softmax (lower = sharper)
            
            :return: (predicted_class, probabilities)
            :rtype: Tuple[str, torch.Tensor]
            """
            # Extract features
            img_features = self.encode_image(image)
            # Create text prompts (following CLIP convention)
            text_prompts = [f"a photo of a {label}" for label in class_labels]
            print(text_prompts)
            text_features = self.encode_text(text_prompts)
            print(text_features.shape)
            
            # Normalize features
            img_features = F.normalize(img_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            logits = (img_features @ text_features.T) / temperature
            probs = F.softmax(logits, dim=-1)[0]
            
            # Get prediction
            pred_idx = probs.argmax().item()
            predicted_class = class_labels[pred_idx]
            
            return predicted_class, probs
            
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample local DINOv2 paths, models
    dinov2_path_local = '/scratch/shayuxin/models/csc2503/dinov2' # local git clone of dinov2 repository
    dinov2_text_model = 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
    scratch_dir = '/scratch/shayuxin/models/csc2503/'
    torch.hub.set_dir(scratch_dir)

    dinotxt = DINOTextBackbone(dinov2_path_local, dinov2_text_model, device)

    test_img = "/scratch/shayuxin/data/imagenette-subset/train/n01440764/n01440764_9966.JPEG"
    img = Image.open(test_img)
    print("Initializing zero-shot classification with DINOv2 text model...")
    print(dinotxt.classify_zero_shot(img, ["fish", "ocean", "flower"]))