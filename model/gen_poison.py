import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from typing import List, Tuple

class PoisonGeneration:
    def __init__(self, dinov2_path_local, device, eps=0.2):
        self.eps = eps
        self.device = device
        self.dinov2 = torch.hub.load(dinov2_path_local, 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l', source='local')
        self.dinov2.to(device)
        print("✓ DINOv2 loaded successfully")
        self.dinov2.eval()
        
        # Image preprocessing (DINOv2 standard)
        self.image_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def transform(self, image: Image.Image):
        return self.image_transform(image).unsqueeze(0).to(self.device)

    def get_latent(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_embedding = self.dinov2.visual_model(img_tensor)
        return img_embedding

    def set_target(self, target_image: Image.Image):
        self.target_image = target_image

    def generate_one(self, pil_image: Image.Image):

        resized_pil_image = self.transform(pil_image)
        source_tensor = resized_pil_image

        target_image = self.target_image
        target_tensor = self.transform(target_image)

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        modifier = torch.clone(source_tensor) * 0.0

        t_size = 2000
        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change

        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
            modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)

            loss = (adv_latent - target_latent).norm()

            tot_loss = loss.sum()
            grad = torch.autograd.grad(tot_loss, modifier)[0]

            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()
    
            if i % 50 == 0:
                print("# Iter: {}\tLoss: {:.3f}".format(i, loss.mean().item()))

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        print(final_adv_batch.shape)
        final_img = tensor2img(final_adv_batch.squeeze())
        return final_img

def tensor2img(img_tensor):
    return T.ToPILImage()(img_tensor)
            
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample local DINOv2 paths, models
    dinov2_path_local = '/root/junk/dinov2' # local git clone of dinov2 repository
    scratch_dir = '/root/junk'
    torch.hub.set_dir(scratch_dir)

    poisoner = PoisonGeneration(dinov2_path_local, device)

    source_path = "/root/junk/vangogh.jpg"
    target_path = "/root/junk/monet.jpg" 
    source_img = Image.open(source_path)
    target_img = Image.open(target_path)

    poisoner.set_target(target_img)
    poisoned = poisoner.generate_one(source_img)
    poisoned.save("poisoned.jpg")
