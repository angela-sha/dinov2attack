import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from typing import List

class ConceptPoisoningDataset(Dataset):
    def __init__(self, img_size=224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size

        # Default preprocessing transformation to apply to a dataset
        self.image_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    """
    Function definitions to implement for children class.
    Should update function state so dataloader is loaded with
    only source, target, or preserve concepts.
    """
    def get_source_concepts(self) -> None:
        raise NotImplementedError
    
    def get_target_concepts(self) -> None: 
        raise NotImplementedError

    def get_preserve_concepts(self) -> None:
        raise NotImplementedError    
