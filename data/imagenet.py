import torch
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from dataset import ConceptPoisoningDataset
import random
import ast 

LOCAL_CLASS_NAMES = "/project/aip-papernot/shayuxin/csc2503/vit-attack/map_clsloc.txt"
LOCAL_TEST_PATH = "/datasets/imagenet/val/"

class ImageNetDataset(ConceptPoisoningDataset):
    """
    ImageNet validation dataset loader using PIL for image processing.
    """
    def __init__(
            self, 
            # cls_file: str, # metadata path
            img_dir: str, # ImageNet images 
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.img_dir = img_dir
        
        # Process class folders and indices
        classes = sorted([d for d in os.listdir(self.img_dir) 
                         if os.path.isdir(os.path.join(self.img_dir, d))])
        self.class_to_label = {cls_name: idx for idx, cls_name in enumerate(classes)}

        self.samples = []
        self.class_to_idxs = {c: [] for c in classes}        

        counter = 0
        for class_name in classes:
            class_dir = os.path.join(self.img_dir, class_name)
            class_idx = self.class_to_label[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
                    self.class_to_idxs[class_name].append(counter)
                    counter += 1

        assert counter == len(self.samples) # no samples dropped
        
        # Initialize valid indices to all images
        self.cur_indices = range(len(self.samples))

    def __len__(self):
        return len(self.cur_indices)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[self.cur_indices[idx]]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.image_transform(image)
            return image, label
        except FileNotFoundError as e:
            print(f"The specified file {img_path} was not found.")
        
    # Update class state to only use subset of data for indexing
    def get_source_concepts(self, source_concept, sample_size=None):
        source_set = self.class_to_idxs[source_concept]
        if sample_size: 
            source_set = random.sample(source_set, sample_size)
        self.cur_indices = source_set


    def get_target_concepts(self, target_concept, sample_size=None):
        target_set = self.class_to_idxs[target_concept]
        if sample_size: 
            target_set = random.sample(target_set, sample_size)
        self.cur_indices = target_set

    def get_preserve_concepts(self, source_concept, sample_size=None):
        has_source_label = self.class_to_idxs[source_concept]
        preserve_set = [i for i in range(len(self.samples)) if i not in has_source_label]
        if sample_size: 
            preserve_set = random.sample(preserve_set, sample_size)
        self.cur_indices = preserve_set

if __name__ == "__main__":
    # Create dataset and loader
    dataset = ImageNetDataset(
        img_dir=LOCAL_TEST_PATH,
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_to_label)}")
    print(f"Number of images: {len(dataset.samples)}")

    # Test loading a batch
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test getting concept label data loader 
    dataset.get_target_concepts("n03255030", sample_size=10)
    label = dataset.class_to_label["n03255030"]
    print(f"Loading target concept for dumbbell (n03255030) with label {label}...")
    print(f"Target indices: {dataset.cur_indices}")
    _, labels = next(iter(loader))
    print(f"Labels for batch size 4: {labels}")
    # Test getting preserved labels data loader 
    dataset.get_preserve_concepts("n03255030", sample_size=10)
    print(f"Loading images to preserve concepts for dumbbell (n03255030) without {label}...")
    print(f"Preserve indices: {dataset.cur_indices}")
    _, labels = next(iter(loader))
    print(f"Labels for batch size 4: {labels}")

    