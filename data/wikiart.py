import torch
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from dataset import ConceptPoisoningDataset
import random
import ast 

LOCAL_CSV_PATH = "/scratch/shayuxin/data/wikiart/classes.csv"
LOCAL_TEST_PATH = "/scratch/shayuxin/data/wikiart/"

class WikiArtDataset(ConceptPoisoningDataset):
    """
    WikiArt dataset loader using PIL for image processing.
    """
    def __init__(
            self, 
            csv_file: str, # metadata path
            img_dir: str, # Wikiart images 
            categorize_by: str, # one of ['artist', 'genre']
            test=True, 
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.categorize_by = categorize_by

        self.data = self.data[self.data['genre_count'] == 1]
        if test:
            self.data = self.data[self.data['subset'] == 'test']
        
        # Create artist to index mapping
        self.artists = sorted(self.data['artist'].unique())
        self.artist_to_label = {artist: idx for idx, artist in enumerate(self.artists)}
        self.artist_to_idx = {artist: self.data[self.data['artist'] == artist].index.tolist()
                              for artist in self.artists}

        # Create genres to index mapping
        self.genres = sorted(self.data['genre'].dropna().apply(lambda x: ast.literal_eval(x)[0]).unique())
        self.genre_to_label = {genre: idx for idx, genre in enumerate(self.genres)}
        self.genre_to_idx = {genre: self.data[\
            self.data['genre'].apply(lambda genres: genre in genres)].index.tolist()\
            for genre in self.genres}        
        # Initialize valid indices to all images
        self.cur_indices = self.data.index.tolist()

    def get_label_idxs(self, name):
        if self.categorize_by == "artist":
            return self.artist_to_idx[name]
        if self.categorize_by == "genre":
            return self.genre_to_idx[name]
        else:
            print(f"The desired label category does not exist for {self.categorize_by}")

    def __len__(self):
        return len(self.cur_indices)
    
    def __getitem__(self, idx):
        row = self.data.loc[self.cur_indices[idx]]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.image_transform(image)
            label = self.artist_to_label[row['artist']]
        
            return image, label

        except FileNotFoundError as e:
            print(f"The specified file {img_path} was not found.")
        
    # Update class state to only use subset of data for indexing
    def get_source_concepts(self, source_concept, sample_size=None):
        source_set = self.get_label_idxs(source_concept)
        if sample_size: 
            source_set = random.sample(source_set, sample_size)
        self.cur_indices = source_set


    def get_target_concepts(self, target_concept, sample_size=None):
        target_set = self.get_label_idxs(target_concept)
        if sample_size: 
            target_set = random.sample(target_set, sample_size)
        self.cur_indices = target_set

    def get_preserve_concepts(self, source_concept, sample_size=None):
        has_source_label = self.get_label_idxs(source_concept)
        preserve_set = self.data.index[~self.data.index.isin(has_source_label)].tolist()
        if sample_size: 
            preserve_set = random.sample(preserve_set, sample_size)
        self.cur_indices = preserve_set

if __name__ == "__main__":
    # Create dataset and loader
    dataset = WikiArtDataset(
        csv_file=LOCAL_CSV_PATH,
        img_dir=LOCAL_TEST_PATH,
        categorize_by='artist'
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of artists: {len(dataset.artists)}")
    print(f"Number of genres: {len(dataset.genres)}")

    # Test loading a batch
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test getting concept label data loader 
    dataset.get_target_concepts("vincent van gogh", sample_size=10)
    vg_label = dataset.artist_to_label["vincent van gogh"]
    print(f"Loading target concept for Vincent van Gogh with label {vg_label}...")
    print(f"Target indices: {dataset.cur_indices}")
    _, labels = next(iter(loader))
    print(f"Labels for batch size 4: {labels}")
    # Test getting preserved labels data loader 
    dataset.get_preserve_concepts("vincent van gogh", sample_size=10)
    print(f"Loading images to preserve concepts for without {vg_label}...")
    print(f"Preserve indices: {dataset.cur_indices}")
    _, labels = next(iter(loader))
    print(f"Labels for batch size 4: {labels}")

    