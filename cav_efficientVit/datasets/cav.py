"""
Dataset file for the CaV Off Road Dataset. Used to preprocess the data and 
prepare it for the data loader. The CaV dataset can be found here:
https://www.cavs.msstate.edu/resources/autonomous_dataset.php
"""


import os
import pathlib
import numpy as np
from PIL import Image
import glob
import cv2
import math
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse
import torchvision.transforms.functional as F
from torch.nn.functional import pad

from efficientvit.models.utils import resize



INPUT_IMAGE_HEIGHT = 1024
INPUT_IMAGE_WIDTH = 672 # 644 Original

class Compose:
    def __init__(self, transforms):
        # Transformations Object
        self.transforms = transforms
        

    def __call__(self, image, target):
        # Applies Transformations to all the images
        for t in self.transforms:
            image = t(image)
            target = t(target)
        # Converts to NpArray and then to Tensor
        image = transforms.ToTensor()(image)
        target = torch.tensor(np.array(target), dtype=torch.int64)
        return image, target

class SegmentationDataset(Dataset):
    # Classes
    classes = ( 'Background', 'Sedan', 'Pickup', 'Off-Road' )
               
    def __init__(self, root_dirs, split='Train', transforms=None):
        
            
        # Defines Objects & calls Image Masking Function
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.split = split
        self.transforms = transforms
        self.image_mask_pairs = self._collect_image_mask_pairs()
        print(f"Matched {len(self.image_mask_pairs)} image-mask pairs.")

    def _collect_image_mask_pairs(self):
        
        image_mask_pairs = []
        
        # Sorts through Directories 
        for root_dir in self.root_dirs:
            image_dir = os.path.join(root_dir, self.split, 'imgs')
            mask_dir = os.path.join(root_dir, self.split, 'annos', 'int_maps')
            
            # Find images ending in .png
            images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
            
            # Create Dictionary to map mask identifiers
            mask_dict = {os.path.basename(mask).split('_')[1].replace('.png', ''): mask for mask in masks}
            for img in images:
                key = os.path.basename(img).split('_')[1].replace('.png', '')
                
                # Appending matched image and mask
                if key in mask_dict:
                    image_mask_pairs.append((img, mask_dict[key]))
                else:
                    # Useful to see if you have missing mask/img pairs
                    print(f"No matching mask for image: {img}")

        return image_mask_pairs
    
    # Return number of matching img/mask pairs
    def __len__(self):
        return len(self.image_mask_pairs)

        
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transforms:
            image, mask = self.transforms(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.tensor(np.array(mask), dtype=torch.int64)
           
        
            
        return {
            "data" : image,
            "label" : mask,
        }
    


# List of root directories
root_dirs = ['CAT/mixed', 'CAT/Brown_Field', 'CAT/Main_Trail', 'CAT/Power_Line']


"""
# Previous Usage of Compose class
transform = Compose([transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST)])

trainDS = SegmentationDataset(root_dirs, split='Train', transforms=transform)
testDS = SegmentationDataset(root_dirs, split='Test', transforms=transform)

train_loader = DataLoader(trainDS, batch_size=4, shuffle=True)
test_loader = DataLoader(testDS, batch_size=4, shuffle=False)



RELLIS CLASSES OBJECT
  classes = (
    'void', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
    'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
    'barrier', 'puddle', 'mud', 'rubble'
   ) 

"""