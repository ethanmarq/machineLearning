# Imports
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import segmentation_models_pytorch as smp
import time  # For measuring inference speed

# Custom Dataset
INPUT_IMAGE_HEIGHT = 1024
INPUT_IMAGE_WIDTH = 672

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
            target = t(target)
        target = torch.tensor(np.array(target), dtype=torch.int64)
        image = transforms.ToTensor()(image)
        return image, target

class SegmentationDataset(Dataset):
    def __init__(self, root_dirs, split='Train', transforms=None):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.split = split
        self.transforms = transforms
        self.image_mask_pairs = self._collect_image_mask_pairs()
        print(f"Matched {len(self.image_mask_pairs)} image-mask pairs.")

    def _collect_image_mask_pairs(self):
        image_mask_pairs = []
        for root_dir in self.root_dirs:
            image_dir = os.path.join(root_dir, self.split, 'imgs')
            mask_dir = os.path.join(root_dir, self.split, 'annos', 'int_maps')

            images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

            mask_dict = {os.path.basename(mask).split('_')[1].replace('.png', ''): mask for mask in masks}
            for img in images:
                key = os.path.basename(img).split('_')[1].replace('.png', '')
                if key in mask_dict:
                    image_mask_pairs.append((img, mask_dict[key]))
                else:
                    print(f"No matching mask for image: {img}")

        return image_mask_pairs

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return image, mask

# Transformation definition
transform = Compose([transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST)])

# List of root directories
root_dirs = ['CAT/mixed', 'CAT/Brown_Field', 'CAT/Main_Trail', 'CAT/Power_Line']

# Creating dataset and dataloader instances
trainDS = SegmentationDataset(root_dirs, split='Train', transforms=transform)
testDS = SegmentationDataset(root_dirs, split='Test', transforms=transform)

train_loader = DataLoader(trainDS, batch_size=4, shuffle=True)
test_loader = DataLoader(testDS, batch_size=4, shuffle=False)

# Debugging: Load a batch of data
for images, masks in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of masks shape: {masks.shape}")
    break

# Instantiate the model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=4).cuda()
# print(model)

# Trainer

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_CLASSES = 4

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_classes, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.device = device
        self.train_iou_per_class = []
        self.val_miou_per_epoch = []

    def calculate_iou(self, pred, target):
        iou_per_class = []
        pred = torch.argmax(pred, dim=1)
        
        for cls in range(self.num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            
            if union == 0:
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append(intersection / union)
        
        return iou_per_class

    def mean_iou(self, iou_list):
        valid_iou = [iou for iou in iou_list if not np.isnan(iou)]
        if len(valid_iou) == 0:
            return float('nan')
        return np.mean(valid_iou)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        epoch_iou_scores = np.zeros((len(self.train_loader), self.num_classes))

        for batch_idx, (images, masks) in enumerate(tqdm(self.train_loader, desc=f'Training Epoch {epoch}')):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate IoU
            iou = self.calculate_iou(outputs, masks)
            epoch_iou_scores[batch_idx] = iou
        
        avg_loss = running_loss / len(self.train_loader)
        avg_iou_scores = np.nanmean(epoch_iou_scores, axis=0)
        self.train_iou_per_class.append(avg_iou_scores)

        print(f"Epoch [{epoch}] Training Loss: {avg_loss}")
        print(f"Epoch [{epoch}] IoU per class: {avg_iou_scores}")
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        iou_scores = []

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                
                # Calculate IoU
                iou = self.calculate_iou(outputs, masks)
                iou_scores.append(iou)
        
        avg_loss = running_loss / len(self.val_loader)
        
        # Compute mean IoU
        iou_scores = np.array(iou_scores)
        miou_per_class = np.nanmean(iou_scores, axis=0)
        miou = self.mean_iou(miou_per_class)
        
        print(f"Mean IoU: {miou}")  # Only print the mean IoU
        return avg_loss, miou, miou_per_class

    def train(self, num_epochs, save_interval=10, save_path='models'):
        os.makedirs(save_path, exist_ok=True)
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch)
            val_loss, val_miou, _ = self.validate_epoch()
            self.val_miou_per_epoch.append(val_miou)  # Store MIoU from validation

            # Save the model every save_interval epochs
            if epoch % save_interval == 0:
                model_save_path = os.path.join(save_path, f'model_epoch_{epoch}.pth')
                torch.save(self.model.state_dict(), model_save_path)
                print(f'Model saved at epoch {epoch} to {model_save_path}')

# Instantiate the Trainer class with weighted loss
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_classes=NUM_CLASSES,
    device='cuda'
)

def print_class_distribution(model, loader, num_classes):
    model.eval()  # Set the model to evaluation mode
    class_counts = [0] * num_classes
    
    with torch.no_grad():  # Disable gradient computation
        for images, masks in loader:
            images = images.to('cuda')
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for pred in preds:
                for cls in range(num_classes):
                    class_counts[cls] += (pred == cls).sum().item()
    
    total_pixels = sum(class_counts)
    class_distribution = [count / total_pixels for count in class_counts]
    
    print(f"Class distribution: {class_distribution}")

# Call the function with the model and loaders
print_class_distribution(model, train_loader, NUM_CLASSES)
print_class_distribution(model, test_loader, NUM_CLASSES)

import numpy as np
import matplotlib.pyplot as plt

def plot_miou(trainer, num_epochs, save_path='miou.svg'):
    epochs = np.arange(1, num_epochs + 1)
    miou_per_epoch = trainer.val_miou_per_epoch  # Use validation MIoU

    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, miou_per_epoch, label='MIoU')
    
    plt.xlabel('Epochs')
    plt.ylabel('MIoU')
    plt.title('UNet with ResNet Encoder')
    plt.legend()
    plt.grid(True)
    
    # Save the figure as an SVG file
    plt.savefig(save_path, format='svg')
    plt.show()

def measure_inference_speed(model, loader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    total_time = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            
            start_time = time.time()
            _ = model(images)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            num_batches += 1
    
    avg_inference_time = total_time / num_batches
    print(f'Average inference time per batch: {avg_inference_time:.6f} seconds')

if __name__ == "__main__":
    # Train the model
    num_epochs = 120
    trainer.train(num_epochs)
    plot_miou(trainer, num_epochs)

    # Measure inference speed
    measure_inference_speed(model, test_loader)
