import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

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
testDS = SegmentationDataset(root_dirs, split='Test', transforms=transform)
test_loader = DataLoader(testDS, batch_size=1, shuffle=False)

def load_model(path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
    checkpoint = torch.load(path)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)
    return model

# Measure inference time
def measure_inference_time(model, dataloader, num_warmup=5, num_runs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    durations = []
    memory_stats = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            for batch in dataloader:
                input_tensor, _ = batch
                input_tensor = input_tensor.to(device)
                torch.cuda.reset_peak_memory_stats()
                
                start_event.record()
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)
                end_event.record()
                
                torch.cuda.synchronize()
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                memory_stats.append(peak_memory)

                duration = start_event.elapsed_time(end_event)
                durations.append(duration)
    
    average_memory = sum(memory_stats) / len(memory_stats)
    avg_duration = sum(durations) / len(durations)
    
    return average_memory, avg_duration

# Evaluate model performance to get IoU distribution per class
def evaluate_model_performance(model, dataloader, num_classes):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    iou_scores_per_class = {cls: [] for cls in range(num_classes)}
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            
            for cls in range(num_classes):
                pred_inds = (pred == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()
                if union != 0:
                    iou_scores_per_class[cls].append(intersection / union)
    
    iou_distribution = {cls: np.mean(iou_scores_per_class[cls]) if iou_scores_per_class[cls] else 0.0 
                        for cls in range(num_classes)}
    return iou_distribution

# Function to get inference metrics and IoU distribution
def get_inference_metrics(model_path, dataloader, batch_sizes, num_classes=4):
    model = load_model(model_path)
    
    # Evaluate IoU distribution per class
    iou_distribution = evaluate_model_performance(model, dataloader, num_classes)
    print(f"IoU distribution per class: {iou_distribution}")
    
    # Measure inference time and memory for different batch sizes
    for batch_size in batch_sizes:
        dataloader = DataLoader(testDS, batch_size=batch_size, shuffle=False)
        avg_memory, avg_duration = measure_inference_time(model, dataloader)
        print(f"batch size = {batch_size}")
        print(f"peak memory = {avg_memory:.2f} MB")
        print(f"average inference time = {avg_duration:.6f} ms")

# Example usage
if __name__ == "__main__":
    model_path = "models/model_epoch_120.pth"
    batch_sizes = [1, 2, 4, 8, 16]
    get_inference_metrics(model_path, test_loader, batch_sizes)
