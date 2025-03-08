import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Load Flickr8k dataset annotations
captions_file = "Flickr8k/captions.txt"

if not os.path.exists(captions_file):
    raise FileNotFoundError(f"Error: {captions_file} not found! Check the dataset path.")

with open(captions_file, "r", encoding="utf-8") as f:
    captions = f.readlines()

# Create image-caption mapping
image_captions = {}
for line in captions:
    parts = line.strip().split(",",1)

    # Ensure the line is correctly formatted (should have at least 2 parts)
    if len(parts) < 2:
        print(f"Skipping malformed line: {line.strip()}")  # Debugging output
        continue

    image_name = parts[0].split("#")[0]  # Remove caption index (e.g., 1000268201_693b08cb0e.jpg#0)
    caption = parts[1]

    if image_name not in image_captions:
        image_captions[image_name] = []

    image_captions[image_name].append(caption)

print(f"✅ Successfully loaded {len(image_captions)} images with captions!")

# Image Transformations (Resize, Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_dict, transform=None):
        self.image_dir = image_dir
        self.captions_dict = captions_dict
        self.image_names = list(captions_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            return None, None

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = self.captions_dict[image_name][0]  # Use the first caption
        return image, caption

# Initialize Dataset
dataset = FlickrDataset(image_dir="Flickr8k_images/", captions_dict=image_captions, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"✅ Dataset successfully loaded with {len(dataset)} images!")

print(f"Total images with captions: {len(image_captions)}")

# Print first 5 image-caption pairs
for i, (image, captions) in enumerate(image_captions.items()):
    print(f"Image: {image} | Captions: {captions}")
    if i == 4:  # Show only first 5 entries
        break

