import torch
import clip
from PIL import Image
import os
import pickle

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define Image Directory
image_dir = "/content/flickr8k/Images/"  # Change if necessary

# Load Captions
captions_file = "/content/flickr8k/captions.txt"
image_captions = {}

with open(captions_file, "r") as f:
    for line in f.readlines():
        parts = line.strip().split(",",1)
        image_name = parts[0].split("#")[0]  # Extract image filename
        caption = parts[1]
        if image_name not in image_captions:
            image_captions[image_name] = []
        image_captions[image_name].append(caption)

# Extract and Save Image & Text Features
image_features = {}
text_features = {}

with torch.no_grad():
    for image_name, captions in image_captions.items():
        image_path = os.path.join(image_dir, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue
        
        # Process Image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_embedding = model.encode_image(image)
        image_features[image_name] = image_embedding.cpu().numpy()
        
        # Process Captions
        caption_embeddings = []
        for caption in captions:
            text = clip.tokenize([caption]).to(device)
            text_embedding = model.encode_text(text)
            caption_embeddings.append(text_embedding.cpu().numpy())
        
        text_features[image_name] = caption_embeddings

# Save the extracted features
with open("image_features.pkl", "wb") as f:
    pickle.dump(image_features, f)

with open("text_features.pkl", "wb") as f:
    pickle.dump(text_features, f)

print("✅ Feature extraction complete! Saved image & text embeddings.")
