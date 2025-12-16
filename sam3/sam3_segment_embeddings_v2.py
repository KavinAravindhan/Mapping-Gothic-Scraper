import os

# Environment Setup
HF_CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"
CUDA_DEVICE = "2"

os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# Imports
import torch
import numpy as np
from PIL import Image

# SAM3 Tracker for segmentation
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

# DINOv2 for embeddings
from transformers import (
    AutoImageProcessor,
    Dinov2Model
)

# CLIP for embeddings (COMMENTED OUT - using DINOv2 instead)
# from transformers import CLIPProcessor, CLIPModel

import json
from typing import List, Dict
import matplotlib.pyplot as plt

# Setup and Load Models
print("\nSetting up device and loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Load SAM3 Tracker for automatic mask generation
print("   Loading SAM3 Tracker model...")
sam_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
sam_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
print("   SAM3 Tracker model loaded successfully")

# Load DINOv2 for creating embeddings
print("   Loading DINOv2 model for embeddings...")
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
print("   DINOv2 model loaded successfully")

# CLIP VERSION (COMMENTED OUT)
# print("   Loading CLIP model for embeddings...")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# print("   CLIP model loaded successfully")


# Load and Prepare Image
print("\nLoading input image...")
image_path = "/home/kr3131/Mapping-Gothic-Scraper/maps_output/Beaumont-sur-Oise-Eglise-Saint-Leonor/images/1065_00003.jpg"
try:
    image = Image.open(image_path).convert("RGB")
    print(f"   Image loaded: {image.size[0]}x{image.size[1]} pixels")
except FileNotFoundError:
    print(f"   ERROR: Image not found at '{image_path}'")
    exit(1)

# Generate masks using DENSE grid sampling to capture everything
print("\nRunning SAM3 automatic mask generation...")
print("   Generating masks using dense grid sampling...")
all_masks = []
all_boxes = []
all_scores = []

width, height = image.size
# Use MUCH denser sampling for small images - adjust based on image size
if max(width, height) < 500:
    points_per_side = 64  # Very dense for small images
else:
    points_per_side = 32  # Standard density for larger images

x_points = np.linspace(0, width, points_per_side, endpoint=False)
y_points = np.linspace(0, height, points_per_side, endpoint=False)

print(f"   Sampling {points_per_side}x{points_per_side} = {points_per_side*points_per_side} points across image...")
print(f"   This will take a few minutes...")

for i, x in enumerate(x_points):
    for j, y in enumerate(y_points):
        if (i * len(y_points) + j) % 200 == 0:
            progress = ((i * len(y_points) + j) / (len(x_points) * len(y_points))) * 100
            print(f"   Progress: {progress:.1f}% ({len(all_masks)} unique masks found)")
        
        input_points = [[[[int(x), int(y)]]]]
        input_labels = [[[1]]]
        
        inputs = sam_processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = sam_model(**inputs, multimask_output=True)
        
        masks_tensor = sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        iou_scores = outputs.iou_scores[0].cpu().numpy()
        iou_scores = iou_scores.flatten()  # Ensure it's 1D
        best_idx = np.argmax(iou_scores)

        mask = masks_tensor[0, best_idx].numpy()
        score = float(iou_scores[best_idx])  # Now this will work
        
        # Lower threshold to capture more segments
        if score < 0.5:  # Changed from 0.7 to 0.5
            continue
        
        mask_bool = mask > 0.5
        if not mask_bool.any():
            continue
        
        coords = np.column_stack(np.where(mask_bool))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = [x_min, y_min, x_max, y_max]
        
        # Allow smaller segments (changed from 100 to 25)
        area = (x_max - x_min) * (y_max - y_min)
        if area < 25:
            continue
        
        # Less strict uniqueness check to capture more variations
        is_unique = True
        for existing_mask in all_masks:
            intersection = np.logical_and(mask_bool, existing_mask > 0.5).sum()
            union = np.logical_or(mask_bool, existing_mask > 0.5).sum()
            iou = intersection / (union + 1e-6)
            if iou > 0.85:  # Changed from 0.7 to 0.85 (less strict)
                is_unique = False
                break
        
        if is_unique:
            all_masks.append(mask)
            all_boxes.append(bbox)
            all_scores.append(score)

print(f"   Generated {len(all_masks)} unique masks")

masks = all_masks
boxes = all_boxes
scores = all_scores

print(f"   Total segments found: {len(masks)}")

# Extract Segments from Original Image
print(f"\nExtracting {len(masks)} segments from original image...")
segments = []
segment_info = []

for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.size[0], x_max)
    y_max = min(image.size[1], y_max)
    
    # Skip invalid boxes
    if x_max <= x_min or y_max <= y_min:
        continue
    
    # Crop the region
    segment_img = image.crop((x_min, y_min, x_max, y_max))
    
    # Apply mask to segment (create masked version)
    mask_crop = mask[y_min:y_max, x_min:x_max]
    segment_array = np.array(segment_img)
    
    if mask_crop.shape[:2] == segment_array.shape[:2]:
        mask_3channel = np.stack([mask_crop] * 3, axis=-1)
        masked_segment = (segment_array * mask_3channel).astype(np.uint8)
        segment_img_masked = Image.fromarray(masked_segment)
    else:
        # If mask doesn't match, use unmasked segment
        segment_img_masked = segment_img
    
    segments.append(segment_img_masked)
    segment_info.append({
        'id': idx,
        'bbox': [x_min, y_min, x_max, y_max],
        'score': float(score),
        'size': segment_img.size
    })
    
    if (idx + 1) % 50 == 0:
        print(f"   Extracted {idx + 1}/{len(masks)} segments")

print(f"   All segments extracted")

# Create Embeddings for Each Segment
print(f"\nCreating embeddings for {len(segments)} segments...")
all_embeddings = []

for idx, segment in enumerate(segments):
    # DINOv2 VERSION
    inputs = dinov2_processor(images=segment, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
        # Get the [CLS] token embedding (global image representation)
        embedding = outputs.last_hidden_state[:, 0]  # Shape: [1, 768] for base model
        # Normalize the embeddings
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()[0]
    
    # CLIP VERSION (COMMENTED OUT)
    # inputs = clip_processor(images=segment, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     image_features = clip_model.get_image_features(**inputs)
    #     embedding = image_features / image_features.norm(dim=-1, keepdim=True)
    #     embedding = embedding.cpu().numpy()[0]
    
    all_embeddings.append(embedding.tolist())
    
    if (idx + 1) % 50 == 0:
        print(f"   Created embeddings for {idx + 1}/{len(segments)} segments")

print(f"   All embeddings created (dimension: {len(all_embeddings[0])})")

# Save and Display Results
print("\nSaving results...")

# Prepare output data
output_data = {
    'num_segments': len(segments),
    'embedding_dimension': len(all_embeddings[0]),
    'embedding_model': 'facebook/dinov2-base',  # Document which model was used
    'segments': []
}

for idx, (embedding, info) in enumerate(zip(all_embeddings, segment_info)):
    output_data['segments'].append({
        'segment_id': idx,
        'bbox': info['bbox'],
        'confidence_score': info['score'],
        'segment_size': info['size'],
        'embedding': embedding
    })

# Save to JSON file
output_file = "segment_embeddings.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"   Saved to '{output_file}'")

# Save embeddings as numpy array
embeddings_array = np.array(all_embeddings)
np.save("segment_embeddings.npy", embeddings_array)
print(f"   Saved embeddings array to 'segment_embeddings.npy'")

# Save segment info separately (without embeddings for easier inspection)
segment_info_file = "segment_info.json"
segment_info_data = {
    'num_segments': len(segments),
    'segments': [
        {
            'segment_id': info['id'],
            'bbox': info['bbox'],
            'confidence_score': info['score'],
            'segment_size': info['size']
        }
        for info in segment_info
    ]
}
with open(segment_info_file, 'w') as f:
    json.dump(segment_info_data, f, indent=2)
print(f"   Saved segment info to '{segment_info_file}'")

# Display Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total segments found: {len(segments)}")
print(f"Embedding dimension: {len(all_embeddings[0])}")
print(f"Embedding model: facebook/dinov2-base")
print(f"\nFirst 3 segments:")
for i in range(min(3, len(output_data['segments']))):
    seg = output_data['segments'][i]
    print(f"\nSegment {i}:")
    print(f"  - Bounding box: {seg['bbox']}")
    print(f"  - Size: {seg['segment_size']}")
    print(f"  - Confidence: {seg['confidence_score']:.3f}")
    print(f"  - Embedding (first 10 values): {seg['embedding'][:10]}")

# Optional: Visualize some segments
print("\n" + "=" * 60)
print("Creating visualization...")
num_to_visualize = min(12, len(segments))
rows = 3
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
axes = axes.flatten()

for i in range(num_to_visualize):
    axes[i].imshow(segments[i])
    axes[i].set_title(f"Segment {i}\nScore: {scores[i]:.3f}", fontsize=10)
    axes[i].axis('off')

# Hide unused subplots
for i in range(num_to_visualize, rows * cols):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("segment_visualization.png", dpi=150, bbox_inches='tight')
print("   Saved visualization to 'segment_visualization.png'")

print("\n" + "=" * 60)
print("✓ COMPLETE! Files saved:")
print(f"  - {output_file} (full data with embeddings)")
print(f"  - segment_embeddings.npy (embeddings only)")
print(f"  - {segment_info_file} (segment info without embeddings)")
print("  - segment_visualization.png (visual preview)")
print("=" * 60)

print("\nDone! 🎉")