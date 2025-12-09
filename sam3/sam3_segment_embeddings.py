import os

# Environment Setup
HF_CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"
CUDA_DEVICE = "6"

os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# Imports
import torch
import numpy as np
from PIL import Image
from transformers import (
    Sam3TrackerProcessor, 
    Sam3TrackerModel,
    CLIPProcessor, 
    CLIPModel
)
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

# Load SAM3 for segmentation
print("   Loading SAM3 model")
sam_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
sam_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
print("   SAM3 model loaded successfully")

# Load CLIP for creating embeddings
print("   Loading CLIP model for embeddings...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("   CLIP model loaded successfully")


# Load and Prepare Image
print("\nLoading input image...")
# Replace with your image path
image_path = "your_image.jpg"  # CHANGE THIS TO YOUR IMAGE PATH
try:
    image = Image.open(image_path).convert("RGB")
    print(f"   Image loaded: {image.size[0]}x{image.size[1]} pixels")
except FileNotFoundError:
    print(f"   ERROR: Image not found at '{image_path}'")
    exit(1)

# Generate Masks with SAM3 (Automatic Mask Generation)
print("\nRunning SAM3 automatic mask generation...")

# We'll use a grid of points to generate masks automatically
def generate_all_masks(image, model, processor, points_per_side=32):
    """
    Generate masks by sampling points across the image in a grid pattern
    """
    width, height = image.size
    all_masks = []
    all_boxes = []
    all_scores = []
    
    # Create a grid of points
    x_points = np.linspace(0, width, points_per_side)
    y_points = np.linspace(0, height, points_per_side)
    
    print(f"   Sampling {points_per_side}x{points_per_side} points across image...")
    
    processed_points = set()
    
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            # Show progress
            if (i * len(y_points) + j) % 100 == 0:
                progress = ((i * len(y_points) + j) / (len(x_points) * len(y_points))) * 100
                print(f"   Progress: {progress:.1f}% ({len(all_masks)} masks found so far)")
            
            point = [[[[int(x), int(y)]]]]
            label = [[[1]]]  # Positive click
            
            # Process with SAM3
            inputs = processor(
                images=image, 
                input_points=point, 
                input_labels=label, 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=True)
            
            # Get the best mask (highest score)
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"]
            )[0]
            
            # Get IoU scores for each mask
            iou_scores = outputs.iou_scores[0].cpu().numpy()
            best_mask_idx = np.argmax(iou_scores)
            
            mask = masks[0, best_mask_idx].numpy()  # Get best mask
            score = iou_scores[best_mask_idx]
            
            # Skip low-quality masks
            if score < 0.7:
                continue
            
            # Get bounding box from mask
            mask_bool = mask > 0.5
            if not mask_bool.any():
                continue
                
            coords = np.column_stack(np.where(mask_bool))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = [x_min, y_min, x_max, y_max]
            
            # Check if this mask is unique (not too similar to existing masks)
            is_unique = True
            for existing_mask in all_masks:
                # Calculate IoU with existing masks
                intersection = np.logical_and(mask_bool, existing_mask > 0.5).sum()
                union = np.logical_or(mask_bool, existing_mask > 0.5).sum()
                iou = intersection / (union + 1e-6)
                if iou > 0.7:  # Too similar to existing mask
                    is_unique = False
                    break
            
            if is_unique:
                all_masks.append(mask)
                all_boxes.append(bbox)
                all_scores.append(score)
    
    print(f"   Generated {len(all_masks)} unique masks")
    return all_masks, all_boxes, all_scores

masks, boxes, scores = generate_all_masks(
    image, sam_model, sam_processor, points_per_side=12
)

print(f"   Total segments found: {len(masks)}")

# Extract Segments from Original Image
print(f"\nExtracting {len(masks)} segments from original image...")
segments = []
segment_info = []

for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Crop the region
    segment_img = image.crop((x_min, y_min, x_max, y_max))
    
    # Apply mask to segment (optional, for cleaner segments)
    mask_crop = mask[y_min:y_max, x_min:x_max]
    segment_array = np.array(segment_img)
    mask_3channel = np.stack([mask_crop] * 3, axis=-1)
    masked_segment = (segment_array * mask_3channel).astype(np.uint8)
    segment_img_masked = Image.fromarray(masked_segment)
    
    segments.append(segment_img_masked)
    segment_info.append({
        'id': idx,
        'bbox': box,
        'score': float(score),
        'size': segment_img.size
    })
    
    if (idx + 1) % 10 == 0:
        print(f"   → Extracted {idx + 1}/{len(masks)} segments")

print(f"   All segments extracted")

# Create Embeddings for Each Segment
print(f"\nCreating embeddings for {len(segments)} segments...")
all_embeddings = []

for idx, segment in enumerate(segments):
    # Process segment with CLIP
    inputs = clip_processor(images=segment, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        # Normalize the embeddings
        embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()[0]
    
    all_embeddings.append(embedding.tolist())
    
    if (idx + 1) % 10 == 0:
        print(f"   Created embeddings for {idx + 1}/{len(segments)} segments")

print(f"   All embeddings created (dimension: {len(all_embeddings[0])})")

# Save and Display Results
print("\nSaving results...")

# Prepare output data
output_data = {
    'num_segments': len(masks),
    'embedding_dimension': len(all_embeddings[0]),
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

# Display Summary
print(f"Total segments found: {len(masks)}")
print(f"Embedding dimension: {len(all_embeddings[0])}")
print(f"\nFirst 3 segments:")
for i in range(min(3, len(output_data['segments']))):
    seg = output_data['segments'][i]
    print(f"\nSegment {i}:")
    print(f"  - Bounding box: {seg['bbox']}")
    print(f"  - Size: {seg['segment_size']}")
    print(f"  - Confidence: {seg['confidence_score']:.3f}")
    print(f"  - Embedding (first 10 values): {seg['embedding'][:10]}")

# Optional: Visualize some segments
print("\nCreating visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(min(6, len(segments))):
    axes[i].imshow(segments[i])
    axes[i].set_title(f"Segment {i}\nScore: {scores[i]:.3f}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("segment_visualization.png", dpi=150, bbox_inches='tight')
print("   Saved visualization to 'segment_visualization.png'")