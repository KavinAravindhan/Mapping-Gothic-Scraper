import os
from datetime import datetime
import urllib.request
import csv
from io import StringIO

# Environment Setup
HF_CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"
CUDA_DEVICE = "2"

os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# Imports
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

# SAM3 for segmentation
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

# DINOv2 for embeddings
from transformers import (
    AutoImageProcessor,
    Dinov2Model
)

# CLIP for embeddings (COMMENTED OUT - using DINOv2 instead)
# from transformers import CLIPProcessor, CLIPModel

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# CONFIGURATION
# =============================================================================

# Image URL
IMAGE_URL = "https://storage.googleapis.com/metrics_annotation_images/architecture/English%20Gothic%20%2820%29/Hadleigh/_ZAC2729.jpg"

# Annotation data (from CSV)
ANNOTATION = {
    'bbox_x_percent': 11.735006119951043,
    'bbox_y_percent': 22.480620155038757,
    'bbox_width_percent': 27.209739464941425,
    'bbox_height_percent': 67.44186046511628,
    'bbox_category': 'arches',
    'bbox_subcategory': 'pointed',
    'bbox_description': 'lancet arch',
    'original_image_width': 4256.0,
    'original_image_height': 2832.0,
}

# Search parameters
SIMILARITY_THRESHOLD = 0.85  # Minimum cosine similarity to consider as "similar"
TOP_K_RESULTS = 10  # Number of top similar segments to find
GRID_DENSITY = 32  # Points per side for dense sampling

# Output directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/mnt/swordfish-pool2/kavin/sam3/bbox_similarity_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SAM3 Bounding Box Similarity Search")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Searching for: {ANNOTATION['bbox_category']} - {ANNOTATION['bbox_subcategory']}")
print("=" * 80)

# =============================================================================
# SETUP MODELS
# =============================================================================

print("\n[1/8] Setting up device and loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Load SAM3 Tracker
print("   Loading SAM3 Tracker model...")
sam_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
sam_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
print("   ✓ SAM3 Tracker loaded")

# Load DINOv2 for embeddings
print("   Loading DINOv2 model...")
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
print("   ✓ DINOv2 loaded")

# CLIP VERSION (COMMENTED OUT)
# print("   Loading CLIP model...")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# print("   ✓ CLIP loaded")

# =============================================================================
# LOAD IMAGE
# =============================================================================

print(f"\n[2/8] Downloading image from GCS...")
response = requests.get(IMAGE_URL)
image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert("RGB")
print(f"   ✓ Image loaded: {image.size[0]}x{image.size[1]} pixels")

# Save original image
original_path = os.path.join(OUTPUT_DIR, "00_original_image.jpg")
image.save(original_path, quality=95)
print(f"   ✓ Saved original image")

# =============================================================================
# CONVERT BBOX PERCENTAGES TO PIXELS
# =============================================================================

print(f"\n[3/8] Processing target bounding box...")

# Convert percentages to pixel coordinates
img_width, img_height = image.size
x_percent = ANNOTATION['bbox_x_percent']
y_percent = ANNOTATION['bbox_y_percent']
w_percent = ANNOTATION['bbox_width_percent']
h_percent = ANNOTATION['bbox_height_percent']

# Calculate pixel coordinates (xyxy format)
x1 = int((x_percent / 100) * img_width)
y1 = int((y_percent / 100) * img_height)
x2 = int(((x_percent + w_percent) / 100) * img_width)
y2 = int(((y_percent + h_percent) / 100) * img_height)

target_bbox_xyxy = [x1, y1, x2, y2]
print(f"   Target bbox (xyxy pixels): {target_bbox_xyxy}")
print(f"   Category: {ANNOTATION['bbox_category']} - {ANNOTATION['bbox_subcategory']}")

# =============================================================================
# SEGMENT TARGET BOUNDING BOX
# =============================================================================

print(f"\n[4/8] Segmenting target bounding box with SAM3...")

# Use SAM3 with bounding box prompt
input_boxes = [[[target_bbox_xyxy[0], target_bbox_xyxy[1], target_bbox_xyxy[2], target_bbox_xyxy[3]]]]

inputs = sam_processor(
    images=image,
    input_boxes=input_boxes,  # No labels needed for Sam3Tracker
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = sam_model(**inputs, multimask_output=False)

# Get the mask
target_mask = sam_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"]
)[0][0, 0].numpy()

print(f"   ✓ Target segment mask created")

# Extract target segment for embedding
x1, y1, x2, y2 = target_bbox_xyxy
target_segment = image.crop((x1, y1, x2, y2))
mask_crop = target_mask[y1:y2, x1:x2]

# Apply mask
segment_array = np.array(target_segment)
if mask_crop.shape[:2] == segment_array.shape[:2]:
    mask_3channel = np.stack([mask_crop] * 3, axis=-1)
    masked_segment = (segment_array * mask_3channel).astype(np.uint8)
    target_segment_masked = Image.fromarray(masked_segment)
else:
    target_segment_masked = target_segment

# Save target segment
target_segment_path = os.path.join(OUTPUT_DIR, "01_target_segment.png")
target_segment_masked.save(target_segment_path)
print(f"   ✓ Saved target segment")

# =============================================================================
# CREATE TARGET EMBEDDING
# =============================================================================

print(f"\n[5/8] Creating embedding for target segment...")

# DINOv2 VERSION
inputs = dinov2_processor(images=target_segment_masked, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = dinov2_model(**inputs)
    target_embedding = outputs.last_hidden_state[:, 0]
    target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.cpu().numpy()[0]

# CLIP VERSION (COMMENTED OUT)
# inputs = clip_processor(images=target_segment_masked, return_tensors="pt").to(device)
# with torch.no_grad():
#     image_features = clip_model.get_image_features(**inputs)
#     target_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
#     target_embedding = target_embedding.cpu().numpy()[0]

print(f"   ✓ Target embedding created (dimension: {len(target_embedding)})")

# Visualize target bbox on original image
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.imshow(image)
rect = patches.Rectangle(
    (x1, y1), x2-x1, y2-y1,
    linewidth=3, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)
ax.text(x1, y1-10, f"Target: {ANNOTATION['bbox_subcategory']}", 
        color='red', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.axis('off')
ax.set_title(f"Target Bounding Box: {ANNOTATION['bbox_category']}", fontsize=14)
plt.tight_layout()
target_viz_path = os.path.join(OUTPUT_DIR, "02_target_bbox_overlay.png")
plt.savefig(target_viz_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved target bbox visualization")

# =============================================================================
# DENSE SEGMENTATION OF ENTIRE IMAGE
# =============================================================================

print(f"\n[6/8] Running dense segmentation across entire image...")
print(f"   Grid density: {GRID_DENSITY}x{GRID_DENSITY} = {GRID_DENSITY**2} points")
print(f"   This may take several minutes...")

all_segments = []
all_masks = []
all_boxes = []
all_scores = []
all_embeddings = []

x_points = np.linspace(0, img_width, GRID_DENSITY, endpoint=False)
y_points = np.linspace(0, img_height, GRID_DENSITY, endpoint=False)

for i, x in enumerate(x_points):
    for j, y in enumerate(y_points):
        if (i * len(y_points) + j) % 100 == 0:
            progress = ((i * len(y_points) + j) / (GRID_DENSITY ** 2)) * 100
            print(f"   Progress: {progress:.1f}% ({len(all_segments)} segments found)")
        
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
        
        iou_scores = outputs.iou_scores[0].cpu().numpy().flatten()
        best_idx = np.argmax(iou_scores)
        
        mask = masks_tensor[0, best_idx].numpy()
        score = float(iou_scores[best_idx])
        
        if score < 0.5:
            continue
        
        mask_bool = mask > 0.5
        if not mask_bool.any():
            continue
        
        coords = np.column_stack(np.where(mask_bool))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = [x_min, y_min, x_max, y_max]
        
        area = (x_max - x_min) * (y_max - y_min)
        if area < 100:  # Skip very small segments
            continue
        
        # Check uniqueness
        is_unique = True
        for existing_mask in all_masks:
            intersection = np.logical_and(mask_bool, existing_mask > 0.5).sum()
            union = np.logical_or(mask_bool, existing_mask > 0.5).sum()
            iou = intersection / (union + 1e-6)
            if iou > 0.85:
                is_unique = False
                break
        
        if not is_unique:
            continue
        
        # Extract and mask segment
        seg_img = image.crop((x_min, y_min, x_max, y_max))
        mask_crop = mask[y_min:y_max, x_min:x_max]
        seg_array = np.array(seg_img)
        
        if mask_crop.shape[:2] == seg_array.shape[:2]:
            mask_3ch = np.stack([mask_crop] * 3, axis=-1)
            masked_seg = (seg_array * mask_3ch).astype(np.uint8)
            seg_masked = Image.fromarray(masked_seg)
        else:
            seg_masked = seg_img
        
        # Create embedding for this segment
        inputs_emb = dinov2_processor(images=seg_masked, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_emb = dinov2_model(**inputs_emb)
            embedding = outputs_emb.last_hidden_state[:, 0]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy()[0]
        
        # CLIP VERSION (COMMENTED OUT)
        # inputs_emb = clip_processor(images=seg_masked, return_tensors="pt").to(device)
        # with torch.no_grad():
        #     features = clip_model.get_image_features(**inputs_emb)
        #     embedding = features / features.norm(dim=-1, keepdim=True)
        #     embedding = embedding.cpu().numpy()[0]
        
        all_segments.append(seg_masked)
        all_masks.append(mask)
        all_boxes.append(bbox)
        all_scores.append(score)
        all_embeddings.append(embedding)

print(f"   ✓ Found {len(all_segments)} total segments")

# =============================================================================
# FIND SIMILAR SEGMENTS
# =============================================================================

print(f"\n[7/8] Finding segments similar to target...")

# Compute cosine similarities
embeddings_array = np.array(all_embeddings)
similarities = np.dot(embeddings_array, target_embedding)

# Get top-k most similar
top_k_indices = np.argsort(similarities)[::-1][:TOP_K_RESULTS]
top_k_similarities = similarities[top_k_indices]

# Filter by threshold
filtered_indices = [idx for idx, sim in zip(top_k_indices, top_k_similarities) 
                   if sim >= SIMILARITY_THRESHOLD]

print(f"   ✓ Found {len(filtered_indices)} similar segments (threshold: {SIMILARITY_THRESHOLD})")
print(f"   Top similarities: {top_k_similarities[:5]}")

# Save results data
results_data = {
    'target_bbox': {
        'xyxy': target_bbox_xyxy,
        'category': ANNOTATION['bbox_category'],
        'subcategory': ANNOTATION['bbox_subcategory'],
        'description': ANNOTATION['bbox_description'],
    },
    'search_params': {
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'top_k': TOP_K_RESULTS,
        'grid_density': GRID_DENSITY,
    },
    'similar_segments': []
}

for rank, idx in enumerate(filtered_indices):
    bbox = all_boxes[idx]
    results_data['similar_segments'].append({
        'rank': rank + 1,
        'bbox_xyxy': [int(x) for x in bbox],
        'similarity_score': float(similarities[idx]),
        'confidence_score': float(all_scores[idx]),
    })

results_json_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_json_path, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f"   ✓ Saved results JSON")

# =============================================================================
# VISUALIZE RESULTS
# =============================================================================

print(f"\n[8/8] Creating visualizations...")

# Visualization 1: All similar segments overlayed
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.imshow(image)

# Draw target bbox in red
x1, y1, x2, y2 = target_bbox_xyxy
rect = patches.Rectangle(
    (x1, y1), x2-x1, y2-y1,
    linewidth=4, edgecolor='red', facecolor='none', linestyle='--'
)
ax.add_patch(rect)
ax.text(x1, y1-15, "TARGET", color='red', fontsize=14, weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Draw similar segments in green
for rank, idx in enumerate(filtered_indices[:TOP_K_RESULTS]):
    bbox = all_boxes[idx]
    sim = similarities[idx]
    x1, y1, x2, y2 = bbox
    
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=3, edgecolor='lime', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x1, y1-10, f"#{rank+1} ({sim:.3f})", 
            color='lime', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax.axis('off')
ax.set_title(f"Similar {ANNOTATION['bbox_category']} Found: {len(filtered_indices)} instances", 
             fontsize=16, weight='bold')
plt.tight_layout()
overlay_path = os.path.join(OUTPUT_DIR, "03_all_similar_segments_overlay.png")
plt.savefig(overlay_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved overlay visualization")

# Visualization 2: Grid of top similar segments
num_display = min(12, len(filtered_indices))
if num_display > 0:
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(num_display):
        idx = filtered_indices[i]
        axes[i].imshow(all_segments[idx])
        sim = similarities[idx]
        axes[i].set_title(f"Rank #{i+1}\nSimilarity: {sim:.4f}", fontsize=11)
        axes[i].axis('off')
    
    for i in range(num_display, 12):
        axes[i].axis('off')
    
    plt.suptitle(f"Top {num_display} Most Similar Segments", fontsize=16, weight='bold')
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "04_top_similar_segments_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved grid visualization")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Target: {ANNOTATION['bbox_category']} - {ANNOTATION['bbox_subcategory']}")
print(f"Target bbox: {target_bbox_xyxy}")
print(f"Total segments searched: {len(all_segments)}")
print(f"Similar segments found: {len(filtered_indices)}")
print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
print(f"\nTop 5 similarities:")
for i, (idx, sim) in enumerate(zip(filtered_indices[:5], top_k_similarities[:5])):
    bbox = all_boxes[idx]
    print(f"  #{i+1}: Similarity {sim:.4f}, BBox: {bbox}")
print("\n" + "=" * 80)
print("FILES SAVED:")
print(f"  - {os.path.join(OUTPUT_DIR, '00_original_image.jpg')}")
print(f"  - {os.path.join(OUTPUT_DIR, '01_target_segment.png')}")
print(f"  - {os.path.join(OUTPUT_DIR, '02_target_bbox_overlay.png')}")
print(f"  - {os.path.join(OUTPUT_DIR, '03_all_similar_segments_overlay.png')}")
print(f"  - {os.path.join(OUTPUT_DIR, '04_top_similar_segments_grid.png')}")
print(f"  - {os.path.join(OUTPUT_DIR, 'results.json')}")
print("=" * 80)
print("\n✓ COMPLETE!")