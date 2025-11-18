import cv2
import pandas as pd
import numpy as np
import string

# Configuration
building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
base_path = f"maps_output/{building_name}"
floor_plan_path = f"{base_path}/{building_name}_floorplan.jpg"
csv_path = f"{base_path}/coordinates.csv"
output_path = f"{base_path}/{building_name}_arrows_visualization.jpg"

# Hyperparameters
num_arrows_to_sample = 10  # Number of arrows to randomly sample
min_distance_pixels = 80    # Minimum distance between arrows in pixels
random_seed = 42            # Set to None for different random selection each time

# Arrow parameters
arrow_length = 40  # pixels
arrow_thickness = 2
arrow_color = (255, 100, 0)  # BGR format - blue arrows

# Label parameters
label_font = cv2.FONT_HERSHEY_SIMPLEX
label_font_scale = 0.7
label_font_thickness = 2
label_color = (0, 0, 255)  # BGR format - red text
label_offset_x = 25  # pixels away from arrow base
label_offset_y = -10  # pixels away from arrow base

# Coordinate adjustment parameters
x_offset = 0
y_offset = 170
x_scale = 1.0
y_scale = 1.0

# Direction mappings (angle in degrees, 0 = right, 90 = up)
direction_angles = {
    'E': 0,
    'NE': 45,
    'N': 90,
    'NW': 135,
    'W': 180,
    'SW': 225,
    'S': 270,
    'SE': 315
}

def select_non_overlapping_arrows(df, num_samples, min_distance, plan_width, plan_height, 
                                  x_scale, y_scale, left_padding, top_padding, 
                                  x_offset, y_offset, seed=None):
    """
    Select a random sample of arrows that don't overlap (maintain minimum distance).
    Uses a greedy approach: shuffle data, then select arrows one by one if they're
    far enough from already selected arrows.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    selected_indices = []
    selected_positions = []
    
    for idx, row in df_shuffled.iterrows():
        # Calculate pixel coordinates
        x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
        y = int(row['y'] * plan_height * y_scale) + top_padding + y_offset
        
        # Check if this arrow is far enough from all previously selected arrows
        is_far_enough = True
        for prev_x, prev_y in selected_positions:
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            if distance < min_distance:
                is_far_enough = False
                break
        
        if is_far_enough:
            selected_indices.append(df_shuffled.loc[idx].name)
            selected_positions.append((x, y))
            
            if len(selected_indices) >= num_samples:
                break
    
    # Return the selected subset with original indices
    selected_df = df_shuffled.loc[selected_indices].reset_index(drop=True)
    
    print(f"\nSelected {len(selected_df)} arrows out of {len(df)} total arrows")
    if len(selected_df) < num_samples:
        print(f"Warning: Could only select {len(selected_df)} non-overlapping arrows")
        print(f"(requested {num_samples}). Try reducing min_distance_pixels or num_arrows_to_sample.")
    
    return selected_df

# Load floor plan
floor_plan = cv2.imread(floor_plan_path)
plan_height, plan_width = floor_plan.shape[:2]
print(f"Floor plan dimensions: {plan_width} x {plan_height}")

# Read coordinates
df = pd.read_csv(csv_path)
print(f"Total arrows in dataset: {len(df)}")

# Find min/max coordinates to determine canvas size
x_coords = df['x'].values
y_coords = df['y'].values
min_x, max_x = x_coords.min(), x_coords.max()
min_y, max_y = y_coords.min(), y_coords.max()

print(f"X range: [{min_x:.3f}, {max_x:.3f}]")
print(f"Y range: [{min_y:.3f}, {max_y:.3f}]")

# Calculate padding needed
margin = 100
left_padding = int(abs(min(0, min_x)) * plan_width * x_scale) + margin
right_padding = int(max(0, max_x - 1) * plan_width * x_scale) + margin
top_padding = int(abs(min(0, min_y)) * plan_height * y_scale) + margin
bottom_padding = int(max(0, max_y - 1) * plan_height * y_scale) + margin

# Create expanded canvas
canvas_width = left_padding + plan_width + right_padding
canvas_height = top_padding + plan_height + bottom_padding
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Place floor plan on canvas
canvas[top_padding:top_padding+plan_height, 
       left_padding:left_padding+plan_width] = floor_plan

# Select non-overlapping sample of arrows
sampled_df = select_non_overlapping_arrows(
    df, num_arrows_to_sample, min_distance_pixels, 
    plan_width, plan_height, x_scale, y_scale,
    left_padding, top_padding, x_offset, y_offset,
    seed=random_seed
)

# Generate letter labels (A, B, C, ...)
letters = list(string.ascii_uppercase)
if len(sampled_df) > 26:
    # If more than 26 arrows, use AA, AB, AC, etc.
    letters = letters + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]

# Draw arrows and labels for sampled coordinates
for idx, row in sampled_df.iterrows():
    # Convert normalized coordinates to pixel coordinates
    x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
    y = int(row['y'] * plan_height * y_scale) + top_padding + y_offset
    
    direction = row['direction']
    image_id = row['image_id']
    label = letters[idx]
    
    if direction in direction_angles:
        # Calculate arrow end point based on direction
        angle_rad = np.radians(direction_angles[direction])
        dx = int(arrow_length * np.cos(angle_rad))
        dy = int(-arrow_length * np.sin(angle_rad))
        
        end_x = x + dx
        end_y = y + dy
        
        # Draw arrow
        cv2.arrowedLine(canvas, (x, y), (end_x, end_y), 
                       arrow_color, arrow_thickness, 
                       tipLength=0.3)
        
        # Draw label with background for better visibility
        label_x = x + label_offset_x
        label_y = y + label_offset_y
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, label_font, label_font_scale, label_font_thickness
        )
        
        # Draw white background rectangle
        cv2.rectangle(canvas, 
                     (label_x - 2, label_y - text_height - 2),
                     (label_x + text_width + 2, label_y + baseline + 2),
                     (255, 255, 255), -1)
        
        # Draw text label
        cv2.putText(canvas, label, (label_x, label_y),
                   label_font, label_font_scale, label_color, 
                   label_font_thickness, cv2.LINE_AA)
        
        print(f"Arrow {label} ({image_id}): coord=({row['x']:.3f}, {row['y']:.3f}) -> pixel=({x}, {y}) dir={direction}")

# Save output
cv2.imwrite(output_path, canvas)
print(f"\nVisualization saved to: {output_path}")
print(f"Canvas size: {canvas_width} x {canvas_height}")