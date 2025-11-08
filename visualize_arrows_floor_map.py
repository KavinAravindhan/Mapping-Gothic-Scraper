import cv2
import pandas as pd
import numpy as np

# Configuration
building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
base_path = f"maps_output/{building_name}"
floor_plan_path = f"{base_path}/{building_name}_floorplan.jpg"
csv_path = f"{base_path}/coordinates.csv"
output_path = f"{base_path}/{building_name}_visualization.jpg"

# Arrow parameters
arrow_length = 40  # pixels
arrow_thickness = 2
arrow_color = (255, 100, 0)  # BGR format - blue arrows

# Coordinate adjustment parameters
# Positive values shift arrows right/down, negative values shift left/up
x_offset = 0
y_offset = 170

# Scale adjustment (if the coordinate system is slightly different)
# 1.0 means no scaling
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

# Load floor plan
floor_plan = cv2.imread(floor_plan_path)
plan_height, plan_width = floor_plan.shape[:2]
print(f"Floor plan dimensions: {plan_width} x {plan_height}")

# Read coordinates to find the range
df = pd.read_csv(csv_path)

# Find min/max coordinates to determine canvas size
x_coords = df['x'].values
y_coords = df['y'].values
min_x, max_x = x_coords.min(), x_coords.max()
min_y, max_y = y_coords.min(), y_coords.max()

print(f"X range: [{min_x:.3f}, {max_x:.3f}]")
print(f"Y range: [{min_y:.3f}, {max_y:.3f}]")

# Calculate padding needed (add extra margin for arrows)
margin = 100  # extra pixels for arrow visibility
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

# Draw arrows for each coordinate
for idx, row in df.iterrows():
    # Convert normalized coordinates to pixel coordinates on expanded canvas
    # Apply scale and offset adjustments
    x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
    y = int(row['y'] * plan_height * y_scale) + top_padding + y_offset
    
    direction = row['direction']
    image_id = row['image_id']
    
    if direction in direction_angles:
        # Calculate arrow end point based on direction
        angle_rad = np.radians(direction_angles[direction])
        dx = int(arrow_length * np.cos(angle_rad))
        dy = int(-arrow_length * np.sin(angle_rad))  # negative because y increases downward
        
        end_x = x + dx
        end_y = y + dy
        
        # Draw arrow
        cv2.arrowedLine(canvas, (x, y), (end_x, end_y), 
                       arrow_color, arrow_thickness, 
                       tipLength=0.3)
        
        # Debug: print first few arrows
        # if idx < 3:
        #     print(f"Arrow {image_id}: coord=({row['x']:.3f}, {row['y']:.3f}) -> pixel=({x}, {y}) dir={direction}")

# Save output
cv2.imwrite(output_path, canvas)
print(f"\nVisualization saved to: {output_path}")