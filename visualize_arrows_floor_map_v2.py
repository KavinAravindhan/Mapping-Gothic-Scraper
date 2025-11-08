import cv2
import pandas as pd
import numpy as np

# Configuration
building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
base_path = f"maps_output/{building_name}"
floor_plan_path = f"{base_path}/{building_name}_floorplan.jpg"
csv_path = f"{base_path}/coordinates.csv"
output_path = f"{base_path}/{building_name}arrows_visualization_v2.jpg"

# Arrow parameters
arrow_length = 40  # pixels
arrow_thickness = 2
arrow_color = (255, 100, 0)  # BGR format - blue arrows

# OFFSET ADJUSTMENTS
x_offset = 0
y_offset = 100

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

# The website displays at 1554 x 1244, but our floor plan is 2000 x 1567
website_width = 1554
website_height = 1244

# Read coordinates to find the range
df = pd.read_csv(csv_path)

# Find min/max coordinates to determine canvas size (using website dimensions)
x_coords = df['x'].values
y_coords = df['y'].values
min_x, max_x = x_coords.min(), x_coords.max()
min_y, max_y = y_coords.min(), y_coords.max()

# Calculate padding needed
margin = 100  # extra pixels for arrow visibility
left_padding = int(abs(min(0, min_x)) * website_width) + margin
right_padding = int(max(0, max_x - 1) * website_width) + margin
top_padding = int(abs(min(0, min_y)) * website_height) + margin
bottom_padding = int(max(0, max_y - 1) * website_height) + margin

# Calculate scaled floor plan size
scaled_plan_width = website_width
scaled_plan_height = website_height

# Resize floor plan to match website dimensions
floor_plan_resized = cv2.resize(floor_plan, (scaled_plan_width, scaled_plan_height))

# Create expanded canvas
canvas_width = left_padding + scaled_plan_width + right_padding
canvas_height = top_padding + scaled_plan_height + bottom_padding
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Place resized floor plan on canvas
canvas[top_padding:top_padding+scaled_plan_height, 
       left_padding:left_padding+scaled_plan_width] = floor_plan_resized

# Draw arrows for each coordinate
for idx, row in df.iterrows():
    # Convert normalized coordinates to pixel coordinates (using website dimensions)
    x = int(row['x'] * website_width) + left_padding + x_offset
    y = int(row['y'] * website_height) + top_padding + y_offset
    
    direction = row['direction']
    
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

# Save output
cv2.imwrite(output_path, canvas)
print(f"\nVisualization saved to: {output_path}")