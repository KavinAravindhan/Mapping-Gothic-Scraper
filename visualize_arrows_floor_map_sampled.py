# visualize_arrows_floor_map_sampled.py

import cv2
import pandas as pd
import numpy as np
import string
import json
import os


def select_non_overlapping_arrows(df, num_samples, min_distance, plan_width, plan_height, 
                                  x_scale, y_scale, left_padding, top_padding, 
                                  x_offset, y_offset, seed=None):
    """
    Select a random sample of arrows that don't overlap (maintain minimum distance).
    Uses a greedy approach: shuffle data, then select arrows one by one if they're
    far enough from already selected arrows.
    If unable to select enough arrows, automatically reduces min_distance by 75% and retries.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle the dataframe once
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    current_min_distance = min_distance
    min_threshold = 20  # Minimum distance threshold in pixels
    attempt = 1
    
    while current_min_distance >= min_threshold:
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
                if distance < current_min_distance:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                selected_indices.append(df_shuffled.loc[idx].name)
                selected_positions.append((x, y))
                
                if len(selected_indices) >= num_samples:
                    break
        
        # Check if we got enough arrows
        if len(selected_indices) >= num_samples:
            selected_df = df_shuffled.loc[selected_indices].reset_index(drop=True)
            if attempt == 1:
                print(f"  Selected {len(selected_df)} arrows out of {len(df)} total arrows (min distance: {current_min_distance:.1f}px)")
            else:
                print(f"  Selected {len(selected_df)} arrows out of {len(df)} total arrows (min distance: {current_min_distance:.1f}px, attempt {attempt})")
            return selected_df
        
        # Not enough arrows, reduce distance and try again
        print(f"  Attempt {attempt}: Could only select {len(selected_indices)} arrows with min_distance={current_min_distance:.1f}px")
        current_min_distance = current_min_distance * 0.75
        attempt += 1
    
    # If we exit the loop, we've hit the minimum threshold
    # Return whatever we could select with the minimum distance
    selected_indices = []
    selected_positions = []
    
    for idx, row in df_shuffled.iterrows():
        x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
        y = int(row['y'] * plan_height * y_scale) + top_padding + y_offset
        
        is_far_enough = True
        for prev_x, prev_y in selected_positions:
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            if distance < min_threshold:
                is_far_enough = False
                break
        
        if is_far_enough:
            selected_indices.append(df_shuffled.loc[idx].name)
            selected_positions.append((x, y))
            
            if len(selected_indices) >= num_samples:
                break
    
    selected_df = df_shuffled.loc[selected_indices].reset_index(drop=True)
    print(f"  Warning: Could only select {len(selected_df)} arrows (requested {num_samples})")
    print(f"  Stopped at minimum threshold of {min_threshold}px")
    
    return selected_df


def create_arrow_visualization(building_name, source_base_path, output_base_path, 
                               num_arrows, min_distance_pixels=200, random_seed=42,
                               arrow_length=60,        # Increase from 40 to 60
                               arrow_thickness=3,      # Increase from 2 to 3
                               label_font_scale=1.0,   # Increase from 0.7 to 1.0
                               label_font_thickness=3, # Increase from 2 to 3
                               x_offset=0, y_offset=170, x_scale=1.0, y_scale=1.0):
    """
    Create arrow visualization for a single building with a specific arrow count
    """
    # Paths
    floor_plan_path = os.path.join(source_base_path, building_name, f"{building_name}_floorplan.jpg")
    csv_path = os.path.join(source_base_path, building_name, "coordinates.csv")
    
    # Check if required files exist
    if not os.path.exists(floor_plan_path):
        print(f"  Error: Floor plan not found at {floor_plan_path}")
        return False
    
    if not os.path.exists(csv_path):
        print(f"  Error: Coordinates CSV not found at {csv_path}")
        return False
    
    # Arrow and label parameters
    arrow_color = (255, 100, 0)  # BGR format - blue arrows
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (0, 0, 255)  # BGR format - red text
    label_offset_x = 25  # pixels away from arrow base
    label_offset_y = -10  # pixels away from arrow base
    
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
    print(f"  Floor plan dimensions: {plan_width} x {plan_height}")
    
    # Read coordinates
    df = pd.read_csv(csv_path)
    print(f"  Total arrows in dataset: {len(df)}")
    
    # Find min/max coordinates to determine canvas size
    x_coords = df['x'].values
    y_coords = df['y'].values
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    
    print(f"  X range: [{min_x:.3f}, {max_x:.3f}]")
    print(f"  Y range: [{min_y:.3f}, {max_y:.3f}]")
    
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
        df, num_arrows, min_distance_pixels, 
        plan_width, plan_height, x_scale, y_scale,
        left_padding, top_padding, x_offset, y_offset,
        seed=random_seed
    )
    
    # Generate letter labels (A, B, C, ...)
    letters = list(string.ascii_uppercase)
    if len(sampled_df) > 26:
        # If more than 26 arrows, use AA, AB, AC, etc.
        letters = letters + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    
    # Create mapping dictionary for evaluation
    mapping = {}
    
    # Draw arrows and labels for sampled coordinates
    for idx, row in sampled_df.iterrows():
        # Convert normalized coordinates to pixel coordinates
        x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
        y = int(row['y'] * plan_height * y_scale) + top_padding + y_offset
        
        direction = row['direction']
        image_id = row['image_id']
        label = letters[idx]
        
        # Add to mapping
        mapping[label] = {
            'image_id': image_id,
            'x': float(row['x']),
            'y': float(row['y']),
            'direction': direction
        }
        
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
            
            print(f"  Arrow {label} ({image_id}): coord=({row['x']:.3f}, {row['y']:.3f}) -> pixel=({x}, {y}) dir={direction}")
    
    # Create output directory
    arrow_folder = f"arrows_{num_arrows}"
    output_dir = os.path.join(output_base_path, arrow_folder, building_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    output_image_path = os.path.join(output_dir, f"{building_name}_arrows_visualization.jpg")
    cv2.imwrite(output_image_path, canvas)
    print(f"  Visualization saved to: {output_image_path}")
    print(f"  Canvas size: {canvas_width} x {canvas_height}")
    
    # Save mapping to JSON
    mapping_output_path = os.path.join(output_dir, "arrow_label_mapping.json")
    with open(mapping_output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"  Label mapping saved to: {mapping_output_path}")
    print(f"  Mapping contains {len(mapping)} labeled arrows: {list(mapping.keys())}")
    
    return True


def process_single_building(building_name, source_base_path, output_base_path, 
                            arrow_counts, min_distance_pixels=200, random_seed=42):
    """
    Process a single building and create arrow visualizations for all arrow counts
    Returns number of successful saves
    """
    print(f"\n{'='*80}")
    print(f"Processing building: {building_name}")
    print(f"{'='*80}")
    
    # Check if building directory exists
    building_dir = os.path.join(source_base_path, building_name)
    if not os.path.exists(building_dir):
        print(f"Skipping {building_name}: Building directory not found at {building_dir}")
        return 0
    
    successful_saves = 0
    
    # Process for each arrow count
    for arrow_count in arrow_counts:
        print(f"\nCreating visualization with {arrow_count} arrows")
        
        try:
            success = create_arrow_visualization(
                building_name,
                source_base_path,
                output_base_path,
                arrow_count,
                min_distance_pixels,
                random_seed
            )
            if success:
                successful_saves += 1
        except Exception as e:
            print(f"  Error creating arrow visualization for {building_name} with {arrow_count} arrows: {e}")
    
    return successful_saves


def process_all_buildings(source_base_path, output_base_path, arrow_counts, 
                         min_distance_pixels=200, random_seed=42):
    """
    Process all buildings in the source directory
    Returns total number of successful saves and buildings processed
    """
    print(f"\n{'='*80}")
    print(f"Processing all buildings from: {source_base_path}")
    print(f"Output directory: {output_base_path}")
    print(f"Arrow counts: {arrow_counts}")
    print(f"{'='*80}")
    
    # Get all subdirectories
    building_folders = [f for f in os.listdir(source_base_path) 
                       if os.path.isdir(os.path.join(source_base_path, f))]
    
    print(f"\nFound {len(building_folders)} building folders")
    
    total_successful_saves = 0
    buildings_with_data = 0
    buildings_without_data = 0
    
    for building_name in sorted(building_folders):
        saves = process_single_building(
            building_name, 
            source_base_path, 
            output_base_path, 
            arrow_counts, 
            min_distance_pixels,
            random_seed
        )
        
        if saves > 0:
            buildings_with_data += 1
            total_successful_saves += saves
        else:
            buildings_without_data += 1
    
    return total_successful_saves, buildings_with_data, buildings_without_data


if __name__ == "__main__":
    
    # Configuration - Hyperparameters
    ARROW_COUNTS = [10, 15, 20]  # List of arrow counts to create
    MIN_DISTANCE_PIXELS = 100    # Minimum distance between arrows in pixels
    RANDOM_SEED = 42             # Set to None for different random selection each time
    
    # Configuration - Paths
    SOURCE_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output"
    OUTPUT_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/arrows_floorplan"
    
    # Configuration - Mode
    SINGLE_BUILDING_MODE = False  # Set to True to process single building, False for all buildings
    SINGLE_BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"  # Only used if SINGLE_BUILDING_MODE is True
    
    # Process based on mode
    if SINGLE_BUILDING_MODE:
        print("Running in SINGLE BUILDING MODE")
        saves = process_single_building(
            SINGLE_BUILDING_NAME,
            SOURCE_BASE_PATH,
            OUTPUT_BASE_PATH,
            ARROW_COUNTS,
            MIN_DISTANCE_PIXELS,
            RANDOM_SEED
        )
        print(f"\n{'='*80}")
        print(f"SUMMARY - Single Building Mode")
        print(f"{'='*80}")
        print(f"Building: {SINGLE_BUILDING_NAME}")
        print(f"Total arrow visualizations created: {saves}")
        print(f"Expected visualizations: {len(ARROW_COUNTS)}")
        
    else:
        print("Running in ALL BUILDINGS MODE")
        total_saves, buildings_with, buildings_without = process_all_buildings(
            SOURCE_BASE_PATH,
            OUTPUT_BASE_PATH,
            ARROW_COUNTS,
            MIN_DISTANCE_PIXELS,
            RANDOM_SEED
        )
        
        print(f"\n{'='*80}")
        print(f"SUMMARY - All Buildings Mode")
        print(f"{'='*80}")
        print(f"Total arrow visualizations created: {total_saves}")
        print(f"Buildings with data: {buildings_with}")
        print(f"Buildings without data: {buildings_without}")
        print(f"Total buildings scanned: {buildings_with + buildings_without}")