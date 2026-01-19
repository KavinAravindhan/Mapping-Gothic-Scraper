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
    
    COORDINATE SYSTEM:
    - x coordinates: normalized 0-1 (left to right) of floor plan width
    - y coordinates: in "width units" (both x and y were divided by floor plan WIDTH, not HEIGHT)
    - Conversion to pixels:
      - x_pixel = x * floor_plan_width + left_padding + x_offset
      - y_pixel = y * floor_plan_width + top_padding + y_offset
        (Note: y * floor_plan_width, NOT y * floor_plan_height)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle the dataframe once
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    current_min_distance = min_distance
    min_threshold = 10  # Minimum distance threshold in pixels (lowered to ensure we get target count)
    attempt = 1
    
    while current_min_distance >= min_threshold:
        selected_indices = []
        selected_positions = []
        
        for idx, row in df_shuffled.iterrows():
            # Calculate pixel coordinates
            # IMPORTANT: Both x and y use plan_width for conversion (not plan_height for y)
            # because y was divided by width in the original dataset
            x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
            y = int(row['y'] * plan_width * y_scale) + top_padding + y_offset
            
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
        # IMPORTANT: Both x and y use plan_width for conversion
        x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
        y = int(row['y'] * plan_width * y_scale) + top_padding + y_offset
        
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


def calculate_padding(coordinates_df, img_width, img_height, margin=100):
    """
    Calculate padding needed based on coordinate ranges
    
    IMPORTANT COORDINATE SYSTEM NOTE:
    - x coordinates are normalized: 0 (left) to 1 (right) of floor plan width
    - y coordinates are in "width units": both x and y were divided by floor plan WIDTH
    - To get proper y range, we need to normalize: y_normalized = y * (width / height)
    
    Returns: (left_padding, top_padding, right_padding, bottom_padding)
    """
    if coordinates_df is None or len(coordinates_df) == 0:
        # No coordinates, use default padding
        return margin, margin, margin, margin
    
    x_coords = coordinates_df['x'].values
    y_coords = coordinates_df['y'].values
    
    # Normalize y coordinates from "width units" to proper 0-1 range for height
    aspect_ratio = img_width / img_height
    y_coords_normalized = y_coords * aspect_ratio
    
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords_normalized.min(), y_coords_normalized.max()
    
    print(f"  Coordinate ranges (raw): X=[{x_coords.min():.3f}, {x_coords.max():.3f}], Y=[{y_coords.min():.3f}, {y_coords.max():.3f}]")
    print(f"  Coordinate ranges (normalized): X=[{min_x:.3f}, {max_x:.3f}], Y=[{min_y:.3f}, {max_y:.3f}]")
    
    # Calculate padding needed
    # For x: coordinates are already in proper 0-1 range
    left_padding = int(abs(min(0, min_x)) * img_width) + margin
    right_padding = int(max(0, max_x - 1) * img_width) + margin
    
    # For y: use normalized coordinates
    top_padding = int(abs(min(0, min_y)) * img_height) + margin
    bottom_padding = int(max(0, max_y - 1) * img_height) + margin
    
    return left_padding, top_padding, right_padding, bottom_padding


def create_arrow_visualization(building_name, source_base_path, output_base_path, 
                               num_arrows, min_distance_pixels=100, random_seed=42,
                               arrow_length=60, arrow_thickness=3,
                               label_font_scale=1.0, label_font_thickness=3,
                               x_offset=0, y_offset=0, x_scale=1.0, y_scale=1.0,
                               margin=100):
    """
    Create arrow visualization for a single building with a specific arrow count
    
    COORDINATE SYSTEM:
    - x: normalized 0-1 (divided by floor plan width)
    - y: in "width units" (divided by floor plan WIDTH, not HEIGHT)
    - This means y can exceed 1.0 for floor plans where height > width
    
    Returns: (success, unusual_aspect, actual_arrow_count)
    """
    # Paths
    floor_plan_path = os.path.join(source_base_path, building_name, f"{building_name}_floorplan.jpg")
    csv_path = os.path.join(source_base_path, building_name, "coordinates.csv")
    
    # Check if required files exist
    if not os.path.exists(floor_plan_path):
        print(f"  Error: Floor plan not found at {floor_plan_path}")
        return False, False, 0
    
    if not os.path.exists(csv_path):
        print(f"  Error: Coordinates CSV not found at {csv_path}")
        return False, False, 0
    
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
    
    # Sanity check: floor plans should typically be wider than tall
    unusual_aspect = plan_width < plan_height
    if unusual_aspect:
        print(f"  WARNING: Unusual aspect ratio - width ({plan_width}) < height ({plan_height})")
    
    # Read coordinates
    df = pd.read_csv(csv_path)
    print(f"  Total arrows in dataset: {len(df)}")
    
    # Store the requested arrow count for folder naming
    requested_arrows = num_arrows
    
    # Check if we have enough images for the requested arrow count
    if len(df) < num_arrows:
        print(f"  WARNING: Only {len(df)} images available, cannot create {num_arrows} arrows")
        print(f"  Will create visualization with {len(df)} arrows instead")
        num_arrows = len(df)
    
    # Calculate padding based on coordinates
    left_padding, top_padding, right_padding, bottom_padding = calculate_padding(
        df, plan_width, plan_height, margin
    )
    print(f"  Padding: L={left_padding}, T={top_padding}, R={right_padding}, B={bottom_padding}")
    
    # Create expanded canvas
    canvas_width = left_padding + plan_width + right_padding
    canvas_height = top_padding + plan_height + bottom_padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    print(f"  Canvas size: {canvas_width} x {canvas_height}")
    
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
    
    # Track actual arrow count
    actual_arrow_count = len(sampled_df)
    
    # Verify we got the exact count (unless dataset is too small)
    if actual_arrow_count < num_arrows and len(df) >= num_arrows:
        print(f"  ERROR: Failed to select {num_arrows} arrows even with minimum distance")
    
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
        # IMPORTANT: Both x and y use plan_width for conversion (not plan_height for y)
        # because y was divided by width in the original dataset
        x = int(row['x'] * plan_width * x_scale) + left_padding + x_offset
        y = int(row['y'] * plan_width * y_scale) + top_padding + y_offset
        
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
    
    # Create output directory using REQUESTED arrow count, not actual count
    # This ensures we always have only 3 folders: arrows_10, arrows_15, arrows_20
    arrow_folder = f"arrows_{requested_arrows}"
    output_dir = os.path.join(output_base_path, arrow_folder, building_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    output_image_path = os.path.join(output_dir, f"{building_name}_arrows_visualization.jpg")
    cv2.imwrite(output_image_path, canvas)
    print(f"  Visualization saved to: {output_image_path}")
    
    # Save mapping to JSON
    mapping_output_path = os.path.join(output_dir, "arrow_label_mapping.json")
    with open(mapping_output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"  Label mapping saved to: {mapping_output_path}")
    print(f"  Mapping contains {len(mapping)} labeled arrows: {list(mapping.keys())}")
    
    return True, unusual_aspect, actual_arrow_count


def process_single_building(building_name, source_base_path, output_base_path, 
                            arrow_counts, min_distance_pixels=100, random_seed=42, margin=100):
    """
    Process a single building and create arrow visualizations for all arrow counts
    Returns: (successful_saves, unusual_aspect, insufficient_images_info)
    insufficient_images_info is a dict: {arrow_count: actual_count}
    """
    print(f"\n{'='*80}")
    print(f"Processing building: {building_name}")
    print(f"{'='*80}")
    
    # Check if building directory exists
    building_dir = os.path.join(source_base_path, building_name)
    if not os.path.exists(building_dir):
        print(f"Skipping {building_name}: Building directory not found at {building_dir}")
        return 0, False, {}
    
    successful_saves = 0
    unusual_aspect = False
    insufficient_images_info = {}
    
    # Process for each arrow count
    for arrow_count in arrow_counts:
        print(f"\nCreating visualization with {arrow_count} arrows")
        
        try:
            success, is_unusual, actual_count = create_arrow_visualization(
                building_name,
                source_base_path,
                output_base_path,
                arrow_count,
                min_distance_pixels,
                random_seed,
                margin=margin
            )
            if success:
                successful_saves += 1
                if actual_count < arrow_count:
                    insufficient_images_info[arrow_count] = actual_count
            if is_unusual:
                unusual_aspect = True
        except Exception as e:
            print(f"  Error creating arrow visualization for {building_name} with {arrow_count} arrows: {e}")
    
    return successful_saves, unusual_aspect, insufficient_images_info


def process_all_buildings(source_base_path, output_base_path, arrow_counts, 
                         min_distance_pixels=100, random_seed=42, margin=100):
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
    buildings_with_unusual_aspect = []
    buildings_with_insufficient_images = {}  # {building_name: {arrow_count: actual_count}}
    
    for building_name in sorted(building_folders):
        saves, unusual_aspect, insufficient_info = process_single_building(
            building_name, 
            source_base_path, 
            output_base_path, 
            arrow_counts, 
            min_distance_pixels,
            random_seed,
            margin
        )
        
        if saves > 0:
            buildings_with_data += 1
            total_successful_saves += saves
            if unusual_aspect:
                buildings_with_unusual_aspect.append(building_name)
            if insufficient_info:
                buildings_with_insufficient_images[building_name] = insufficient_info
        else:
            buildings_without_data += 1
    
    return (total_successful_saves, buildings_with_data, buildings_without_data, 
            buildings_with_unusual_aspect, buildings_with_insufficient_images)


if __name__ == "__main__":
    
    # Configuration - Hyperparameters
    ARROW_COUNTS = [10, 15, 20]  # List of arrow counts to create
    MIN_DISTANCE_PIXELS = 100    # Minimum distance between arrows in pixels (will auto-reduce if needed)
    RANDOM_SEED = 42             # Set to None for different random selection each time
    MARGIN = 100                 # Margin for padding around coordinates
    
    # Configuration - Paths
    SOURCE_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output"
    OUTPUT_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/arrows_floorplan"
    
    # Configuration - Mode
    SINGLE_BUILDING_MODE = False  # Set to True to process single building, False for all buildings
    SINGLE_BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"  # Only used if SINGLE_BUILDING_MODE is True
    
    # Process based on mode
    if SINGLE_BUILDING_MODE:
        print("Running in SINGLE BUILDING MODE")
        saves, unusual_aspect, insufficient_info = process_single_building(
            SINGLE_BUILDING_NAME,
            SOURCE_BASE_PATH,
            OUTPUT_BASE_PATH,
            ARROW_COUNTS,
            MIN_DISTANCE_PIXELS,
            RANDOM_SEED,
            MARGIN
        )
        print(f"\n{'='*80}")
        print(f"SUMMARY - Single Building Mode")
        print(f"{'='*80}")
        print(f"Building: {SINGLE_BUILDING_NAME}")
        print(f"Total arrow visualizations created: {saves}")
        print(f"Expected visualizations: {len(ARROW_COUNTS)}")
        if unusual_aspect:
            print(f"WARNING: Building has unusual aspect ratio (height > width)")
        if insufficient_info:
            print(f"\nInsufficient images for some arrow counts:")
            for arrow_count, actual_count in insufficient_info.items():
                print(f"  arrows_{arrow_count}: only {actual_count} arrows created")
        
    else:
        print("Running in ALL BUILDINGS MODE")
        result = process_all_buildings(
            SOURCE_BASE_PATH,
            OUTPUT_BASE_PATH,
            ARROW_COUNTS,
            MIN_DISTANCE_PIXELS,
            RANDOM_SEED,
            MARGIN
        )
        total_saves, buildings_with, buildings_without, unusual_buildings, insufficient_buildings = result
        
        print(f"\n{'='*80}")
        print(f"SUMMARY - All Buildings Mode")
        print(f"{'='*80}")
        print(f"Total arrow visualizations created: {total_saves}")
        print(f"Buildings with data: {buildings_with}")
        print(f"Buildings without data: {buildings_without}")
        print(f"Total buildings scanned: {buildings_with + buildings_without}")
        
        if unusual_buildings:
            print(f"\nBuildings with unusual aspect ratio (height > width): {len(unusual_buildings)}")
            for building in unusual_buildings:
                print(f"  - {building}")
        
        if insufficient_buildings:
            print(f"\nBuildings with insufficient images: {len(insufficient_buildings)}")
            for building_name, info in insufficient_buildings.items():
                print(f"  {building_name}:")
                for arrow_count, actual_count in info.items():
                    print(f"    arrows_{arrow_count}: only {actual_count} arrows (requested {arrow_count})")