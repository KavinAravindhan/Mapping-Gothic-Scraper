# create_grid_overlay.py

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np


def load_font(label_size=80):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, label_size)
            print(f"Loaded font: {font_path}")
            return font
        except:
            continue
    
    print("Using default font")
    return ImageFont.load_default()


def get_text_dimensions(draw, text, font):
    """Get text width and height in a compatible way"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        # Fallback estimate
        return len(text) * 40, 60


def generate_column_labels(num_columns):
    """
    Generate column labels dynamically (A, B, C, ..., Z, AA, AB, ...)
    """
    labels = []
    for i in range(num_columns):
        label = ""
        num = i
        while True:
            label = chr(65 + (num % 26)) + label
            num = num // 26
            if num == 0:
                break
            num -= 1
        labels.append(label)
    return labels


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


def create_padded_image(image, left_pad, top_pad, right_pad, bottom_pad):
    """
    Create a padded version of the image with white background
    """
    img_width, img_height = image.size
    
    # Create new image with padding
    padded_width = left_pad + img_width + right_pad
    padded_height = top_pad + img_height + bottom_pad
    padded_image = Image.new('RGB', (padded_width, padded_height), 'white')
    
    # Paste original image
    padded_image.paste(image, (left_pad, top_pad))
    
    return padded_image


def create_grid_overlay_with_padding(image_path, output_path, coordinates_df,
                                     grid_cols=10, grid_rows=10,
                                     line_color=(255, 0, 0), line_width=3,
                                     label_size=80, margin=100):
    """
    Create grid overlay on image with padding based on coordinates
    Returns: (output_path, left_padding, top_padding, padded_width, padded_height)
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size
    print(f"  Original image size: {img_width}x{img_height}")
    
    # Calculate padding based on coordinates
    left_pad, top_pad, right_pad, bottom_pad = calculate_padding(
        coordinates_df, img_width, img_height, margin
    )
    print(f"  Padding: L={left_pad}, T={top_pad}, R={right_pad}, B={bottom_pad}")
    
    # Create padded image
    padded_image = create_padded_image(image, left_pad, top_pad, right_pad, bottom_pad)
    padded_width, padded_height = padded_image.size
    print(f"  Padded image size: {padded_width}x{padded_height}")
    
    # Calculate grid spacing
    col_spacing = padded_width / grid_cols
    row_spacing = padded_height / grid_rows
    
    print(f"  Grid spacing: {col_spacing:.1f}px x {row_spacing:.1f}px")
    print(f"  Grid configuration: {grid_cols} columns x {grid_rows} rows")
    
    # Create padding for labels
    label_padding = max(120, label_size + 40)
    
    # Create new image with label padding
    final_width = padded_width + label_padding * 2
    final_height = padded_height + label_padding * 2
    final_image = Image.new('RGB', (final_width, final_height), 'white')
    final_image.paste(padded_image, (label_padding, label_padding))
    
    # Create drawing context and load font
    draw = ImageDraw.Draw(final_image)
    font = load_font(label_size)
    
    # Generate dynamic column labels
    columns = generate_column_labels(grid_cols)
    
    # Draw vertical lines and column labels
    for i in range(grid_cols + 1):
        x = label_padding + i * col_spacing
        
        # Draw vertical line
        draw.line([(x, label_padding), (x, label_padding + padded_height)],
                 fill=line_color, width=line_width)
        
        # Add column labels
        if i < grid_cols:
            letter = columns[i]
            label_x = x + col_spacing / 2
            text_width, text_height = get_text_dimensions(draw, letter, font)
            
            # Top label
            draw.text((label_x - text_width/2, label_padding/2 - text_height/2),
                     letter, fill='black', font=font)
            
            # Bottom label
            draw.text((label_x - text_width/2, label_padding + padded_height + label_padding/2 - text_height/2),
                     letter, fill='black', font=font)
    
    # Draw horizontal lines and row labels
    for i in range(grid_rows + 1):
        y = label_padding + i * row_spacing
        
        # Draw horizontal line
        draw.line([(label_padding, y), (label_padding + padded_width, y)],
                 fill=line_color, width=line_width)
        
        # Add row labels
        if i < grid_rows:
            number = str(i + 1)
            label_y = y + row_spacing / 2
            text_width, text_height = get_text_dimensions(draw, number, font)
            
            # Left label
            draw.text((label_padding/2 - text_width/2, label_y - text_height/2),
                     number, fill='black', font=font)
            
            # Right label
            draw.text((label_padding + padded_width + label_padding/2 - text_width/2, label_y - text_height/2),
                     number, fill='black', font=font)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_image.save(output_path, 'JPEG', quality=95)
    print(f"Saved: {os.path.basename(output_path)}")
    
    # Return padding info for ground truth calculation
    return output_path, left_pad, top_pad, padded_width, padded_height


def pixel_to_grid_cell(x_pixel, y_pixel, padded_width, padded_height, grid_cols, grid_rows):
    """
    Convert pixel coordinates to grid cell (e.g., 'A5')
    """
    # Calculate which column and row
    col_index = int(x_pixel / padded_width * grid_cols)
    row_index = int(y_pixel / padded_height * grid_rows)
    
    # Clamp to valid range
    col_index = max(0, min(col_index, grid_cols - 1))
    row_index = max(0, min(row_index, grid_rows - 1))
    
    # Generate column label
    columns = generate_column_labels(grid_cols)
    col_label = columns[col_index]
    row_label = str(row_index + 1)
    
    return f"{col_label}{row_label}"


def generate_ground_truth(coordinates_df, orig_img_width, orig_img_height,
                         left_pad, top_pad, padded_width, padded_height,
                         grid_size, output_path):
    """
    Generate ground truth CSV file mapping image IDs to grid cells
    
    IMPORTANT COORDINATE SYSTEM NOTE:
    - x coordinates: normalized 0-1 (left to right) of floor plan width
    - y coordinates: in "width units" (both x and y were divided by floor plan WIDTH)
    - Conversion to pixels:
      - x_pixel = x * floor_plan_width
      - y_pixel = y * floor_plan_width (NOT height!)
    - We then normalize y to proper range: y_normalized = y * (width / height)
    """
    if coordinates_df is None or len(coordinates_df) == 0:
        print(f"  No coordinates available, skipping ground truth generation")
        return
    
    ground_truth_data = []
    
    # Calculate aspect ratio for y coordinate normalization
    aspect_ratio = orig_img_width / orig_img_height
    
    for idx, row in coordinates_df.iterrows():
        image_id = row['image_id']
        x_raw = row['x']
        y_raw = row['y']
        direction = row['direction']
        
        # X coordinate: already normalized, convert directly to pixels
        # x in range [0, 1] maps to [0, orig_img_width]
        x_pixel = x_raw * orig_img_width + left_pad
        
        # Y coordinate: in "width units", need to convert to pixels using WIDTH
        # y_raw * orig_img_width gives the actual pixel position in the original coordinate system
        # Then normalize to [0, 1] range for height and convert to pixels
        y_normalized = y_raw * aspect_ratio
        y_pixel = y_normalized * orig_img_height + top_pad
        
        # Convert to grid cell
        grid_cell = pixel_to_grid_cell(x_pixel, y_pixel, padded_width, padded_height, 
                                      grid_size, grid_size)
        
        ground_truth_data.append({
            'image_id': image_id,
            'grid_cell': grid_cell,
            'direction': direction
        })
    
    # Create DataFrame and save
    gt_df = pd.DataFrame(ground_truth_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gt_df.to_csv(output_path, index=False)
    print(f"  Ground truth saved: {os.path.basename(output_path)}")


def process_single_building(building_name, source_base_path, output_base_path, 
                            grid_sizes, line_width=3, label_size=80, margin=100):
    """
    Process a single building and create grids for all grid sizes
    Returns number of successful saves and aspect ratio check flag
    """
    print(f"\n{'='*80}")
    print(f"Processing building: {building_name}")
    print(f"{'='*80}")
    
    # Check if floorplan exists
    floorplan_path = os.path.join(source_base_path, building_name, 
                                  f"{building_name}_floorplan.jpg")
    
    if not os.path.exists(floorplan_path):
        print(f"Skipping {building_name}: No floorplan found at {floorplan_path}")
        return 0, False
    
    # Load coordinates if available
    coordinates_path = os.path.join(source_base_path, building_name, "coordinates.csv")
    coordinates_df = None
    
    if os.path.exists(coordinates_path):
        try:
            coordinates_df = pd.read_csv(coordinates_path)
            print(f"Loaded {len(coordinates_df)} coordinates from coordinates.csv")
        except Exception as e:
            print(f"Warning: Could not load coordinates.csv: {e}")
    else:
        print(f"No coordinates.csv found")
    
    # Load image to get original dimensions
    try:
        image = Image.open(floorplan_path)
        orig_width, orig_height = image.size
        print(f"Original dimensions: {orig_width}x{orig_height}")
        
        # Sanity check: floor plans should typically be wider than tall
        unusual_aspect = orig_width < orig_height
        if unusual_aspect:
            print(f"  WARNING: Unusual aspect ratio - width ({orig_width}) < height ({orig_height})")
    except Exception as e:
        print(f"Error loading floorplan for {building_name}: {e}")
        return 0, False
    
    successful_saves = 0
    
    # Process for each grid size
    for grid_size in grid_sizes:
        print(f"\nCreating {grid_size}x{grid_size} grid")
        
        # Define output paths
        grid_folder = f"grid_size_{grid_size}"
        ground_truth_folder = f"grid_size_{grid_size}_ground_truth"
        
        gridded_output_dir = os.path.join(output_base_path, grid_folder)
        ground_truth_output_dir = os.path.join(output_base_path, ground_truth_folder)
        
        gridded_output_path = os.path.join(gridded_output_dir, 
                                          f"{building_name}_floorplan_gridded.jpg")
        ground_truth_path = os.path.join(ground_truth_output_dir, 
                                        f"{building_name}_ground_truth.csv")
        
        try:
            # Create grid with padding
            _, left_pad, top_pad, padded_width, padded_height = create_grid_overlay_with_padding(
                floorplan_path,
                gridded_output_path,
                coordinates_df,
                grid_cols=grid_size,
                grid_rows=grid_size,
                line_width=line_width,
                label_size=label_size,
                margin=margin
            )
            successful_saves += 1
            
            # Generate ground truth if coordinates exist
            if coordinates_df is not None:
                generate_ground_truth(
                    coordinates_df,
                    orig_width,
                    orig_height,
                    left_pad,
                    top_pad,
                    padded_width,
                    padded_height,
                    grid_size,
                    ground_truth_path
                )
            
        except Exception as e:
            print(f"Error creating grid for {building_name} with size {grid_size}: {e}")
    
    return successful_saves, unusual_aspect


def process_all_buildings(source_base_path, output_base_path, grid_sizes, 
                         line_width=3, label_size=80, margin=100):
    """
    Process all buildings in the source directory
    Returns total number of successful saves and buildings processed
    """
    print(f"\n{'='*80}")
    print(f"Processing all buildings from: {source_base_path}")
    print(f"Output directory: {output_base_path}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"{'='*80}")
    
    # Get all subdirectories
    building_folders = [f for f in os.listdir(source_base_path) 
                       if os.path.isdir(os.path.join(source_base_path, f))]
    
    print(f"\nFound {len(building_folders)} building folders")
    
    total_successful_saves = 0
    buildings_with_floorplans = 0
    buildings_without_floorplans = 0
    buildings_with_unusual_aspect = []
    
    for building_name in sorted(building_folders):
        saves, unusual_aspect = process_single_building(
            building_name, 
            source_base_path, 
            output_base_path, 
            grid_sizes, 
            line_width, 
            label_size,
            margin
        )
        
        if saves > 0:
            buildings_with_floorplans += 1
            total_successful_saves += saves
            if unusual_aspect:
                buildings_with_unusual_aspect.append(building_name)
        else:
            buildings_without_floorplans += 1
    
    return total_successful_saves, buildings_with_floorplans, buildings_without_floorplans, buildings_with_unusual_aspect


if __name__ == "__main__":
    
    # Configuration - Hyperparameters
    GRID_SIZES = [10, 15, 20]  # List of grid sizes to create
    LINE_WIDTH = 3
    LABEL_SIZE = 80
    MARGIN = 100  # Margin for padding around coordinates
    
    # Configuration - Paths
    SOURCE_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output"
    OUTPUT_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/grids_floorplan"
    
    # Configuration - Mode
    SINGLE_BUILDING_MODE = False  # Set to True to process single building, False for all buildings
    SINGLE_BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"  # Only used if SINGLE_BUILDING_MODE is True
    
    # Process based on mode
    if SINGLE_BUILDING_MODE:
        print("Running in SINGLE BUILDING MODE")
        saves, unusual_aspect = process_single_building(
            SINGLE_BUILDING_NAME,
            SOURCE_BASE_PATH,
            OUTPUT_BASE_PATH,
            GRID_SIZES,
            LINE_WIDTH,
            LABEL_SIZE,
            MARGIN
        )
        print(f"\n{'='*80}")
        print(f"SUMMARY - Single Building Mode")
        print(f"{'='*80}")
        print(f"Building: {SINGLE_BUILDING_NAME}")
        print(f"Total grids created: {saves}")
        print(f"Expected grids: {len(GRID_SIZES)}")
        if unusual_aspect:
            print(f"WARNING: Building has unusual aspect ratio (height > width)")
        
    else:
        print("Running in ALL BUILDINGS MODE")
        total_saves, buildings_with, buildings_without, unusual_buildings = process_all_buildings(
            SOURCE_BASE_PATH,
            OUTPUT_BASE_PATH,
            GRID_SIZES,
            LINE_WIDTH,
            LABEL_SIZE,
            MARGIN
        )
        
        print(f"\n{'='*80}")
        print(f"SUMMARY - All Buildings Mode")
        print(f"{'='*80}")
        print(f"Total grids created: {total_saves}")
        print(f"Buildings with floorplans: {buildings_with}")
        print(f"Buildings without floorplans: {buildings_without}")
        print(f"Total buildings scanned: {buildings_with + buildings_without}")
        
        if unusual_buildings:
            print(f"\nBuildings with unusual aspect ratio (height > width): {len(unusual_buildings)}")
            for building in unusual_buildings:
                print(f"  - {building}")