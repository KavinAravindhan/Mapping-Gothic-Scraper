# Standard library imports
import os
import string
from pathlib import Path

# Third-party imports
from PIL import Image, ImageDraw, ImageFont


def create_grid_overlay(floor_plan_path, output_path, grid_cols=10, grid_rows=8, 
                        line_color=(255, 0, 0), line_width=2, label_size=80):
    """
    Create a grid overlay on a floor plan image with letter columns (A, B, C...) 
    and number rows (1, 2, 3...)
    
    Args:
        floor_plan_path: Path to the input floor plan image
        output_path: Path where the gridded image will be saved
        grid_cols: Number of vertical grid lines (columns)
        grid_rows: Number of horizontal grid lines (rows)
        line_color: RGB tuple for grid line color
        line_width: Width of grid lines in pixels
        label_size: Font size for grid labels
    """
    
    # Load the floor plan image
    print(f"Loading floor plan: {floor_plan_path}")
    image = Image.open(floor_plan_path)
    original_width, original_height = image.size
    print(f"Original image size: {original_width}x{original_height}")
    
    # Create padding for labels - scale with label size
    padding = max(100, label_size + 40)
    
    # Create new image with padding
    new_width = original_width + padding * 2
    new_height = original_height + padding * 2
    padded_image = Image.new('RGB', (new_width, new_height), 'white')
    
    # Paste original image in the center
    padded_image.paste(image, (padding, padding))
    
    # Create drawing context
    draw = ImageDraw.Draw(padded_image)
    
    # Try multiple font options
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\ariblk.ttf",  # Windows
        "arial.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, label_size)
            print(f"✓ Loaded font: {font_path} at size {label_size}")
            break
        except:
            continue
    
    # If no TrueType font found, create a workaround
    if font is None:
        print(f"⚠ Warning: Could not load TrueType font. Using alternative method.")
        # We'll use a larger default and draw multiple times for thickness
        font = ImageFont.load_default()
    
    # Calculate grid spacing
    col_spacing = original_width / grid_cols
    row_spacing = original_height / grid_rows
    
    print(f"Grid: {grid_cols} columns x {grid_rows} rows")
    print(f"Cell size: {col_spacing:.1f}px x {row_spacing:.1f}px")
    print(f"Label size: {label_size}, Padding: {padding}")
    
    # Draw vertical lines and column labels (A, B, C...)
    letters = string.ascii_uppercase
    for i in range(grid_cols + 1):
        x = padding + i * col_spacing
        
        # Draw vertical line
        draw.line([(x, padding), (x, padding + original_height)], 
                 fill=line_color, width=line_width)
        
        # Add letter labels at top and bottom
        if i < grid_cols:
            letter = letters[i] if i < len(letters) else f"{letters[i // 26 - 1]}{letters[i % 26]}"
            
            # Calculate center position for the label
            label_x = x + col_spacing / 2
            
            if font is not None and hasattr(font, 'getbbox'):
                # Modern PIL version
                bbox = font.getbbox(letter)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Fallback for older versions or default font
                try:
                    bbox = draw.textbbox((0, 0), letter, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    # Rough estimate
                    text_width = label_size * 0.6
                    text_height = label_size
            
            # Top label
            top_y = padding/2 - text_height/2
            draw.text((label_x - text_width/2, top_y), 
                     letter, fill='black', font=font)
            
            # If using default font, draw multiple times for visibility
            if font == ImageFont.load_default():
                for offset_x in range(-2, 3):
                    for offset_y in range(-2, 3):
                        draw.text((label_x - text_width/2 + offset_x, top_y + offset_y), 
                                letter, fill='black', font=font)
            
            # Bottom label
            bottom_y = padding + original_height + padding/2 - text_height/2
            draw.text((label_x - text_width/2, bottom_y), 
                     letter, fill='black', font=font)
            
            # If using default font, draw multiple times for visibility
            if font == ImageFont.load_default():
                for offset_x in range(-2, 3):
                    for offset_y in range(-2, 3):
                        draw.text((label_x - text_width/2 + offset_x, bottom_y + offset_y), 
                                letter, fill='black', font=font)
    
    # Draw horizontal lines and row labels (1, 2, 3...)
    for i in range(grid_rows + 1):
        y = padding + i * row_spacing
        
        # Draw horizontal line
        draw.line([(padding, y), (padding + original_width, y)], 
                 fill=line_color, width=line_width)
        
        # Add number labels at left and right
        if i < grid_rows:
            number = str(i + 1)
            
            # Calculate center position for the label
            label_y = y + row_spacing / 2
            
            if font is not None and hasattr(font, 'getbbox'):
                bbox = font.getbbox(number)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                try:
                    bbox = draw.textbbox((0, 0), number, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    text_width = label_size * 0.6
                    text_height = label_size
            
            # Left label
            left_x = padding/2 - text_width/2
            draw.text((left_x, label_y - text_height/2), 
                     number, fill='black', font=font)
            
            # If using default font, draw multiple times for visibility
            if font == ImageFont.load_default():
                for offset_x in range(-2, 3):
                    for offset_y in range(-2, 3):
                        draw.text((left_x + offset_x, label_y - text_height/2 + offset_y), 
                                number, fill='black', font=font)
            
            # Right label
            right_x = padding + original_width + padding/2 - text_width/2
            draw.text((right_x, label_y - text_height/2), 
                     number, fill='black', font=font)
            
            # If using default font, draw multiple times for visibility
            if font == ImageFont.load_default():
                for offset_x in range(-2, 3):
                    for offset_y in range(-2, 3):
                        draw.text((right_x + offset_x, label_y - text_height/2 + offset_y), 
                                number, fill='black', font=font)
    
    # Save the result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    padded_image.save(output_path, 'JPEG', quality=95)
    print(f"✓ Saved gridded floor plan to: {output_path}")
    
    return output_path


def process_all_floorplans(base_dir='maps_output', grid_cols=10, grid_rows=8, label_size=80):
    """
    Process all floor plans in the maps_output directory
    
    Args:
        base_dir: Base directory containing building folders
        grid_cols: Number of columns in the grid
        grid_rows: Number of rows in the grid
        label_size: Size of grid labels
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} not found")
        return
    
    processed_count = 0
    
    # Iterate through all building directories
    for building_dir in base_path.iterdir():
        if building_dir.is_dir():
            # Look for the floorplan file
            floorplan_path = building_dir / f"{building_dir.name}_floorplan.jpg"
            
            if floorplan_path.exists():
                print(f"\n{'='*60}")
                print(f"Processing: {building_dir.name}")
                
                # Output path for gridded version
                output_path = building_dir / f"{building_dir.name}_floorplan_gridded.jpg"
                
                try:
                    create_grid_overlay(
                        str(floorplan_path),
                        str(output_path),
                        grid_cols=grid_cols,
                        grid_rows=grid_rows,
                        label_size=label_size
                    )
                    processed_count += 1
                except Exception as e:
                    print(f"✗ Error processing {building_dir.name}: {str(e)}")
            else:
                print(f"✗ Floorplan not found for {building_dir.name}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Processed {processed_count} floor plans")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example 1: Process a single floor plan with LARGE labels
    create_grid_overlay(
        "maps_output/Beaumont-sur-Oise-Eglise-Saint-Leonor/Beaumont-sur-Oise-Eglise-Saint-Leonor_floorplan.jpg",
        "maps_output/Beaumont-sur-Oise-Eglise-Saint-Leonor/Beaumont-sur-Oise-Eglise-Saint-Leonor_floorplan_gridded.jpg",
        grid_cols=10,
        grid_rows=8,
        line_width=3,
        label_size=80  # Large labels!
    )
    
    # Example 2: Process all floor plans in the maps_output directory
    # process_all_floorplans(base_dir='maps_output', grid_cols=10, grid_rows=8, label_size=80)