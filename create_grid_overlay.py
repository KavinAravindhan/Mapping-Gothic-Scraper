# Standard library imports
import os
from pathlib import Path

# Third-party imports
from PIL import Image, ImageDraw, ImageFont


def load_font(label_size=80):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\ariblk.ttf",
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


def create_grid_overlay_consistent(image_path, output_path, reference_size,
                                   grid_cols=10, grid_rows=10,
                                   line_color=(255, 0, 0), line_width=3,
                                   label_size=80):

    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size
    print(f"  Image size: {img_width}x{img_height}")
    print(f"  Reference size: {reference_size[0]}x{reference_size[1]}")
    
    # Calculate grid spacing based on reference dimensions
    ref_width, ref_height = reference_size
    col_spacing = ref_width / grid_cols
    row_spacing = ref_height / grid_rows
    
    # Calculate scaling factor if image differs from reference
    scale_x = img_width / ref_width
    scale_y = img_height / ref_height
    
    # Adjust spacing for this image
    actual_col_spacing = col_spacing * scale_x
    actual_row_spacing = row_spacing * scale_y
    
    print(f"  Grid spacing: {actual_col_spacing:.1f}px x {actual_row_spacing:.1f}px")
    
    # Create padding for labels
    padding = max(120, label_size + 40)
    
    # Create new image with padding
    new_width = img_width + padding * 2
    new_height = img_height + padding * 2
    padded_image = Image.new('RGB', (new_width, new_height), 'white')
    padded_image.paste(image, (padding, padding))
    
    # Create drawing context and load font
    draw = ImageDraw.Draw(padded_image)
    font = load_font(label_size)
    
    # Draw vertical lines and column labels (A-J)
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for i in range(grid_cols + 1):
        x = padding + i * actual_col_spacing
        
        # Draw vertical line
        draw.line([(x, padding), (x, padding + img_height)],
                 fill=line_color, width=line_width)
        
        # Add column labels
        if i < grid_cols:
            letter = columns[i]
            label_x = x + actual_col_spacing / 2
            text_width, text_height = get_text_dimensions(draw, letter, font)
            
            # Top label
            draw.text((label_x - text_width/2, padding/2 - text_height/2),
                     letter, fill='black', font=font)
            
            # Bottom label
            draw.text((label_x - text_width/2, padding + img_height + padding/2 - text_height/2),
                     letter, fill='black', font=font)
    
    # Draw horizontal lines and row labels (1-10)
    for i in range(grid_rows + 1):
        y = padding + i * actual_row_spacing
        
        # Draw horizontal line
        draw.line([(padding, y), (padding + img_width, y)],
                 fill=line_color, width=line_width)
        
        # Add row labels
        if i < grid_rows:
            number = str(i + 1)
            label_y = y + actual_row_spacing / 2
            text_width, text_height = get_text_dimensions(draw, number, font)
            
            # Left label
            draw.text((padding/2 - text_width/2, label_y - text_height/2),
                     number, fill='black', font=font)
            
            # Right label
            draw.text((padding + img_width + padding/2 - text_width/2, label_y - text_height/2),
                     number, fill='black', font=font)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    padded_image.save(output_path, 'JPEG', quality=95)
    print(f"Saved: {os.path.basename(output_path)}")
    
    return output_path


def process_floor_plans_with_consistent_grid(base_floorplan_path, arrows_floorplan_path,
                                             base_output_path, arrows_output_path,
                                             grid_cols=10, grid_rows=10,
                                             line_width=3, label_size=80):

    print("Processing Floor Plans With Consistent Grid")

    # Load base floor plan to get reference dimensions
    base_image = Image.open(base_floorplan_path)
    reference_size = base_image.size
    print(f"\nUsing reference dimensions: {reference_size[0]}x{reference_size[1]}")
    print(f"Grid configuration: {grid_cols} columns x {grid_rows} rows")
    
    # Process base floor plan
    create_grid_overlay_consistent(
        base_floorplan_path,
        base_output_path,
        reference_size,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        line_width=line_width,
        label_size=label_size
    )
    
    # Process arrows floor plan with same reference
    create_grid_overlay_consistent(
        arrows_floorplan_path,
        arrows_output_path,
        reference_size,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        line_width=line_width,
        label_size=label_size
    )
    
    print("Both floor plans processed with consistent 10x10 grid")


if __name__ == "__main__":

    building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    
    # Define paths using building name
    base_dir = f"maps_output/{building_name}"
    
    base_floorplan = f"{base_dir}/{building_name}_floorplan.jpg"
    arrows_floorplan = f"{base_dir}/{building_name}_arrows_visualization.jpg"
    
    base_output = f"{base_dir}/{building_name}_floorplan_gridded.jpg"
    arrows_output = f"{base_dir}/{building_name}_floorplan_arrows_gridded.jpg"
    
    # Process both with consistent 10x10 grid
    process_floor_plans_with_consistent_grid(
        base_floorplan_path=base_floorplan,
        arrows_floorplan_path=arrows_floorplan,
        base_output_path=base_output,
        arrows_output_path=arrows_output,
        grid_cols=10,
        grid_rows=10,
        line_width=3,
        label_size=80
    )