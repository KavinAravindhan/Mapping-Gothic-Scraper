import json
import os
from PIL import Image, ImageDraw
import sys

def load_building_data(building_name):
    """Load the scraped data for a building"""
    data_path = f"maps_output/{building_name}/data.json"
    image_path = f"maps_output/{building_name}/{building_name}_floorplan.jpg"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return None, None
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return None, None
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    image = Image.open(image_path)
    
    return data, image

def get_arrow_points(direction, center_x, center_y, size=20):
    """
    Get polygon points for an arrow based on direction
    Returns list of (x, y) tuples for the arrow shape
    """
    arrows = {
        'N':  [(0, -size), (-size//2, 0), (size//2, 0)],
        'NE': [(size, -size), (0, -size//2), (0, size//2)],
        'E':  [(size, 0), (0, -size//2), (0, size//2)],
        'SE': [(size, size), (0, 0), (-size//2, 0)],
        'S':  [(0, size), (-size//2, 0), (size//2, 0)],
        'SW': [(-size, size), (0, 0), (0, -size//2)],
        'W':  [(-size, 0), (0, -size//2), (0, size//2)],
        'NW': [(-size, -size), (0, -size//2), (size//2, 0)]
    }
    
    if direction not in arrows:
        direction = 'N'  # Default
    
    # Translate points to center position
    points = [(center_x + dx, center_y + dy) for dx, dy in arrows[direction]]
    return points

def draw_arrow(draw, x, y, direction, color='blue', arrow_length=40, line_width=3):
    """Draw an arrow as a line with arrowhead at the given position"""
    
    # Direction angles (in degrees, where 0 is East, 90 is North)
    angles = {
        'E': 0,
        'NE': 45,
        'N': 90,
        'NW': 135,
        'W': 180,
        'SW': 225,
        'S': 270,
        'SE': 315
    }
    
    if direction not in angles:
        direction = 'N'
    
    import math
    angle = math.radians(angles[direction])
    
    # Calculate arrow end point
    end_x = x + arrow_length * math.cos(angle)
    end_y = y - arrow_length * math.sin(angle)  # Negative because Y increases downward in images
    
    # Draw the main line
    draw.line([(x, y), (end_x, end_y)], fill=color, width=line_width)
    
    # Draw arrowhead (a small triangle at the end)
    arrow_size = 10
    angle1 = angle + math.radians(150)
    angle2 = angle - math.radians(150)
    
    p1 = (end_x + arrow_size * math.cos(angle1), end_y - arrow_size * math.sin(angle1))
    p2 = (end_x + arrow_size * math.cos(angle2), end_y - arrow_size * math.sin(angle2))
    
    draw.polygon([p1, (end_x, end_y), p2], fill=color)

def visualize_floor_plan(building_name, output_path=None, padding=100, y_offset=0):
    """Create a visualization with arrows on the floor plan"""
    
    # Load data
    data, image = load_building_data(building_name)
    if data is None or image is None:
        return
    
    width, height = image.size
    print(f"Floor plan size: {width}x{height}")

    width, height = image.size
    dzi_width = 2454  # From the .dzi file
    dzi_height = 1923  # From the .dzi file
    
    print(f"Stitched image size: {width}x{height}")
    print(f"DZI declared size: {dzi_width}x{dzi_height}")
    
    if width != dzi_width or height != dzi_height:
        print(f"⚠️  SIZE MISMATCH! This will cause positioning errors.")
        print(f"   Width difference: {width - dzi_width}")
        print(f"   Height difference: {height - dzi_height}")
    
    # Add padding to show arrows outside the floor plan
    padded_width = width + 2 * padding
    padded_height = height + 2 * padding
    
    # Create a new image with padding (white background)
    result_image = Image.new('RGB', (padded_width, padded_height), 'white')
    # Paste the floor plan in the center
    result_image.paste(image, (padding, padding))
    
    draw = ImageDraw.Draw(result_image)
    
    print(f"Drawing {len(data['overlays'])} arrows...")
    
    # Draw each arrow
    for overlay in data['overlays']:
        # Convert normalized coordinates (0-1) to pixel coordinates
        # X stays the same (left to right)
        pixel_x = int(overlay['x'] * width) + padding
        
        # Y needs to be flipped (0,0 is bottom-left, not top-left)
        # Y coordinate (0,0 is top-left in OpenSeadragon)
        # pixel_y = int(overlay['y'] * height) + padding
        pixel_y = int(overlay['y'] * height) + padding + y_offset
        
        direction = overlay.get('direction', 'N')
        
        # Draw the arrow
        draw_arrow(draw, pixel_x, pixel_y, direction, color='blue', arrow_length=50, line_width=3)
        
        print(f"  {overlay['image_id']}: ({overlay['x']:.3f}, {overlay['y']:.3f}) -> pixel ({pixel_x}, {pixel_y}) - {direction}")
    
    # Save result
    if output_path is None:
        output_path = f"maps_output/{building_name}/{building_name}_visualization.jpg"
    
    # Draw coordinate labels for debugging
    from PIL import ImageFont
    
    # Draw a few test points to understand the coordinate system
    test_points = [
        (0, 0, "Origin (0,0)"),
        (0.5, 0.5, "Center (0.5,0.5)"),
        (1, 1, "Corner (1,1)"),
    ]
    
    for test_x, test_y, label in test_points:
        px = int(test_x * width) + padding
        py = int(test_y * height) + padding
        
        # Draw a red circle
        draw.ellipse([px-5, py-5, px+5, py+5], fill='red', outline='red')
        # Draw label
        draw.text((px+10, py), label, fill='red')
    
    print("\nDebug test points:")
    print(f"  (0,0) -> pixel ({padding}, {padding})")
    print(f"  (0.5,0.5) -> pixel ({width//2 + padding}, {height//2 + padding})")
    print(f"  (1,1) -> pixel ({width + padding}, {height + padding})")  
    
    result_image.save(output_path, 'JPEG', quality=95)
    print(f"\n✓ Saved visualization to: {output_path}")
    
    return result_image

if __name__ == "__main__":
    # Example usage
    building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    
    # You can also pass a building name as command line argument
    if len(sys.argv) > 1:
        building_name = sys.argv[1]
    
    print(f"Visualizing: {building_name}\n")
    # visualize_floor_plan(building_name)
    visualize_floor_plan(building_name, y_offset=+150)  # Try -50, -30, -70, etc.

# python3 visualize_floor_plan.py
# python3 visualize_floor_plan.py Albi-Cathedrale-Sainte-Cecile