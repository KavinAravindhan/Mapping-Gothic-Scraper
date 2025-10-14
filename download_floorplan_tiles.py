# Standard library imports
import math
import os
from io import BytesIO
from urllib.parse import urljoin

# Third-party imports
from PIL import Image
import requests


def download_and_stitch_tiles(dzi_url, output_path):
    """Download all tiles from a .dzi Deep Zoom Image and stitch them together"""
    
    # Get the base URL (remove .dzi extension)
    base_url = dzi_url.replace('.dzi', '_files/')
    
    # Parse the .dzi file to get dimensions
    response = requests.get(dzi_url)
    if response.status_code != 200:
        print(f"Failed to download .dzi file")
        return False
    
    # Extract dimensions from XML (simple parsing)
    content = response.text
    height = int(content.split('Height="')[1].split('"')[0])
    width = int(content.split('Width="')[1].split('"')[0])
    tile_size = int(content.split('TileSize="')[1].split('"')[0])
    
    print(f"Image size: {width}x{height}")
    print(f"Tile size: {tile_size}")
    
    # Create output image
    final_image = Image.new('RGB', (width, height))
    
    # Calculate number of tiles
    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    
    # Deep Zoom images have multiple zoom levels
    # We want the highest zoom level (last one)
    max_level = math.ceil(math.log(max(width, height), 2))
    
    print(f"Downloading tiles from level {max_level}...")
    print(f"Grid: {cols}x{rows} tiles")
    
    # Download and place each tile
    success_count = 0
    for row in range(rows):
        for col in range(cols):
            tile_url = f"{base_url}{max_level}/{col}_{row}.jpeg"
            
            try:
                tile_response = requests.get(tile_url)
                if tile_response.status_code == 200:
                    # Open tile image
                    tile_image = Image.open(BytesIO(tile_response.content))
                    
                    # Calculate position
                    x = col * tile_size
                    y = row * tile_size
                    
                    # Paste tile
                    final_image.paste(tile_image, (x, y))
                    success_count += 1
                    print(f"✓ Downloaded tile {col}_{row} ({success_count}/{cols*rows})", end='\r')
                else:
                    print(f"\n✗ Failed to download tile {col}_{row}: {tile_response.status_code}")
            except Exception as e:
                print(f"\n✗ Error downloading tile {col}_{row}: {str(e)}")
    
    print(f"\n\nSuccessfully downloaded {success_count}/{cols*rows} tiles")
    
    # Save final image
    final_image.save(output_path, 'JPEG', quality=95)
    print(f"✓ Saved floor plan to: {output_path}")
    return True

# # Usage
# from io import BytesIO

# dzi_url = "https://mcid.mcah.columbia.edu/media/images/other/1065_00042/1065_00042.dzi"
# output_path = "maps_output/Beaumont-sur-Oise-Eglise-Saint-Leonor/floorplan.jpg"

# download_and_stitch_tiles(dzi_url, output_path)