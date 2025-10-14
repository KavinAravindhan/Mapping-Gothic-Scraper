import os
import subprocess

def convert_dzi_to_jpg(dzi_file_path):
    """Convert .dzi file to .jpg using vips"""
    output_path = dzi_file_path.replace('.dzi', '.jpg')
    
    try:
        # Using vips/libvips command line tool
        subprocess.run([
            'vips', 
            'dzsave', 
            dzi_file_path, 
            output_path,
            '--tile-size', '254'
        ], check=True)
        print(f"✓ Converted: {output_path}")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")

# Usage: convert all .dzi files in maps_output
for root, dirs, files in os.walk('maps_output'):
    for file in files:
        if file.endswith('.dzi'):
            dzi_path = os.path.join(root, file)
            convert_dzi_to_jpg(dzi_path)