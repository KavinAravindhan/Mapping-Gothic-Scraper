import os
from pathlib import Path

def count_images_in_buildings(base_path):
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_path} does not exist")
        return
    
    # Dictionary to store results
    building_counts = {}
    total_images = 0
    
    # Iterate through all subdirectories (building folders)
    for building_folder in sorted(base_dir.iterdir()):
        if building_folder.is_dir():
            images_folder = building_folder / 'images'
            
            if images_folder.exists() and images_folder.is_dir():
                # Count image files
                image_count = sum(
                    1 for file in images_folder.iterdir()
                    if file.is_file() and file.suffix.lower() in image_extensions
                )
                
                building_counts[building_folder.name] = image_count
                total_images += image_count
            else:
                building_counts[building_folder.name] = 0
    
    # Print results
    print("=" * 70)
    print(f"Image Count Report for: {base_path}")
    print("=" * 70)
    print(f"{'Building Folder':<50} {'Image Count':>15}")
    print("-" * 70)
    
    for building, count in building_counts.items():
        print(f"{building:<50} {count:>15,}")
    
    print("-" * 70)
    print(f"{'TOTAL':<50} {total_images:>15,}")
    print("=" * 70)
    
    return building_counts, total_images

if __name__ == "__main__":
    base_path = "/mnt/swordfish-pool2/kavin/maps_output"
    building_counts, total = count_images_in_buildings(base_path)