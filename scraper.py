# Standard library
import csv
import json
import os
import re
import time
from urllib.parse import urljoin

# Third-party
import requests
from bs4 import BeautifulSoup

# Get List of All Map Directories
def get_all_map_directories():
    """Scrape the main page to get all building directory URLs"""
    base_url = "https://mcid.mcah.columbia.edu/media/plotted-images/maps/"
    
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all directory links (they end with /)
    directories = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('/') and href != '../':
            full_url = base_url + href
            directories.append({
                'name': href.rstrip('/'),
                'url': full_url
            })
    
    return directories

# Extract Data from a Single Building's index.html
def extract_building_data(building_url):
    """
    Extract floor plan image, arrow coordinates, and image IDs from a building's page
    Returns a dictionary with all the data
    """
    response = requests.get(building_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the script tag containing the OpenSeadragon viewer configuration
    script_tags = soup.find_all('script')
    
    data = {
        'building_url': building_url,
        'building_name': building_url.rstrip('/').split('/')[-1],
        'floor_plan_image': None,
        'overlays': []
    }
    
    for script in script_tags:
        if script.string and 'OpenSeadragon' in script.string:
            script_content = script.string

            # TEMPORARY DEBUG - REMOVE LATER
            # print("=== FOUND OPENSEADRAGON SCRIPT ===")
            # print(script_content[:2000])  # Print first 2000 characters
            
            # Extract tileSources (floor plan image)
            # tile_match = re.search(r'tileSources:\s*["\']([^"\']+)["\']', script_content)
            tile_match = re.search(r'tileSources:\s*\["([^"]+)"\]', script_content)
            if tile_match:
                data['floor_plan_image'] = tile_match.group(1)
            
            # Extract overlays (arrow locations and image IDs)
            # Look for patterns like: x: 0.123, y: 0.456, id: 'image_id'
            overlay_pattern = r'\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*id:\s*["\']([^"\']+)["\']\}'
            overlays = re.findall(overlay_pattern, script_content)
            
            for x, y, image_id in overlays:
                data['overlays'].append({
                    'image_id': image_id,
                    'x': float(x),
                    'y': float(y)
                })
    
    return data

# Download Floor Plan Image
def download_floor_plan(data, output_dir='output'):
    """Download the floor plan image"""
    os.makedirs(output_dir, exist_ok=True)
    
    building_name = data['building_name']
    floor_plan_url = data['floor_plan_image']
    
    # Make absolute URL if relative
    if not floor_plan_url.startswith('http'):
        floor_plan_url = urljoin(data['building_url'], floor_plan_url)
    
    # Determine file extension
    ext = os.path.splitext(floor_plan_url)[1] or '.jpg'
    filename = f"{building_name}_floorplan{ext}"
    filepath = os.path.join(output_dir, filename)
    
    print(f"Downloading floor plan: {floor_plan_url}")
    response = requests.get(floor_plan_url)
    
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    print(f"Saved to: {filepath}")
    return filepath

# Download All Arrow Images (Full Resolution)
def download_arrow_images(data, output_dir='output'):
    """Download all images associated with arrows"""
    os.makedirs(output_dir, exist_ok=True)
    
    building_name = data['building_name']
    base_url = data['building_url']
    
    downloaded_images = []
    
    for overlay in data['overlays']:
        image_id = overlay['image_id']
        
        # Construct image URL - typically in an 'images' subdirectory
        # The full-size image usually has the same ID as the thumbnail
        # image_url = urljoin(base_url, f"images/{image_id}.jpg")
        image_url = urljoin(base_url, f"imgs/{image_id}.jpg")
        
        filename = f"{image_id}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        try:
            print(f"Downloading: {image_id}")
            response = requests.get(image_url)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_images.append({
                    'image_id': image_id,
                    'filepath': filepath,
                    'x': overlay['x'],
                    'y': overlay['y']
                })
                print(f"  ✓ Saved: {filepath}")
            else:
                print(f"  ✗ Failed: {response.status_code}")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return downloaded_images

# Save Coordinates to JSON/CSV
def save_coordinates_json(data, output_path):
    """Save all data including coordinates to JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved coordinates to: {output_path}")

def save_coordinates_csv(data, output_path):
    """Save arrow coordinates to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'x', 'y'])
        
        for overlay in data['overlays']:
            writer.writerow([
                overlay['image_id'],
                overlay['x'],
                overlay['y']
            ])
    
    print(f"Saved coordinates to: {output_path}")

# Complete Pipeline (Putting it all together)
def scrape_single_building(building_url, output_base_dir='maps_output'):
    """Complete pipeline for a single building"""
    print(f"\n{'='*60}")
    print(f"Processing: {building_url}")
    print(f"{'='*60}")
    
    # Extract data
    data = extract_building_data(building_url)
    building_name = data['building_name']
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, building_name)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Download floor plan
    download_floor_plan(data, output_dir)
    
    # Download all arrow images
    download_arrow_images(data, images_dir)
    
    # Save coordinates
    save_coordinates_json(data, os.path.join(output_dir, 'data.json'))
    save_coordinates_csv(data, os.path.join(output_dir, 'coordinates.csv'))
    
    print(f"\n✓ Completed: {building_name}")
    print(f"  - Floor plan: {output_dir}")
    print(f"  - {len(data['overlays'])} images downloaded")
    
    return data

def scrape_all_buildings():
    """Scrape all buildings from the site"""
    directories = get_all_map_directories()
    print(f"Found {len(directories)} buildings to process\n")
    
    results = []
    for i, directory in enumerate(directories, 1):
        print(f"\n[{i}/{len(directories)}]")
        try:
            result = scrape_single_building(directory['url'])
            results.append(result)
            time.sleep(1)  # Be polite, wait 1 second between requests
        except Exception as e:
            print(f"✗ Error processing {directory['name']}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Processed {len(results)} buildings")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    # Test with ONE building first:
    # scrape_single_building("https://mcid.mcah.columbia.edu/media/plotted-images/maps/Beaumont-sur-Oise-Eglise-Saint-Leonor/")
    scrape_single_building("https://mcid.mcah.columbia.edu/media/plotted-images/maps/Albi-Cathedrale-Sainte-Cecile/")
    
    # Once that works, comment out the line above and uncomment this to do all:
    # scrape_all_buildings()