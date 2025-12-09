import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datasets import Dataset, Features, Value, Image
import requests
from io import BytesIO
from PIL import Image as PILImage
import time

def download_image(url: str, max_retries: int = 3) -> Optional[PILImage.Image]:
    """Download image from URL with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = PILImage.open(BytesIO(response.content))
            return img.convert('RGB')
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

def parse_label_studio_annotations(json_data: List[Dict[str, Any]]) -> Dataset:
    
    rows = []
    
    for task in json_data:
        # Extract building-level metadata
        building_id = task['id']
        building_name = task['data'].get('building_name', '')
        building_name_pretty = task['data'].get('building_name_pretty', '')
        style = task['data'].get('style', '')
        style_pretty = task['data'].get('style_pretty', '')
        building_images = task['data'].get('building_images', [])
        building_image_urls = task['data'].get('building_image_urls', [])
        
        # Check for building-level caption/description
        building_caption = task['data'].get('building_description', None)
        building_notes = task['data'].get('notes', None)
        
        # Process each annotation
        for annotation in task['annotations']:

            annotation_id = annotation['id']
            annotator_email = annotation['completed_by'].get('email', '')
            created_at = annotation.get('created_at', '')
            updated_at = annotation.get('updated_at', '')
            
            # Group results by image index
            image_annotations = {}
            # c = 0
            for result in annotation['result']:
                # c += 1
                # if c >= 2:
                #     break
                # Skip if missing required keys
                if 'to_name' not in result or 'from_name' not in result:
                    continue
                    
                to_name = result['to_name']

                # Extract image index from 'image_0', 'image_1', etc.
                try:
                    image_idx = int(to_name.split('_')[-1])
                except (ValueError, IndexError):
                    continue
                
                # Initialize image entry if not exists
                if image_idx not in image_annotations:
                    image_annotations[image_idx] = {
                        'has_text': None,
                        'unique': None,
                        'image_caption': None,
                        'bboxes': {}
                    }
                
                from_name = result['from_name']
                result_type = result.get('type', '')
                value = result.get('value', {})
                
                # Parse image-level annotations
                if from_name.startswith('hastext_'):
                    choices = value.get('choices', [])
                    if choices:
                        image_annotations[image_idx]['has_text'] = choices[0]
                    
                elif from_name.startswith('unique_'):
                    choices = value.get('choices', [])
                    if choices:
                        image_annotations[image_idx]['unique'] = choices[0]
                    
                elif from_name.startswith('caption_') and result_type == 'textarea':
                    # Image-level caption (not bbox-specific)
                    text = value.get('text', [])
                    if text:
                        image_annotations[image_idx]['image_caption'] = ' | '.join(text)
                
                # Parse bounding box annotations
                elif from_name.startswith('bbox_'):
                    bbox_id = result.get('id')
                    if not bbox_id:
                        continue
                    
                    # Initialize bbox entry if not exists
                    if bbox_id not in image_annotations[image_idx]['bboxes']:
                        image_annotations[image_idx]['bboxes'][bbox_id] = {
                            'x': None,
                            'y': None,
                            'width': None,
                            'height': None,
                            'rotation': None,
                            'category': None,
                            'subcategory': None,
                            'repeated': None,
                            'caption': None,
                            'original_width': None,
                            'original_height': None
                        }
                    
                    # Extract rectangle coordinates and primary category
                    if result_type == 'rectanglelabels':
                        rectanglelabels = value.get('rectanglelabels', [])
                        image_annotations[image_idx]['bboxes'][bbox_id].update({
                            'x': value.get('x'),
                            'y': value.get('y'),
                            'width': value.get('width'),
                            'height': value.get('height'),
                            'rotation': value.get('rotation', 0),
                            'category': rectanglelabels[0] if rectanglelabels else None,
                            'original_width': result.get('original_width'),
                            'original_height': result.get('original_height')
                        })
                    
                    # Extract subcategory
                    elif result_type == 'choices' and 'subclass' in from_name:
                        choices = value.get('choices', [])
                        if choices:
                            image_annotations[image_idx]['bboxes'][bbox_id]['subcategory'] = choices[0]
                    
                    # Extract repeated status (Y/N)
                    elif result_type == 'choices' and 'repeated' in from_name:
                        choices = value.get('choices', [])
                        if choices:
                            image_annotations[image_idx]['bboxes'][bbox_id]['repeated'] = choices[0]
                    
                    # Extract bbox-specific caption/description
                    elif result_type == 'textarea' and 'caption' in from_name:
                        text = value.get('text', [])
                        if text:
                            image_annotations[image_idx]['bboxes'][bbox_id]['caption'] = ' | '.join(text)
            
            # Create building-level row if building caption exists
            if building_caption or building_notes:
                row = {
                    # Data type identifier
                    'data_type': 'building',
                    
                    # Building metadata
                    'building_id': building_id,
                    'building_name': building_name,
                    'building_name_pretty': building_name_pretty,
                    'architectural_style': style,
                    'architectural_style_pretty': style_pretty,
                    'building_caption': building_caption or building_notes,
                    
                    # Annotation metadata
                    'annotation_id': annotation_id,
                    'annotator_email': annotator_email,
                    'annotation_created_at': created_at,
                    'annotation_updated_at': updated_at,
                    
                    # Image metadata (all None for building-level)
                    'image_index': None,
                    'image': None,
                    'image_filename': None,
                    'image_has_text': None,
                    'image_unique_for_style': None,
                    'image_description': None,
                    
                    # Bounding box metadata (all None for building-level)
                    'bbox_id': None,
                    'bbox_x_percent': None,
                    'bbox_y_percent': None,
                    'bbox_width_percent': None,
                    'bbox_height_percent': None,
                    'bbox_rotation_degrees': None,
                    'bbox_category': None,
                    'bbox_subcategory': None,
                    'bbox_is_repeated': None,
                    'bbox_description': None,
                    'original_image_width': None,
                    'original_image_height': None
                }
                rows.append(row)
            
            # Create dataset rows for images and bboxes

            for image_idx, image_data in image_annotations.items():

                # Get image URL and filename
                image_url = building_image_urls[image_idx] if image_idx < len(building_image_urls) else None
                image_filename = building_images[image_idx] if image_idx < len(building_images) else None
                
                # Download the image
                print(f"Downloading image {image_idx} for building {building_id}...")
                image = download_image(image_url) if image_url else None
                
                # Create a row for each bounding box
                if image_data['bboxes']:
                    for bbox_id, bbox_info in image_data['bboxes'].items():
                        row = {
                            # Data type identifier
                            'data_type': 'bbox',
                            
                            # Building metadata
                            'building_id': building_id,
                            'building_name': building_name,
                            'building_name_pretty': building_name_pretty,
                            'architectural_style': style,
                            'architectural_style_pretty': style_pretty,
                            'building_caption': None,
                            
                            # Annotation metadata
                            'annotation_id': annotation_id,
                            'annotator_email': annotator_email,
                            'annotation_created_at': created_at,
                            'annotation_updated_at': updated_at,
                            
                            # Image metadata
                            'image_index': image_idx,
                            'image': image,
                            'image_filename': image_filename,
                            'image_has_text': image_data['has_text'],
                            'image_unique_for_style': image_data['unique'],
                            'image_description': image_data['image_caption'],
                            
                            # Bounding box metadata
                            'bbox_id': bbox_id,
                            'bbox_x_percent': bbox_info['x'],
                            'bbox_y_percent': bbox_info['y'],
                            'bbox_width_percent': bbox_info['width'],
                            'bbox_height_percent': bbox_info['height'],
                            'bbox_rotation_degrees': bbox_info['rotation'],
                            'bbox_category': bbox_info['category'],
                            'bbox_subcategory': bbox_info['subcategory'],
                            'bbox_is_repeated': bbox_info['repeated'],
                            'bbox_description': bbox_info['caption'],
                            'original_image_width': bbox_info['original_width'],
                            'original_image_height': bbox_info['original_height']
                        }
                        rows.append(row)
                else:
                    # Create image-level row (no bboxes)
                    row = {
                        # Data type identifier
                        'data_type': 'image',
                        
                        # Building metadata
                        'building_id': building_id,
                        'building_name': building_name,
                        'building_name_pretty': building_name_pretty,
                        'architectural_style': style,
                        'architectural_style_pretty': style_pretty,
                        'building_caption': None,
                        
                        # Annotation metadata
                        'annotation_id': annotation_id,
                        'annotator_email': annotator_email,
                        'annotation_created_at': created_at,
                        'annotation_updated_at': updated_at,
                        
                        # Image metadata
                        'image_index': image_idx,
                        'image': image,
                        'image_filename': image_filename,
                        'image_has_text': image_data['has_text'],
                        'image_unique_for_style': image_data['unique'],
                        'image_description': image_data['image_caption'],
                        
                        # Bounding box metadata (all None for image-level)
                        'bbox_id': None,
                        'bbox_x_percent': None,
                        'bbox_y_percent': None,
                        'bbox_width_percent': None,
                        'bbox_height_percent': None,
                        'bbox_rotation_degrees': None,
                        'bbox_category': None,
                        'bbox_subcategory': None,
                        'bbox_is_repeated': None,
                        'bbox_description': None,
                        'original_image_width': None,
                        'original_image_height': None
                    }
                    rows.append(row)
    
    print(f"\nCreating dataset from {len(rows)} rows...")
    
    # Create dataset directly from dict instead of pandas
    # This avoids the PyArrow conversion issue
    dataset = Dataset.from_dict({
        key: [row[key] for row in rows] 
        for key in rows[0].keys()
    })
    
    print("Dataset created, now casting image column...")
    
    # Cast the image column to Image type
    dataset = dataset.cast_column("image", Image())
    
    print("Image column cast complete!")
    
    return dataset


def main():    
    # input_file = '/home/kr3131/Mapping-Gothic-Scraper/benchmarking/1_export_197277_project-197277-at-2025-11-17-18-59-c7e831d2.json'
    input_file = '/home/kr3131/Mapping-Gothic-Scraper/benchmarking/2_export_197277_project-197277-at-2025-12-08-13-27-56f90eae.json'
    
    print(f"Loading Label Studio annotations from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both single object and list formats
    if isinstance(json_data, dict):
        json_data = [json_data]
    
    print(f"Processing {len(json_data)} tasks...")
    
    dataset = parse_label_studio_annotations(json_data)
    
    # Display dataset information
    print(f"\nDataset created successfully!")
    print(f"  Total rows: {len(dataset)}")
    print(f"  Features ({len(dataset.features)}):")
    for feature_name in dataset.features:
        print(f"    - {feature_name}")
    
    # Convert to pandas for analysis
    print("\nConverting to pandas for analysis...")
    df = pd.DataFrame(dataset[:])
    
    # Show data type distribution
    print(f"\n  Data Type Distribution:")
    print(df['data_type'].value_counts())
    
    # Save dataset locally first
    # output_dir = 'benchmarking/v5/architecture_annotations_hf_dataset'
    output_dir = '/mnt/swordfish-pool2/kavin/benchmarking/v5/architecture_annotations_to_hf'
    print(f"\nSaving dataset to '{output_dir}'...")
    dataset.save_to_disk(output_dir)
    print(f"Dataset saved!")
    
    # Export to CSV (without images)
    csv_file = 'benchmarking/v5/architecture_annotations.csv'
    print(f"\nExporting to CSV '{csv_file}'...")
    df_no_images = df.drop(columns=['image'])
    df_no_images.to_csv(csv_file, index=False)
    print(f"CSV saved!")
    
    # Push to Hugging Face Hub
    hub_repo = "kr3131/architecture-annotations-v2"
    print(f"\n{'='*60}")
    print(f"Pushing to Hugging Face Hub: {hub_repo}")
    print(f"This may take several minutes due to image uploads...")
    print(f"{'='*60}")
    
    try:
        dataset.push_to_hub(hub_repo, private=True)
        print(f"\n✓ Dataset successfully pushed to: https://huggingface.co/datasets/{hub_repo}")
    except Exception as e:
        print(f"\n✗ Error pushing to hub: {e}")
        print("Dataset is saved locally and you can try pushing again later with:")
        print(f"  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{output_dir}')")
        print(f"  dataset.push_to_hub('{hub_repo}', private=True)")
    
    return dataset

if __name__ == "__main__":
    dataset = main()