import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datasets import Dataset, Features, Value, Sequence


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
            
            for result in annotation['result']:
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
                    'image_filename': None,
                    'image_url': None,
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
                            'building_caption': None,  # Not applicable at bbox level
                            
                            # Annotation metadata
                            'annotation_id': annotation_id,
                            'annotator_email': annotator_email,
                            'annotation_created_at': created_at,
                            'annotation_updated_at': updated_at,
                            
                            # Image metadata
                            'image_index': image_idx,
                            'image_filename': image_filename,
                            'image_url': image_url,
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
                        'building_caption': None,  # Not applicable at image level
                        
                        # Annotation metadata
                        'annotation_id': annotation_id,
                        'annotator_email': annotator_email,
                        'annotation_created_at': created_at,
                        'annotation_updated_at': updated_at,
                        
                        # Image metadata
                        'image_index': image_idx,
                        'image_filename': image_filename,
                        'image_url': image_url,
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
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(df)
    
    return dataset


def main():    

    input_file = '/home/kr3131/Mapping-Gothic-Scraper/benchmarking/export_197277_project-197277-at-2025-11-17-18-59-c7e831d2.json'
    
    print(f"Loading Label Studio annotations from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both single object and list formats
    if isinstance(json_data, dict):
        json_data = [json_data]
    
    print(f"Processing {len(json_data)} tasks...")
    
    dataset = parse_label_studio_annotations(json_data)
    
    # Display dataset information
    print(f"\nDataset created")
    print(f"  Total rows: {len(dataset)}")
    print(f"  Features ({len(dataset.features)}):")
    for feature_name in dataset.features:
        print(f"    - {feature_name}")
    
    # Convert to pandas for analysis
    df = pd.DataFrame(dataset[:])
    
    # Show data type distribution
    print(f"\n  Data Type Distribution:")
    print(df['data_type'].value_counts())
    
    # Show sample rows
    print(f"\n  Sample rows:")
    sample_df = pd.DataFrame(dataset[:5])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(sample_df.to_string())
    
    # Show samples by data type
    for data_type in ['building', 'image', 'bbox']:
        type_df = df[df['data_type'] == data_type]
        if len(type_df) > 0:
            print(f"\n  Sample {data_type}-level row:")
            print(type_df.head(1).to_string())
    
    # Save dataset
    output_dir = 'architecture_annotations_hf_dataset'
    dataset.save_to_disk(output_dir)
    print(f"\nDataset saved to '{output_dir}'")
    
    # Export to CSV
    csv_file = 'architecture_annotations.csv'
    df.to_csv(csv_file, index=False)
    print(f"CSV saved to '{csv_file}'")
    
    # hub_repo = "username/architecture-annotations"
    # dataset.push_to_hub(hub_repo)
    # print(f"Dataset pushed to Hugging Face Hub: {hub_repo}")
    
    return dataset

if __name__ == "__main__":
    dataset = main()