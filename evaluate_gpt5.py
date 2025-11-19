import os
import json
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import time
from datetime import datetime

class Config:
    # Building and paths
    BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    BASE_MAPS_DIR = "/mnt/swordfish-pool2/kavin/maps_output"
    
    # Task variant: "grid_direction" or "labeled_arrows"
    TASK_VARIANT = "grid_direction"
    
    # Prompt type: "zero_shot" or "few_shot"
    PROMPT_TYPE = "zero_shot"
    
    # Model Options: "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano"
    MODEL = "gpt-5.1"
    
    # Reasoning effort: "none", "low", "medium", "high"
    # "none" for fastest responses, "high" for complex reasoning
    REASONING_EFFORT = "medium"
    
    # Verbosity: "low", "medium", "high"
    VERBOSITY = "medium"
    
    # API settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    MAX_OUTPUT_TOKENS = 1000
    
    # For labeled_arrows variant
    NUM_ARROW_SAMPLES = 15  # Number of arrows to label
    RANDOM_SEED = 42
    
    # Output
    OUTPUT_DIR = None  # Will be set dynamically
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def get_building_dir(cls):
        """Get the building directory path"""
        return f"{cls.BASE_MAPS_DIR}/{cls.BUILDING_NAME}"
    
    @classmethod
    def setup_output_dir(cls):
        """Setup output directory"""
        cls.OUTPUT_DIR = f"{cls.get_building_dir()}/evaluation_results"
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR

def encode_image_to_base64(image_path):
    """Encode image to base64 string for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_ground_truth(building_name):
    """Load coordinates CSV with ground truth"""
    building_dir = Config.get_building_dir()
    csv_path = f"{building_dir}/coordinates.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Coordinates CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Convert normalized coordinates to grid cells (10x10 grid: A-J, 1-10)
    df['grid_col'] = df['x'].apply(lambda x: chr(65 + min(int(x * 10), 9)))  # A-J
    df['grid_row'] = df['y'].apply(lambda y: min(int(y * 10) + 1, 10))  # 1-10
    df['grid_cell'] = df['grid_col'] + df['grid_row'].astype(str)
    
    return df

def calculate_grid_distance(pred_cell, true_cell):
    """Calculate Manhattan distance between two grid cells"""
    if not pred_cell or len(pred_cell) < 2:
        return None
    
    try:
        pred_col = ord(pred_cell[0].upper()) - 65
        pred_row = int(pred_cell[1:]) - 1
        true_col = ord(true_cell[0].upper()) - 65
        true_row = int(true_cell[1:]) - 1
        
        return abs(pred_col - true_col) + abs(pred_row - true_row)
    except:
        return None

def calculate_direction_distance(pred_dir, true_dir):
    """Calculate angular distance between two directions (in degrees)"""
    direction_angles = {
        'E': 0, 'NE': 45, 'N': 90, 'NW': 135,
        'W': 180, 'SW': 225, 'S': 270, 'SE': 315
    }
    
    if pred_dir not in direction_angles or true_dir not in direction_angles:
        return None
    
    angle_diff = abs(direction_angles[pred_dir] - direction_angles[true_dir])
    # Take the shorter angular distance
    return min(angle_diff, 360 - angle_diff)

def get_zero_shot_prompt(task_variant):
    """Generate zero-shot prompt"""
    if task_variant == "grid_direction":
        return """Given a gridded floor plan and a photograph taken inside or near the building, identify:
1. The grid cell where the photo was taken (format: [Letter][Number], e.g., C5)
2. The camera direction (N, NE, E, SE, S, SW, W, NW)

Analyze the architectural features, columns, windows, and spatial relationships carefully.
If the photo appears to be taken outside the main floor plan area, make your best estimate based on the visible features.

Think through your reasoning step by step, then provide your answer in this exact format:
REASONING: [Your detailed step-by-step analysis]
GRID_CELL: [Letter][Number]
DIRECTION: [Direction]"""
    
    else:  # labeled_arrows
        return """Given a floor plan with labeled arrows (A, B, C, etc.) and a photograph taken inside or near the building, identify which arrow corresponds to the location and viewing direction of the photograph.

Each arrow represents a specific location and camera direction. Analyze:
1. Architectural features visible in the photo
2. Spatial relationships and layout
3. Column positions, walls, and openings
4. Match these to the arrow locations on the floor plan

Think through your reasoning step by step, then provide your answer in this exact format:
REASONING: [Your detailed step-by-step analysis]
ARROW_LABEL: [Letter]"""

def get_few_shot_prompt(task_variant):
    """Generate few-shot prompt with examples"""
    
    if task_variant == "grid_direction":
        return """I'll show you examples of how to match photos to floor plan locations, then you'll solve a new one.

EXAMPLE 1:
Analysis: The photo shows a triple-column arcade along the left side and a large window ahead. On the floor plan, this matches the north aisle area with columns running north-south and the window to the east.
Answer: 
REASONING: Triple arcade on left indicates north aisle colonnade. Large east-facing window visible ahead. Position is mid-nave.
GRID_CELL: C4
DIRECTION: E

EXAMPLE 2:
Analysis: The photo captures a corner view with vaulted ceiling converging toward the apse. Single column visible in foreground with ambulatory visible to the right.
Answer:
REASONING: Converging vaults indicate view toward apse. Column placement and ambulatory access suggests southern crossing area looking northwest.
GRID_CELL: F6
DIRECTION: NW

NOW YOUR TURN:
Given a gridded floor plan and a photograph taken inside or near the building, identify:
1. The grid cell where the photo was taken (format: [Letter][Number], e.g., C5)
2. The camera direction (N, NE, E, SE, S, SW, W, NW)

Think through your reasoning step by step, then provide your answer in this exact format:
REASONING: [Your detailed step-by-step analysis]
GRID_CELL: [Letter][Number]
DIRECTION: [Direction]"""
    
    else:  # labeled_arrows
        return """I'll show you examples of matching photos to labeled arrows, then you'll solve a new one.

EXAMPLE 1:
Analysis: Photo shows arcade with three visible columns on the left, window on the right. Arrow C on the floor plan is positioned in the north aisle with this exact configuration.
Answer:
REASONING: Three-column arcade on left matches north colonnade. East window visible right. Arrow C position and orientation aligns perfectly.
ARROW_LABEL: C

EXAMPLE 2:
Analysis: Photo shows vaulted ceiling converging toward apse, ambulatory visible. Arrow H points northwest from southern crossing, matching this view.
Answer:
REASONING: Apse-directed vaulting and ambulatory access visible. Arrow H from southern crossing facing NW matches exactly.
ARROW_LABEL: H

NOW YOUR TURN:
Given a floor plan with labeled arrows and a photograph, identify which arrow corresponds to the photo location and direction.

Think through your reasoning step by step, then provide your answer in this exact format:
REASONING: [Your detailed step-by-step analysis]
ARROW_LABEL: [Letter]"""

def prepare_batch_requests(ground_truth_df, task_variant, prompt_type):
    """Prepare batch API requests for GPT-5.1 Responses API"""
    
    building_dir = Config.get_building_dir()
    
    # Get appropriate floor map
    if task_variant == "grid_direction":
        floor_map_path = f"{building_dir}/{Config.BUILDING_NAME}_floorplan_gridded.jpg"
        label_mapping = None
    else:  # labeled_arrows
        # Create labeled arrows version
        base_map = f"{building_dir}/{Config.BUILDING_NAME}_arrows_visualization.jpg"
        floor_map_path = f"{building_dir}/{Config.BUILDING_NAME}_floorplan_arrows_gridded.jpg"

    if not os.path.exists(floor_map_path):
        raise FileNotFoundError(f"Floor map not found: {floor_map_path}")
    
    # Encode floor map
    floor_map_base64 = encode_image_to_base64(floor_map_path)
    
    # Get prompt
    if prompt_type == "zero_shot":
        system_prompt = get_zero_shot_prompt(task_variant)
    else:
        system_prompt = get_few_shot_prompt(task_variant)
    
    # Prepare requests
    batch_requests = []
    
    # For labeled_arrows, only use the sampled images
    if task_variant == "labeled_arrows" and label_mapping:
        test_images = [info['image_id'] for info in label_mapping.values()]
        test_df = ground_truth_df[ground_truth_df['image_id'].isin(test_images)]
    else:
        test_df = ground_truth_df
    
    images_dir = f"{building_dir}/images"
    
    for idx, row in test_df.iterrows():
        image_path = f"{images_dir}/{row['image_id']}.jpg"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Create request using GPT-5.1 Responses API format
        request = {
            "custom_id": f"{task_variant}_{prompt_type}_{row['image_id']}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": Config.MODEL,
                "input": [
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": floor_map_base64
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ],
                "reasoning": {
                    "effort": Config.REASONING_EFFORT
                },
                "text": {
                    "verbosity": Config.VERBOSITY
                },
                "max_output_tokens": Config.MAX_OUTPUT_TOKENS
            }
        }
        
        batch_requests.append(request)
    
    return batch_requests, label_mapping

def save_batch_file(requests, filename):
    """Save requests to JSONL file for batch API"""
    output_dir = Config.setup_output_dir()
    filepath = f"{output_dir}/{filename}"
    
    with open(filepath, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Saved {len(requests)} requests to {filepath}")
    return filepath

def parse_response(response_text, task_variant):
    """Parse model response to extract predictions"""
    result = {
        'reasoning': '',
        'grid_cell': None,
        'direction': None,
        'arrow_label': None,
        'raw_response': response_text
    }
    
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('REASONING:'):
            result['reasoning'] = line.replace('REASONING:', '').strip()
        elif line.startswith('GRID_CELL:'):
            result['grid_cell'] = line.replace('GRID_CELL:', '').strip().upper()
        elif line.startswith('DIRECTION:'):
            result['direction'] = line.replace('DIRECTION:', '').strip().upper()
        elif line.startswith('ARROW_LABEL:'):
            result['arrow_label'] = line.replace('ARROW_LABEL:', '').strip().upper()
    
    return result

def calculate_metrics(predictions_df, ground_truth_df, task_variant, label_mapping=None):
    """Calculate evaluation metrics"""
    
    metrics = {
        'total_samples': len(predictions_df),
        'grid_cell_accuracy': 0,
        'direction_accuracy': 0,
        'exact_match_accuracy': 0,
        'avg_grid_distance': 0,
        'avg_direction_distance': 0,
        'unparseable_responses': 0
    }
    
    grid_distances = []
    direction_distances = []
    exact_matches = 0
    grid_correct = 0
    direction_correct = 0
    
    for idx, row in predictions_df.iterrows():
        # Get ground truth
        if task_variant == "labeled_arrows":
            # Map arrow label back to ground truth
            arrow_label = row.get('arrow_label')
            if arrow_label and arrow_label in label_mapping:
                gt_info = label_mapping[arrow_label]
                true_cell = gt_info['grid_cell']
                true_dir = gt_info['direction']
            else:
                metrics['unparseable_responses'] += 1
                continue
        else:
            gt_row = ground_truth_df[ground_truth_df['image_id'] == row['image_id']].iloc[0]
            true_cell = gt_row['grid_cell']
            true_dir = gt_row['direction']
        
        pred_cell = row.get('grid_cell')
        pred_dir = row.get('direction')
        
        # Check if response is parseable
        if task_variant == "grid_direction":
            if not pred_cell or not pred_dir:
                metrics['unparseable_responses'] += 1
                continue
        
        # Calculate metrics
        if pred_cell == true_cell:
            grid_correct += 1
        
        if pred_dir == true_dir:
            direction_correct += 1
        
        if pred_cell == true_cell and pred_dir == true_dir:
            exact_matches += 1
        
        # Distance metrics
        grid_dist = calculate_grid_distance(pred_cell, true_cell)
        if grid_dist is not None:
            grid_distances.append(grid_dist)
        
        dir_dist = calculate_direction_distance(pred_dir, true_dir)
        if dir_dist is not None:
            direction_distances.append(dir_dist)
    
    # Calculate final metrics
    valid_samples = len(predictions_df) - metrics['unparseable_responses']
    
    if valid_samples > 0:
        metrics['grid_cell_accuracy'] = grid_correct / valid_samples
        metrics['direction_accuracy'] = direction_correct / valid_samples
        metrics['exact_match_accuracy'] = exact_matches / valid_samples
    
    if grid_distances:
        metrics['avg_grid_distance'] = np.mean(grid_distances)
        metrics['median_grid_distance'] = np.median(grid_distances)
        metrics['std_grid_distance'] = np.std(grid_distances)
    
    if direction_distances:
        metrics['avg_direction_distance'] = np.mean(direction_distances)
        metrics['median_direction_distance'] = np.median(direction_distances)
        metrics['std_direction_distance'] = np.std(direction_distances)
    
    return metrics, predictions_df

def save_results(metrics, predictions_df, label_mapping=None):
    """Save evaluation results"""
    output_dir = Config.setup_output_dir()
    
    # Save metrics
    metrics_file = f"{output_dir}/metrics_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save predictions
    predictions_file = f"{output_dir}/predictions_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")
    
    # Save label mapping if applicable
    if label_mapping:
        mapping_file = f"{output_dir}/arrow_label_mapping_{Config.TIMESTAMP}.json"
        with open(mapping_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Arrow label mapping saved to: {mapping_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Building: {Config.BUILDING_NAME}")
    print(f"Task Variant: {Config.TASK_VARIANT}")
    print(f"Prompt Type: {Config.PROMPT_TYPE}")
    print(f"Model: {Config.MODEL}")
    print(f"Reasoning Effort: {Config.REASONING_EFFORT}")
    print(f"Verbosity: {Config.VERBOSITY}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Unparseable Responses: {metrics['unparseable_responses']}")
    print(f"\nAccuracy Metrics:")
    print(f"  Grid Cell Accuracy: {metrics['grid_cell_accuracy']:.2%}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"\nDistance Metrics:")
    print(f"  Avg Grid Distance: {metrics.get('avg_grid_distance', 0):.2f} cells")
    print(f"  Median Grid Distance: {metrics.get('median_grid_distance', 0):.2f} cells")
    print(f"  Avg Direction Distance: {metrics.get('avg_direction_distance', 0):.2f}°")
    print(f"  Median Direction Distance: {metrics.get('median_direction_distance', 0):.2f}°")
    print("="*60)

def process_batch_results(batch_results_file, ground_truth_df, task_variant, label_mapping=None):
    """Process results from OpenAI Batch API"""
    
    print(f"Processing batch results from: {batch_results_file}")
    
    # Load batch results
    predictions = []
    with open(batch_results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            
            # Extract custom_id to match with ground truth
            custom_id = result['custom_id']
            image_id = custom_id.split('_')[-1]
            
            # Extract response - GPT-5.1 uses different response format
            if result['response']['status_code'] == 200:
                # GPT-5.1 Responses API returns output_text directly
                response_text = result['response']['body'].get('output_text', '')
                
                # Fallback to other possible formats
                if not response_text and 'choices' in result['response']['body']:
                    response_text = result['response']['body']['choices'][0]['message']['content']
                
                parsed = parse_response(response_text, task_variant)
                parsed['image_id'] = image_id
                parsed['custom_id'] = custom_id
                predictions.append(parsed)
            else:
                print(f"Error for {custom_id}: {result['response']['status_code']}")
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics, predictions_df = calculate_metrics(predictions_df, ground_truth_df, task_variant, label_mapping)
    
    # Save results
    save_results(metrics, predictions_df, label_mapping)
    
    return metrics, predictions_df

def submit_batch(batch_filepath):
    """Submit batch file to OpenAI API"""
    from openai import OpenAI
    
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    print(f"\nUploading batch file: {batch_filepath}")
    
    # Upload file
    with open(batch_filepath, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose='batch'
        )
    
    print(f"Uploaded file ID: {batch_file.id}")
    
    # Create batch
    print(f"Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint='/v1/responses',
        completion_window='24h'
    )
    
    print(f"Batch created successfully!")
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    print(f"  Created at: {batch.created_at}")
    
    # Save batch info
    output_dir = Config.setup_output_dir()
    batch_info_file = f"{output_dir}/batch_info_{Config.TIMESTAMP}.json"
    
    batch_info = {
        'batch_id': batch.id,
        'file_id': batch_file.id,
        'status': batch.status,
        'created_at': batch.created_at,
        'endpoint': '/v1/responses',
        'building_name': Config.BUILDING_NAME,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'model': Config.MODEL,
        'reasoning_effort': Config.REASONING_EFFORT,
        'verbosity': Config.VERBOSITY
    }
    
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"Batch info saved to: {batch_info_file}")
    
    return batch.id, batch_info_file

def check_batch_status(batch_id):
    """Check status of a batch job"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    batch = client.batches.retrieve(batch_id)
    
    print(f"\nBatch Status Report")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created: {batch.created_at}")
    
    if hasattr(batch, 'request_counts'):
        counts = batch.request_counts
        print(f"\nRequest Counts:")
        print(f"  Total: {counts.total}")
        print(f"  Completed: {counts.completed}")
        print(f"  Failed: {counts.failed}")
    
    if batch.status == 'completed':
        print(f"\nBatch completed!")
        print(f"  Output file ID: {batch.output_file_id}")
        if batch.error_file_id:
            print(f"  Error file ID: {batch.error_file_id}")
    elif batch.status == 'failed':
        print(f"\nBatch failed!")
        if hasattr(batch, 'errors'):
            print(f"  Errors: {batch.errors}")
    elif batch.status in ['validating', 'in_progress', 'finalizing']:
        print(f"\nBatch is still processing...")
    
    return batch

def download_batch_results(batch_id, output_filename=None):
    """Download results from a completed batch"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Check status first
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != 'completed':
        print(f"Batch not completed yet. Current status: {batch.status}")
        return None
    
    print(f"\nDownloading batch results...")
    
    # Download results
    result_file_id = batch.output_file_id
    result_content = client.files.content(result_file_id)
    
    # Save to file
    output_dir = Config.setup_output_dir()
    if output_filename is None:
        output_filename = f"batch_results_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.jsonl"
    
    output_path = f"{output_dir}/{output_filename}"
    
    with open(output_path, 'wb') as f:
        f.write(result_content.content)
    
    print(f"Results saved to: {output_path}")
    
    # Download errors if any
    if batch.error_file_id:
        print(f"\nDownloading error file...")
        error_content = client.files.content(batch.error_file_id)
        error_path = f"{output_dir}/batch_errors_{Config.TIMESTAMP}.jsonl"
        
        with open(error_path, 'wb') as f:
            f.write(error_content.content)
        
        print(f"Errors saved to: {error_path}")
    
    return output_path

def monitor_batch(batch_id, check_interval=60):
    """Monitor batch progress until completion"""
    from openai import OpenAI
    import time
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    print(f"\nMonitoring batch {batch_id}")
    print(f"Checking every {check_interval} seconds...")
    print(f"{'='*60}")
    
    while True:
        batch = client.batches.retrieve(batch_id)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = batch.status
        
        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            progress = f"{counts.completed}/{counts.total}"
            print(f"[{timestamp}] Status: {status:<15} Progress: {progress}")
        else:
            print(f"[{timestamp}] Status: {status}")
        
        if status == 'completed':
            print(f"\nBatch completed successfully!")
            return batch
        elif status == 'failed':
            print(f"\nBatch failed!")
            return batch
        elif status in ['expired', 'cancelled']:
            print(f"\nBatch {status}!")
            return batch
        
        time.sleep(check_interval)

# ==================== COMPLETE WORKFLOW ====================

def run_complete_evaluation(auto_submit=False, monitor=True):
    """
    Run complete evaluation workflow:
    1. Prepare batch requests
    2. Optionally submit to OpenAI
    3. Optionally monitor progress
    4. Download and process results
    """
    
    # Step 1: Prepare batch
    print("\n[1/4] Preparing batch requests...")
    
    try:
        ground_truth_df = load_ground_truth(Config.BUILDING_NAME)
        print(f"Loaded {len(ground_truth_df)} images")
        
        batch_requests, label_mapping = prepare_batch_requests(
            ground_truth_df, 
            Config.TASK_VARIANT, 
            Config.PROMPT_TYPE
        )
        print(f"Prepared {len(batch_requests)} requests")
        
        batch_filename = f"batch_requests_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.jsonl"
        batch_filepath = save_batch_file(batch_requests, batch_filename)
        
    except Exception as e:
        print(f"Error preparing batch: {e}")
        return
    
    # Step 2: Submit batch
    if not auto_submit:
        print(f"\nBatch file ready at: {batch_filepath}")
        return batch_filepath, label_mapping
    
    print("\n[2/4] Submitting batch to OpenAI...")
    
    try:
        batch_id, batch_info_file = submit_batch(batch_filepath)
    except Exception as e:
        print(f"Error submitting batch: {e}")
        return
    
    # Step 3: Monitor batch
    if not monitor:
        print(f"\nBatch submitted. Check status with:")
        print(f"  check_batch_status('{batch_id}')")
        return batch_id, label_mapping
    
    print("\n[3/4] Monitoring batch progress...")
    
    try:
        final_batch = monitor_batch(batch_id, check_interval=60)
    except KeyboardInterrupt:
        print(f"\n\nMonitoring interrupted. Batch is still running.")
        print(f"Check status later with: check_batch_status('{batch_id}')")
        return batch_id, label_mapping
    except Exception as e:
        print(f"Error monitoring batch: {e}")
        return batch_id, label_mapping
    
    if final_batch.status != 'completed':
        print(f"Batch did not complete successfully")
        return batch_id, label_mapping
    
    # Step 4: Download and process results
    print("\n[4/4] Processing results...")
    
    try:
        results_file = download_batch_results(batch_id)
        
        if results_file:
            print(f"\nCalculating metrics...")
            metrics, predictions_df = process_batch_results(
                results_file,
                ground_truth_df,
                Config.TASK_VARIANT,
                label_mapping
            )
            
            print("EVALUATION COMPLETE!")
            
            return metrics, predictions_df
        
    except Exception as e:
        print(f"Error processing results: {e}")
        return batch_id, label_mapping

def process_existing_batch(batch_id):
    """Process results from an already submitted batch"""
    
    print(f"Processing existing batch: {batch_id}")
    
    # Check status
    batch = check_batch_status(batch_id)
    
    if batch.status != 'completed':
        print(f"\nBatch not ready for processing.")
        return
    
    # Download results
    results_file = download_batch_results(batch_id)
    
    if not results_file:
        return
    
    # Load ground truth
    ground_truth_df = load_ground_truth(Config.BUILDING_NAME)
    
    # Load label mapping if it exists
    label_mapping = None
    if Config.TASK_VARIANT == "labeled_arrows":
        output_dir = Config.setup_output_dir()
        mapping_files = list(Path(output_dir).glob("arrow_label_mapping_*.json"))
        
        if mapping_files:
            # Use the most recent mapping file
            latest_mapping = sorted(mapping_files)[-1]
            print(f"Loading label mapping from: {latest_mapping}")
            
            with open(latest_mapping, 'r') as f:
                label_mapping = json.load(f)
    
    # Process results
    metrics, predictions_df = process_batch_results(
        results_file,
        ground_truth_df,
        Config.TASK_VARIANT,
        label_mapping
    )
    
    return metrics, predictions_df

def main(auto_submit=False, monitor=False):
    """
    Main function with optional auto-submission
    
    Args:
        auto_submit: If True, automatically submit batch to OpenAI
        monitor: If True, monitor batch progress until completion
    """
    
    if auto_submit or monitor:
        return run_complete_evaluation(auto_submit=auto_submit, monitor=monitor)
    
    print(f"Building: {Config.BUILDING_NAME}")
    print(f"Base Directory: {Config.BASE_MAPS_DIR}")
    print(f"Task Variant: {Config.TASK_VARIANT}")
    print(f"Prompt Type: {Config.PROMPT_TYPE}")
    print(f"Model: {Config.MODEL}")
    print(f"Reasoning Effort: {Config.REASONING_EFFORT}")
    print(f"Verbosity: {Config.VERBOSITY}")
    
    # Load ground truth
    print("Loading ground truth...")
    try:
        ground_truth_df = load_ground_truth(Config.BUILDING_NAME)
        print(f"Loaded {len(ground_truth_df)} images with coordinates")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Prepare batch requests
    print("\nPreparing batch requests...")
    try:
        batch_requests, label_mapping = prepare_batch_requests(
            ground_truth_df, 
            Config.TASK_VARIANT, 
            Config.PROMPT_TYPE
        )
        print(f"Prepared {len(batch_requests)} batch requests")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Save batch file
    batch_filename = f"batch_requests_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.jsonl"
    batch_filepath = save_batch_file(batch_requests, batch_filename)
    
    print(f"Batch File: {batch_filepath}")
    print(f"Requests: {len(batch_requests)}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    auto_submit = '--auto-submit' in sys.argv or '-s' in sys.argv
    monitor = '--monitor' in sys.argv or '-m' in sys.argv
    
    main(auto_submit=auto_submit, monitor=monitor)

# python evaluate_gpt5_spatial_localization.py
# python evaluate_gpt5_spatial_localization.py --auto-submit --monitor
# python evaluate_gpt5_spatial_localization.py --auto-submit