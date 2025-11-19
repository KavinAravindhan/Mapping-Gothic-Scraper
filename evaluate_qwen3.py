import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import torch
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
from tqdm import tqdm

class Config:
    # Building and paths
    BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    BASE_MAPS_DIR = "/mnt/swordfish-pool2/kavin/maps_output"
    
    # Task variant: "grid_direction" or "labeled_arrows"
    TASK_VARIANT = "grid_direction"
    
    # Prompt type: "zero_shot" or "few_shot"
    PROMPT_TYPE = "zero_shot"
    
    # Model Options: "qwen3-vl-32b-thinking", "qwen3-vl-7b-thinking"
    MODEL_VARIANT = "qwen3-vl-7b-thinking"
    
    # Model configuration
    MODEL_CONFIGS = {
        "qwen3-vl-32b-thinking": {
            "model_name": "Qwen/Qwen3-VL-30B-A3B-Thinking",
            "model_class": Qwen3VLMoeForConditionalGeneration,
            "display_name": "Qwen3-VL-32B-Thinking"
        },
        "qwen3-vl-7b-thinking": {
            "model_name": "Qwen/Qwen3-VL-8B-Thinking",
            "model_class": Qwen3VLForConditionalGeneration,
            "display_name": "Qwen3-VL-7B-Thinking"
        }
    }
    
    # Generation settings
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.7
    TOP_P = 0.95
    
    # Environment setup
    HF_CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"
    CUDA_DEVICE = "0"
    
    # For labeled_arrows variant
    NUM_ARROW_SAMPLES = 15
    RANDOM_SEED = 42
    
    # Output
    OUTPUT_DIR = None
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def get_building_dir(cls):
        return f"{cls.BASE_MAPS_DIR}/{cls.BUILDING_NAME}"
    
    @classmethod
    def setup_output_dir(cls):
        cls.OUTPUT_DIR = f"{cls.get_building_dir()}/evaluation_results"
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR
    
    @classmethod
    def get_model_config(cls):
        return cls.MODEL_CONFIGS[cls.MODEL_VARIANT]

# Set environment variables
os.environ['HF_HOME'] = Config.HF_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_DEVICE

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

def load_model_and_processor():
    """Load Qwen3-VL model and processor"""
    model_config = Config.get_model_config()
    model_name = model_config["model_name"]
    model_class = model_config["model_class"]
    
    print(f"Loading {model_config['display_name']}...")
    print(f"Model path: {model_name}")
    print(f"Cache directory: {Config.HF_CACHE_DIR}")
    
    # Load model
    model = model_class.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        cache_dir=Config.HF_CACHE_DIR
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=Config.HF_CACHE_DIR
    )
    
    print(f"Model loaded successfully")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    
    return model, processor

def run_inference(model, processor, floor_plan_path, photo_path, prompt_text):
    """Run inference on a single image pair"""
    
    # Prepare messages with two images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": floor_plan_path},
                {"type": "image", "image": photo_path},
            ],
        }
    ]
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            do_sample=True
        )
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def evaluate_all_images(ground_truth_df, task_variant, prompt_type, label_mapping=None):
    """Evaluate all images with the model"""
    
    building_dir = Config.get_building_dir()
    
    # Get appropriate floor map
    if task_variant == "grid_direction":
        floor_map_path = f"{building_dir}/{Config.BUILDING_NAME}_floorplan_gridded.jpg"
    else:  # labeled_arrows
        floor_map_path = f"{building_dir}/{Config.BUILDING_NAME}_floorplan_arrows_gridded.jpg"
    
    if not os.path.exists(floor_map_path):
        raise FileNotFoundError(f"Floor map not found: {floor_map_path}")
    
    # Get prompt
    if prompt_type == "zero_shot":
        prompt_text = get_zero_shot_prompt(task_variant)
    else:
        prompt_text = get_few_shot_prompt(task_variant)
    
    # Load model
    model, processor = load_model_and_processor()
    
    # Prepare test set
    if task_variant == "labeled_arrows" and label_mapping:
        test_images = [info['image_id'] for info in label_mapping.values()]
        test_df = ground_truth_df[ground_truth_df['image_id'].isin(test_images)]
    else:
        test_df = ground_truth_df
    
    images_dir = f"{building_dir}/images"
    
    # Run inference
    predictions = []

    print(f"Running inference on {len(test_df)} images...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing images"):
        image_path = f"{images_dir}/{row['image_id']}.jpg"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Run inference
            output_text = run_inference(model, processor, floor_map_path, image_path, prompt_text)
            
            # Parse response
            parsed = parse_response(output_text, task_variant)
            parsed['image_id'] = row['image_id']
            parsed['ground_truth_cell'] = row['grid_cell']
            parsed['ground_truth_direction'] = row['direction']
            
            predictions.append(parsed)
            
        except Exception as e:
            print(f"\nError processing {row['image_id']}: {e}")
            predictions.append({
                'image_id': row['image_id'],
                'reasoning': '',
                'grid_cell': None,
                'direction': None,
                'arrow_label': None,
                'raw_response': f"ERROR: {str(e)}",
                'ground_truth_cell': row['grid_cell'],
                'ground_truth_direction': row['direction']
            })
    
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df

def parse_response(response_text, task_variant):
    """Parse model response to extract predictions"""
    result = {
        'reasoning': '',
        'grid_cell': None,
        'direction': None,
        'arrow_label': None,
        'raw_response': response_text,
        'thinking': ''
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
    
    # If reasoning not found, try to extract from full response
    if not result['reasoning'] and response_text:
        result['reasoning'] = response_text.split('GRID_CELL:')[0].strip() if 'GRID_CELL:' in response_text else response_text[:500]
    
    return result

def calculate_metrics(predictions_df, ground_truth_df, task_variant, label_mapping=None):
    
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
            arrow_label = row.get('arrow_label')
            if arrow_label and arrow_label in label_mapping:
                gt_info = label_mapping[arrow_label]
                true_cell = gt_info['grid_cell']
                true_dir = gt_info['direction']
            else:
                metrics['unparseable_responses'] += 1
                continue
        else:
            true_cell = row['ground_truth_cell']
            true_dir = row['ground_truth_direction']
        
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

    output_dir = Config.setup_output_dir()
    
    model_config = Config.get_model_config()
    model_name_short = Config.MODEL_VARIANT
    
    # Save metrics
    metrics_file = f"{output_dir}/metrics_{model_name_short}_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.json"
    
    # Add config info to metrics
    metrics['config'] = {
        'building': Config.BUILDING_NAME,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'model_variant': Config.MODEL_VARIANT,
        'model_name': model_config['model_name'],
        'max_new_tokens': Config.MAX_NEW_TOKENS,
        'temperature': Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'timestamp': Config.TIMESTAMP
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save predictions
    predictions_file = f"{output_dir}/predictions_{model_name_short}_{Config.TASK_VARIANT}_{Config.PROMPT_TYPE}_{Config.TIMESTAMP}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")
    
    # Save label mapping if applicable
    if label_mapping:
        mapping_file = f"{output_dir}/arrow_label_mapping_{model_name_short}_{Config.TIMESTAMP}.json"
        with open(mapping_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Arrow label mapping saved to: {mapping_file}")
    
    # Print summary
    print(f"Building: {Config.BUILDING_NAME}")
    print(f"Model: {model_config['display_name']}")
    print(f"Task Variant: {Config.TASK_VARIANT}")
    print(f"Prompt Type: {Config.PROMPT_TYPE}")
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
    
    return metrics_file, predictions_file

def main():
    
    model_config = Config.get_model_config()
    
    print(f"Building: {Config.BUILDING_NAME}")
    print(f"Base Directory: {Config.BASE_MAPS_DIR}")
    print(f"Model: {model_config['display_name']}")
    print(f"Task Variant: {Config.TASK_VARIANT}")
    print(f"Prompt Type: {Config.PROMPT_TYPE}")
    print(f"Cache Directory: {Config.HF_CACHE_DIR}")
    print(f"CUDA Device: {Config.CUDA_DEVICE}")
    
    # Load ground truth
    print("Loading ground truth...")
    try:
        ground_truth_df = load_ground_truth(Config.BUILDING_NAME)
        print(f"Loaded {len(ground_truth_df)} images with coordinates")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Handle labeled arrows variant
    label_mapping = None
    if Config.TASK_VARIANT == "labeled_arrows":
        print("\nCreating labeled arrows floor plan...")
        building_dir = Config.get_building_dir()
        base_map = f"{building_dir}/{Config.BUILDING_NAME}_arrows_visualization.jpg"
        output_map = f"{building_dir}/{Config.BUILDING_NAME}_floorplan_arrows_gridded.jpg"
    
    # Run evaluation
    try:
        predictions_df = evaluate_all_images(
            ground_truth_df,
            Config.TASK_VARIANT,
            Config.PROMPT_TYPE,
            label_mapping
        )
        
        print(f"\nCompleted inference on {len(predictions_df)} images")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    try:
        metrics, predictions_df = calculate_metrics(
            predictions_df,
            ground_truth_df,
            Config.TASK_VARIANT,
            label_mapping
        )
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    try:
        save_results(metrics, predictions_df, label_mapping)
        print("\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()