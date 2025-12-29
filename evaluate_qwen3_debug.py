# evaluate_qwen3_debug.py

import os
import sys
import functools
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
import re

class Config:
    
    # Mode settings
    SINGLE_BUILDING_MODE = False  # True for single building, False for all buildings
    SINGLE_BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    
    # Debug settings
    DEBUG_MODE = False  # Set to True to print first few responses
    DEBUG_SAMPLES = 3   # Number of responses to print in debug mode
    
    # Paths
    SOURCE_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output"  # Original building data
    GRIDDED_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/grids_floorplan"  # Gridded floorplans
    OUTPUT_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/qwen_evaluation"  # Results
    
    # Grid configuration
    GRID_SIZES = [10]  # List of grid sizes to evaluate: [10, 15, 20]
    
    # Task variant: "grid_direction" or "labeled_arrows"
    TASK_VARIANT = "grid_direction"
    
    # Prompt type: "zero_shot" or "few_shot"
    PROMPT_TYPE = "zero_shot"
    
    # Model Options
    MODEL_VARIANT = "qwen3-vl-8b-instruct"
    
    # Model configuration
    MODEL_CONFIGS = {
        "qwen3-vl-32b-thinking": {
            "model_name": "Qwen/Qwen3-VL-32B-Thinking",
            "model_class": Qwen3VLMoeForConditionalGeneration,
            "display_name": "Qwen3-VL-32B-Thinking"
        },
        "qwen3-vl-8b-thinking": {
            "model_name": "Qwen/Qwen3-VL-8B-Thinking",
            "model_class": Qwen3VLForConditionalGeneration,
            "display_name": "Qwen3-VL-8B-Thinking"
        },
        "qwen3-vl-32b-instruct": {
            "model_name": "Qwen/Qwen3-VL-32B-Instruct",
            "model_class": Qwen3VLMoeForConditionalGeneration,
            "display_name": "Qwen3-VL-32B-Instruct"
        },
        "qwen3-vl-8b-instruct": {
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "model_class": Qwen3VLForConditionalGeneration,
            "display_name": "Qwen3-VL-8B-Instruct"
        }
    }
    
    # Generation settings - OPTIMIZED FOR BEST PERFORMANCE
    MAX_NEW_TOKENS = 512  # Balanced for speed and completeness
    USE_GREEDY = True     # Deterministic output
    TEMPERATURE = 0.1     # Only used if USE_GREEDY=False
    TOP_P = 0.95          # Only used if USE_GREEDY=False
    
    # Environment setup
    HF_CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"
    CUDA_DEVICE = "7"
    
    # For labeled_arrows variant
    NUM_ARROW_SAMPLES = 15
    
    # Runtime
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = None
    
    @classmethod
    def get_building_source_dir(cls, building_name):
        """Get source directory for building data"""
        print(f"DEBUG: get_building_source_dir({building_name})")
        sys.stdout.flush()
        return f"{cls.SOURCE_BASE_PATH}/{building_name}"
    
    @classmethod
    def get_gridded_floorplan_path(cls, building_name, grid_size):
        """Get path to gridded floorplan"""
        print(f"DEBUG: get_gridded_floorplan_path({building_name}, {grid_size})")
        sys.stdout.flush()
        return f"{cls.GRIDDED_BASE_PATH}/grid_size_{grid_size}/{building_name}_floorplan_gridded.jpg"
    
    @classmethod
    def get_ground_truth_csv_path(cls, building_name, grid_size):
        """Get path to pre-computed ground truth CSV"""
        print(f"DEBUG: get_ground_truth_csv_path({building_name}, {grid_size})")
        sys.stdout.flush()
        return f"{cls.GRIDDED_BASE_PATH}/grid_size_{grid_size}_ground_truth/{building_name}_ground_truth.csv"
    
    @classmethod
    def setup_output_dir(cls, grid_size=None, building_name=None):
        """Setup output directory structure"""
        print(f"DEBUG: setup_output_dir(grid_size={grid_size}, building_name={building_name})")
        sys.stdout.flush()
        base_dir = f"{cls.OUTPUT_BASE_PATH}/{cls.TIMESTAMP}"
        
        if grid_size is not None:
            base_dir = f"{base_dir}/grid_size_{grid_size}"
        
        if building_name is not None:
            base_dir = f"{base_dir}/{building_name}"
        
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    @classmethod
    def get_model_config(cls):
        print(f"DEBUG: get_model_config()")
        sys.stdout.flush()
        return cls.MODEL_CONFIGS[cls.MODEL_VARIANT]

# Set environment variables
os.environ['HF_HOME'] = Config.HF_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_DEVICE


def generate_column_label(col_index, num_cols):
    """
    Generate column label for a given index (A, B, C, ..., Z, AA, AB, ...)
    Matches the logic in create_grid_overlay.py
    """
    # Note: Not adding debug here as this is called frequently
    if col_index >= num_cols:
        col_index = num_cols - 1
    
    label = ""
    num = col_index
    while True:
        label = chr(65 + (num % 26)) + label
        num = num // 26
        if num == 0:
            break
        num -= 1
    return label


def validate_building_folder(building_name):
    """
    Validate that a building folder has all required files
    Returns (is_valid, missing_items)
    """
    print(f"DEBUG: validate_building_folder({building_name})")
    sys.stdout.flush()
    
    building_dir = Config.get_building_source_dir(building_name)
    missing_items = []
    
    # Check for required items
    required_items = {
        'images': os.path.join(building_dir, 'images'),
        'floorplan': os.path.join(building_dir, f'{building_name}_floorplan.jpg'),
        'coordinates': os.path.join(building_dir, 'coordinates.csv'),
        'data': os.path.join(building_dir, 'data.json')
    }
    
    for item_name, item_path in required_items.items():
        if not os.path.exists(item_path):
            missing_items.append(item_name)
    
    is_valid = len(missing_items) == 0
    return is_valid, missing_items


def get_all_buildings():
    """
    Get list of all building folders and validate them
    Returns (valid_buildings, invalid_buildings_info)
    """
    print(f"DEBUG: get_all_buildings()")
    sys.stdout.flush()
    
    if not os.path.exists(Config.SOURCE_BASE_PATH):
        print(f"ERROR: Source path does not exist: {Config.SOURCE_BASE_PATH}")
        sys.stdout.flush()
        return [], {}
    
    all_folders = [f for f in os.listdir(Config.SOURCE_BASE_PATH) 
                   if os.path.isdir(os.path.join(Config.SOURCE_BASE_PATH, f))]
    
    valid_buildings = []
    invalid_buildings = {}
    
    for building_name in sorted(all_folders):
        is_valid, missing_items = validate_building_folder(building_name)
        
        if is_valid:
            valid_buildings.append(building_name)
        else:
            invalid_buildings[building_name] = missing_items
    
    return valid_buildings, invalid_buildings


def load_ground_truth(building_name, grid_size):
    """Load pre-computed ground truth CSV for specific grid size"""
    print(f"DEBUG: load_ground_truth({building_name}, {grid_size})")
    sys.stdout.flush()
    
    ground_truth_csv_path = Config.get_ground_truth_csv_path(building_name, grid_size)
    
    print(f"DEBUG: Checking if ground truth CSV exists: {ground_truth_csv_path}")
    sys.stdout.flush()
    
    if not os.path.exists(ground_truth_csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv_path}")
    
    print(f"DEBUG: Reading ground truth CSV...")
    sys.stdout.flush()
    
    # Load pre-computed ground truth
    df = pd.read_csv(ground_truth_csv_path)
    
    print(f"DEBUG: Loaded {len(df)} rows from ground truth CSV")
    sys.stdout.flush()
    
    # Validate required columns
    required_columns = ['image_id', 'grid_cell', 'direction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
    
    print(f"DEBUG: Ground truth validation complete")
    sys.stdout.flush()
    
    return df


def calculate_grid_distance(pred_cell, true_cell):
    """Calculate Manhattan distance between two grid cells"""
    # Note: Not adding debug here as this is called frequently
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
    # Note: Not adding debug here as this is called frequently
    direction_angles = {
        'E': 0, 'NE': 45, 'N': 90, 'NW': 135,
        'W': 180, 'SW': 225, 'S': 270, 'SE': 315
    }
    
    if pred_dir not in direction_angles or true_dir not in direction_angles:
        return None
    
    angle_diff = abs(direction_angles[pred_dir] - direction_angles[true_dir])
    return min(angle_diff, 360 - angle_diff)


def get_zero_shot_prompt(grid_size, task_variant):
    """Generate zero-shot prompt with Gothic architecture context"""
    print(f"DEBUG: get_zero_shot_prompt(grid_size={grid_size}, task_variant={task_variant})")
    sys.stdout.flush()
    
    if task_variant == "grid_direction":
        last_col = generate_column_label(grid_size - 1, grid_size)
        
        return f"""You are analyzing a Gothic church floor plan and interior photograph from 12th-13th century France.

ARCHITECTURAL CONTEXT:
This is a Gothic church building with typical features including:
- Nave (central space), side aisles, transept (crossing), choir, apse
- Columns/pillars supporting pointed arches and ribbed vaults
- Large windows (often with tracery), clerestory windows
- Ambulatory (walkway around choir/apse in some churches)
- Stone walls, vaulted ceilings, decorative capitals

TASK: Determine where the photo was taken and the camera direction.

GRID INFORMATION:
- Grid size: {grid_size}×{grid_size} grid overlaid on the floor plan
- Columns: A to {last_col} (left to right)
- Rows: 1 to {grid_size} (top to bottom)
- Directions: N (North), NE (Northeast), E (East), SE (Southeast), S (South), SW (Southwest), W (West), NW (Northwest)

ANALYSIS STEPS:
1. IDENTIFY visible Gothic architectural features in the photograph:
   - Column/pillar positions and their spacing
   - Pointed arches and vault configurations
   - Window locations, sizes, and orientations
   - Wall positions and openings
   - Spatial arrangement (nave, aisle, transept, choir, etc.)

2. LOCATE these features on the gridded floor plan:
   - Match column positions shown as dots/circles on plan
   - Identify window locations along walls
   - Find the corresponding section (nave, aisle, crossing, choir, apse)
   - Match the spatial layout pattern

3. DETERMINE camera position:
   - Which grid cell contains the viewpoint that would show these features in this arrangement?

4. DETERMINE camera direction:
   - Based on what architectural elements are visible ahead and to the sides, which direction is the camera facing?

IMPORTANT CONSTRAINTS:
- Only analyze architectural features CLEARLY VISIBLE in the photograph
- Gothic churches typically have east-west orientation (altar/apse usually to the east)
- Do not assume features not visible in the photo
- Base your answer on concrete architectural elements you can identify
- If uncertain, provide your best estimate but acknowledge this in reasoning

OUTPUT FORMAT (MANDATORY - YOU MUST PROVIDE ALL THREE FIELDS):
REASONING: [2-3 sentences identifying key Gothic features visible and their location on the plan]
GRID_CELL: [Letter][Number]
DIRECTION: [One of: N, NE, E, SE, S, SW, W, NW]

EXAMPLE OUTPUT:
REASONING: The photograph shows a row of three cylindrical columns on the left supporting pointed arches, with a large Gothic window with tracery visible ahead. This configuration matches the north aisle colonnade at grid cell C4, with the camera facing east toward the window.
GRID_CELL: C4
DIRECTION: E

Now analyze the provided floor plan and photograph. You MUST provide all three fields (REASONING, GRID_CELL, DIRECTION) even if uncertain."""
    
    else:  # labeled_arrows
        return """You are analyzing a Gothic church floor plan with labeled arrows and an interior photograph from 12th-13th century France.

ARCHITECTURAL CONTEXT:
This is a Gothic church with typical features:
- Nave, side aisles, transept, choir, apse
- Columns supporting pointed arches and ribbed vaults
- Gothic windows, stone walls, vaulted ceilings

TASK: Identify which labeled arrow (A, B, C, etc.) corresponds to the photograph's location and viewing direction.

ANALYSIS STEPS:
1. IDENTIFY visible Gothic architectural features in the photograph:
   - Column positions and spacing
   - Pointed arches and vault patterns
   - Window locations and types
   - Spatial section (nave, aisle, transept, choir, apse)

2. MATCH these features to the arrows on the floor plan:
   - Each arrow shows a position and viewing direction
   - Find the arrow that matches both the location and orientation of the photo

IMPORTANT CONSTRAINTS:
- Only use architectural features CLEARLY VISIBLE in the photograph
- Gothic churches typically have east-west orientation
- Do not assume features not shown
- Base your answer on concrete architectural matches

OUTPUT FORMAT (MANDATORY - YOU MUST PROVIDE ALL FIELDS):
REASONING: [2-3 sentences identifying key Gothic features and matching them to an arrow]
ARROW_LABEL: [Single letter A-Z]

EXAMPLE OUTPUT:
REASONING: Photo shows three columns on left supporting pointed arches with a Gothic window on right. Arrow C in the north aisle has this exact column spacing and east-facing window orientation.
ARROW_LABEL: C

Now analyze the images and provide your answer in the exact format above. You MUST provide both fields (REASONING, ARROW_LABEL) even if uncertain."""


def get_few_shot_prompt(grid_size, task_variant):
    """Generate few-shot prompt with Gothic architecture examples"""
    print(f"DEBUG: get_few_shot_prompt(grid_size={grid_size}, task_variant={task_variant})")
    sys.stdout.flush()
    
    if task_variant == "grid_direction":
        last_col = generate_column_label(grid_size - 1, grid_size)
        
        return f"""You are analyzing a Gothic church floor plan and interior photograph from 12th-13th century France.

ARCHITECTURAL CONTEXT:
Gothic churches have: nave, aisles, transept, choir, apse, columns with pointed arches, ribbed vaults, large windows.

I'll show you examples, then you'll solve a new one.

EXAMPLE 1:
Floor plan shows a {grid_size}×{grid_size} grid. Photo shows triple colonnade on left with pointed arches, large tracery window ahead.
Analysis: The triple arcade indicates north aisle with columns running along nave wall. The Gothic window to the east matches grid cell C4.
Answer:
REASONING: Triple arcade on left indicates north aisle colonnade. Large east-facing Gothic window visible ahead. Column spacing matches mid-nave position.
GRID_CELL: C4
DIRECTION: E

EXAMPLE 2:
Photo shows converging ribbed vaults toward apse, single column in foreground, ambulatory visible right.
Analysis: Converging vaults indicate view toward eastern apse. Column and ambulatory access suggests southern crossing area.
Answer:
REASONING: Converging ribbed vaults indicate view toward apse. Column placement and ambulatory access suggests southern crossing area looking northwest.
GRID_CELL: F6
DIRECTION: NW

NOW YOUR TURN:

GRID INFORMATION:
- Grid size: {grid_size}×{grid_size}
- Columns: A to {last_col} (left to right)
- Rows: 1 to {grid_size} (top to bottom)
- Directions: N, NE, E, SE, S, SW, W, NW

TASK: Identify where the photo was taken (grid cell) and camera direction.

ANALYSIS STEPS:
1. Identify visible Gothic features (columns, arches, windows, vaults)
2. Locate these features on the gridded floor plan
3. Determine grid cell and direction

OUTPUT FORMAT (MANDATORY):
REASONING: [2-3 sentences on Gothic features and their location]
GRID_CELL: [Letter][Number]
DIRECTION: [Direction]

You MUST provide all three fields even if uncertain."""
    
    else:  # labeled_arrows
        return """You are analyzing a Gothic church floor plan with labeled arrows and an interior photograph from 12th-13th century France.

ARCHITECTURAL CONTEXT:
Gothic churches have: nave, aisles, transept, choir, apse, columns with pointed arches, ribbed vaults, windows.

I'll show you examples, then you'll solve a new one.

EXAMPLE 1:
Photo shows arcade with three columns on left supporting pointed arches, Gothic window on right.
Analysis: Arrow C in north aisle matches this column configuration and east window orientation.
Answer:
REASONING: Three-column arcade on left matches north colonnade with pointed arches. East-facing Gothic window visible right. Arrow C position and orientation aligns perfectly.
ARROW_LABEL: C

EXAMPLE 2:
Photo shows ribbed vaulting converging toward apse, ambulatory walkway visible.
Analysis: Arrow H points northwest from southern crossing, matching this apse view.
Answer:
REASONING: Apse-directed ribbed vaulting and ambulatory access visible. Arrow H from southern crossing facing NW matches exactly.
ARROW_LABEL: H

NOW YOUR TURN:

TASK: Identify which arrow (A, B, C, etc.) corresponds to the photo location and direction.

ANALYSIS STEPS:
1. Identify visible Gothic features in photo
2. Match to arrow positions on floor plan
3. Find arrow with matching location and viewing direction

OUTPUT FORMAT (MANDATORY):
REASONING: [2-3 sentences on Gothic features and arrow match]
ARROW_LABEL: [Letter]

You MUST provide both fields even if uncertain."""


def load_model_and_processor():
    """Load Qwen3-VL model and processor"""
    print(f"DEBUG: load_model_and_processor()")
    sys.stdout.flush()
    
    model_config = Config.get_model_config()
    model_name = model_config["model_name"]
    model_class = model_config["model_class"]
    
    print(f"\nLoading {model_config['display_name']}...")
    print(f"Model path: {model_name}")
    print(f"Cache directory: {Config.HF_CACHE_DIR}")
    sys.stdout.flush()
    
    print(f"DEBUG: Loading model from pretrained...")
    sys.stdout.flush()
    
    # Load model
    model = model_class.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        cache_dir=Config.HF_CACHE_DIR
    )
    
    print(f"DEBUG: Loading processor...")
    sys.stdout.flush()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=Config.HF_CACHE_DIR
    )
    
    print(f"Model loaded successfully")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    sys.stdout.flush()
    
    return model, processor


def run_inference(model, processor, floor_plan_path, photo_path, prompt_text):
    """Run inference on a single image pair"""
    print(f"DEBUG: run_inference(floor_plan={floor_plan_path}, photo={photo_path})")
    sys.stdout.flush()
    
    print(f"DEBUG: Preparing messages...")
    sys.stdout.flush()
    
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
    
    print(f"DEBUG: Applying chat template...")
    sys.stdout.flush()
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    print(f"DEBUG: Running model.generate()...")
    sys.stdout.flush()
    
    # Generate with optimized settings
    with torch.no_grad():
        if Config.USE_GREEDY:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                do_sample=True
            )
    
    print(f"DEBUG: Decoding output...")
    sys.stdout.flush()
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"DEBUG: Inference complete, output length: {len(output_text)}")
    sys.stdout.flush()
    
    return output_text


def extract_structured_answer(response_text, task_variant):
    """Extract answer using multiple strategies with robust parsing"""
    # Note: Not adding debug here as this is called frequently
    result = {
        'reasoning': '',
        'grid_cell': None,
        'direction': None,
        'arrow_label': None,
        'raw_response': response_text,
        'thinking': ''
    }
    
    # Strategy 1: Look for exact format (case-insensitive)
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        line_upper = line.upper()
        
        if line_upper.startswith('REASONING:'):
            result['reasoning'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif line_upper.startswith('GRID_CELL:') or line_upper.startswith('GRID CELL:'):
            cell_text = line.split(':', 1)[1].strip() if ':' in line else ''
            cell_match = re.search(r'\b([A-Z]+)(\d{1,2})\b', cell_text.upper())
            if cell_match:
                result['grid_cell'] = cell_match.group(1) + cell_match.group(2)
            else:
                result['grid_cell'] = cell_text.upper()
        elif line_upper.startswith('DIRECTION:'):
            dir_text = line.split(':', 1)[1].strip() if ':' in line else ''
            for direction in ['NE', 'NW', 'SE', 'SW', 'N', 'S', 'E', 'W']:
                if direction in dir_text.upper():
                    result['direction'] = direction
                    break
        elif line_upper.startswith('ARROW_LABEL:') or line_upper.startswith('ARROW LABEL:'):
            label_text = line.split(':', 1)[1].strip() if ':' in line else ''
            letter_match = re.search(r'\b([A-Z])\b', label_text.upper())
            if letter_match:
                result['arrow_label'] = letter_match.group(1)
    
    # Strategy 2: Fallback extraction from text if structured format not found
    if task_variant == "grid_direction":
        if not result['grid_cell']:
            grid_pattern = r'\b([A-Z]+)(\d{1,2})\b'
            matches = re.findall(grid_pattern, response_text.upper())
            
            for col, row in reversed(matches):
                try:
                    row_num = int(row)
                    if row_num >= 1:  # Valid row number
                        result['grid_cell'] = f"{col}{row}"
                        break
                except ValueError:
                    continue
        
        if not result['direction']:
            text_upper = response_text.upper()
            
            for direction in ['NE', 'NW', 'SE', 'SW']:
                if direction in text_upper:
                    result['direction'] = direction
                    break
            
            if not result['direction']:
                direction_words = {
                    'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W',
                    ' N ': 'N', ' S ': 'S', ' E ': 'E', ' W ': 'W'
                }
                for word, abbrev in direction_words.items():
                    if word in f' {text_upper} ':
                        result['direction'] = abbrev
                        break
    
    elif task_variant == "labeled_arrows":
        if not result['arrow_label']:
            arrow_pattern = r'(?:ARROW|LABEL)[:\s]*([A-Z])\b'
            match = re.search(arrow_pattern, response_text.upper())
            if match:
                result['arrow_label'] = match.group(1)
            else:
                letter_match = re.search(r'\b([A-Z])\b', response_text.upper()[-100:])
                if letter_match:
                    result['arrow_label'] = letter_match.group(1)
    
    # Extract reasoning if not found
    if not result['reasoning']:
        reasoning_part = response_text.split('GRID_CELL:')[0].split('ARROW_LABEL:')[0]
        reasoning_part = reasoning_part.replace('REASONING:', '').strip()
        sentences = [s.strip() for s in reasoning_part.split('.') if len(s.strip()) > 20]
        if sentences:
            result['reasoning'] = '. '.join(sentences[:3]) + '.'
    
    return result


def validate_extraction(result, task_variant):
    """Validate that all required fields were extracted"""
    # Note: Not adding debug here as this is called frequently
    missing_fields = []
    
    if task_variant == "grid_direction":
        if not result['grid_cell']:
            missing_fields.append('GRID_CELL')
        if not result['direction']:
            missing_fields.append('DIRECTION')
    elif task_variant == "labeled_arrows":
        if not result['arrow_label']:
            missing_fields.append('ARROW_LABEL')
    
    if not result['reasoning']:
        missing_fields.append('REASONING')
    
    return missing_fields


def parse_response(response_text, task_variant):
    """Parse model response to extract predictions"""
    # Note: Not adding debug here as this is called frequently
    return extract_structured_answer(response_text, task_variant)


def evaluate_building(building_name, grid_size, model, processor):
    """
    Evaluate a single building with a specific grid size
    Returns (metrics, predictions_list) or (None, None) if error
    """
    print(f"DEBUG: evaluate_building({building_name}, {grid_size})")
    sys.stdout.flush()
    
    print(f"\n{'='*60}")
    print(f"Building: {building_name}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Load ground truth
    try:
        print(f"DEBUG: Loading ground truth...")
        sys.stdout.flush()
        ground_truth_df = load_ground_truth(building_name, grid_size)
        print(f"Loaded {len(ground_truth_df)} images with ground truth")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR loading ground truth: {e}")
        sys.stdout.flush()
        return None, None
    
    print(f"DEBUG: Getting gridded floorplan path...")
    sys.stdout.flush()
    
    # Get gridded floorplan path
    floor_map_path = Config.get_gridded_floorplan_path(building_name, grid_size)
    
    if not os.path.exists(floor_map_path):
        print(f"ERROR: Gridded floorplan not found: {floor_map_path}")
        sys.stdout.flush()
        return None, None
    
    print(f"DEBUG: Getting prompt text...")
    sys.stdout.flush()
    
    # Get prompt
    if Config.PROMPT_TYPE == "zero_shot":
        prompt_text = get_zero_shot_prompt(grid_size, Config.TASK_VARIANT)
    else:
        prompt_text = get_few_shot_prompt(grid_size, Config.TASK_VARIANT)
    
    print(f"DEBUG: Setting up images directory...")
    sys.stdout.flush()
    
    # Get images directory
    building_dir = Config.get_building_source_dir(building_name)
    images_dir = os.path.join(building_dir, 'images')
    
    # Run inference
    predictions = []
    
    # Disable tqdm progress bar for background execution
    disable_tqdm = not sys.stdout.isatty()
    
    print(f"Starting inference on {len(ground_truth_df)} images...")
    sys.stdout.flush()
    
    for idx, row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df), 
                         desc=f"Processing {building_name}", leave=False, disable=disable_tqdm):
        image_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            sys.stdout.flush()
            continue
        
        try:
            # Run inference
            output_text = run_inference(model, processor, floor_map_path, image_path, prompt_text)
            
            # Parse response
            parsed = parse_response(output_text, Config.TASK_VARIANT)
            
            # Validate extraction
            missing_fields = validate_extraction(parsed, Config.TASK_VARIANT)
            if missing_fields and Config.DEBUG_MODE:
                print(f"Image {row['image_id']}: Missing fields: {', '.join(missing_fields)}")
                sys.stdout.flush()
            
            # Add debug output for first few samples
            if Config.DEBUG_MODE and len(predictions) < Config.DEBUG_SAMPLES:
                print(f"\n{'='*60}")
                print(f"DEBUG - Image {row['image_id']}")
                print(f"{'='*60}")
                print(f"Raw Response:\n{output_text[:500]}...")
                print(f"\nParsed:")
                print(f"  GRID_CELL: {parsed.get('grid_cell')}")
                print(f"  DIRECTION: {parsed.get('direction')}")
                print(f"  REASONING: {parsed.get('reasoning')[:100]}...")
                print(f"{'='*60}\n")
                sys.stdout.flush()
            
            parsed['image_id'] = row['image_id']
            parsed['ground_truth_cell'] = row['grid_cell']
            parsed['ground_truth_direction'] = row['direction']
            
            predictions.append(parsed)
            
            # Print progress every 10 images when tqdm is disabled
            if disable_tqdm and (len(predictions) % 10 == 0):
                print(f"  Processed {len(predictions)}/{len(ground_truth_df)} images...")
                sys.stdout.flush()
            
        except Exception as e:
            print(f"\nError processing {row['image_id']}: {e}")
            sys.stdout.flush()
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
    
    print(f"Completed inference on {len(predictions)} images")
    sys.stdout.flush()
    
    print(f"DEBUG: Calculating metrics...")
    sys.stdout.flush()
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, Config.TASK_VARIANT)
    
    return metrics, predictions


def calculate_metrics(predictions, task_variant):
    """Calculate evaluation metrics from predictions"""
    print(f"DEBUG: calculate_metrics(predictions_count={len(predictions)}, task_variant={task_variant})")
    sys.stdout.flush()
    
    metrics = {
        'total_samples': len(predictions),
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
    
    for pred in predictions:
        true_cell = pred['ground_truth_cell']
        true_dir = pred['ground_truth_direction']
        pred_cell = pred.get('grid_cell')
        pred_dir = pred.get('direction')
        
        # Check if response is parseable
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
    valid_samples = len(predictions) - metrics['unparseable_responses']
    
    if valid_samples > 0:
        metrics['grid_cell_accuracy'] = grid_correct / valid_samples
        metrics['direction_accuracy'] = direction_correct / valid_samples
        metrics['exact_match_accuracy'] = exact_matches / valid_samples
    
    if grid_distances:
        metrics['avg_grid_distance'] = float(np.mean(grid_distances))
        metrics['median_grid_distance'] = float(np.median(grid_distances))
        metrics['std_grid_distance'] = float(np.std(grid_distances))
    
    if direction_distances:
        metrics['avg_direction_distance'] = float(np.mean(direction_distances))
        metrics['median_direction_distance'] = float(np.median(direction_distances))
        metrics['std_direction_distance'] = float(np.std(direction_distances))
    
    print(f"DEBUG: Metrics calculation complete")
    sys.stdout.flush()
    
    return metrics


def save_building_results(building_name, grid_size, metrics, predictions):
    """Save results for a single building"""
    print(f"DEBUG: save_building_results({building_name}, {grid_size})")
    sys.stdout.flush()
    
    output_dir = Config.setup_output_dir(grid_size, building_name)
    
    model_config = Config.get_model_config()
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    full_metrics = {
        'building': building_name,
        'grid_size': grid_size,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'model_variant': Config.MODEL_VARIANT,
        'model_name': model_config['model_name'],
        'timestamp': Config.TIMESTAMP,
        'metrics': metrics
    }
    
    print(f"DEBUG: Writing metrics to {metrics_file}")
    sys.stdout.flush()
    
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'predictions.json')
    
    print(f"DEBUG: Writing predictions to {predictions_file}")
    sys.stdout.flush()
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\nResults saved for {building_name} (grid {grid_size}x{grid_size}):")
    print(f"  Metrics: {metrics_file}")
    print(f"  Predictions: {predictions_file}")
    sys.stdout.flush()
    
    return metrics_file, predictions_file


def save_combined_results(all_results):
    """
    Save combined results across all buildings and grid sizes
    all_results: dict[grid_size] -> list of (building_name, metrics)
    """
    print(f"DEBUG: save_combined_results(grid_sizes={list(all_results.keys())})")
    sys.stdout.flush()
    
    output_dir = Config.setup_output_dir()
    
    combined_file = os.path.join(output_dir, 'combined_results.json')
    
    # Organize results
    combined_data = {
        'timestamp': Config.TIMESTAMP,
        'model_variant': Config.MODEL_VARIANT,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'grid_sizes': {},
        'summary': {}
    }
    
    # Add results for each grid size
    for grid_size, building_results in all_results.items():
        grid_data = {
            'buildings': {},
            'aggregate_metrics': {}
        }
        
        # Collect metrics for aggregation
        all_metrics = {
            'grid_cell_accuracy': [],
            'direction_accuracy': [],
            'exact_match_accuracy': [],
            'avg_grid_distance': [],
            'avg_direction_distance': [],
            'unparseable_responses': []
        }
        
        total_samples = 0
        
        for building_name, metrics in building_results:
            grid_data['buildings'][building_name] = metrics
            
            # Aggregate
            total_samples += metrics['total_samples']
            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])
        
        # Calculate aggregate statistics
        grid_data['aggregate_metrics'] = {
            'total_buildings': len(building_results),
            'total_samples': total_samples,
            'mean_grid_cell_accuracy': float(np.mean(all_metrics['grid_cell_accuracy'])) if all_metrics['grid_cell_accuracy'] else 0,
            'mean_direction_accuracy': float(np.mean(all_metrics['direction_accuracy'])) if all_metrics['direction_accuracy'] else 0,
            'mean_exact_match_accuracy': float(np.mean(all_metrics['exact_match_accuracy'])) if all_metrics['exact_match_accuracy'] else 0,
            'mean_avg_grid_distance': float(np.mean(all_metrics['avg_grid_distance'])) if all_metrics['avg_grid_distance'] else 0,
            'mean_avg_direction_distance': float(np.mean(all_metrics['avg_direction_distance'])) if all_metrics['avg_direction_distance'] else 0,
            'total_unparseable_responses': sum(all_metrics['unparseable_responses'])
        }
        
        combined_data['grid_sizes'][f'grid_{grid_size}'] = grid_data
    
    print(f"DEBUG: Writing combined results to {combined_file}")
    sys.stdout.flush()
    
    # Save
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\nCombined results saved: {combined_file}")
    sys.stdout.flush()
    
    return combined_file


def print_summary(all_results):
    """Print summary of evaluation results"""
    print(f"DEBUG: print_summary()")
    sys.stdout.flush()
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    for grid_size, building_results in all_results.items():
        print(f"\nGrid Size: {grid_size}x{grid_size}")
        print(f"Buildings evaluated: {len(building_results)}")
        
        # Aggregate metrics
        all_grid_acc = [m['grid_cell_accuracy'] for _, m in building_results]
        all_dir_acc = [m['direction_accuracy'] for _, m in building_results]
        all_exact_acc = [m['exact_match_accuracy'] for _, m in building_results]
        
        print(f"Mean Grid Cell Accuracy: {np.mean(all_grid_acc):.2%}")
        print(f"Mean Direction Accuracy: {np.mean(all_dir_acc):.2%}")
        print(f"Mean Exact Match Accuracy: {np.mean(all_exact_acc):.2%}")
        sys.stdout.flush()


def main():
    # Override print with auto-flush version
    global print
    print = functools.partial(print, flush=True)

    print(f"DEBUG: main() starting")
    sys.stdout.flush()
    
    print(f"{'='*80}")
    print("QWEN3-VL EVALUATION")
    print(f"{'='*80}")
    
    model_config = Config.get_model_config()
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'SINGLE BUILDING' if Config.SINGLE_BUILDING_MODE else 'ALL BUILDINGS'}")
    print(f"  Model: {model_config['display_name']}")
    print(f"  Task Variant: {Config.TASK_VARIANT}")
    print(f"  Prompt Type: {Config.PROMPT_TYPE}")
    print(f"  Grid Sizes: {Config.GRID_SIZES}")
    print(f"  Timestamp: {Config.TIMESTAMP}")
    
    # Get buildings to process
    if Config.SINGLE_BUILDING_MODE:
        buildings_to_process = [Config.SINGLE_BUILDING_NAME]
        print(f"\nProcessing single building: {Config.SINGLE_BUILDING_NAME}")
    else:
        print(f"DEBUG: Getting all buildings...")
        valid_buildings, invalid_buildings = get_all_buildings()
        buildings_to_process = valid_buildings
        
        print(f"\nBuilding validation:")
        print(f"  Valid buildings: {len(valid_buildings)}")
        print(f"  Invalid buildings: {len(invalid_buildings)}")
        
        if invalid_buildings:
            print(f"\nInvalid buildings (missing items):")
            for building, missing in list(invalid_buildings.items())[:10]:
                print(f"  - {building}: {', '.join(missing)}")
            if len(invalid_buildings) > 10:
                print(f"  ... and {len(invalid_buildings) - 10} more")
    
    # Load model once
    print(f"\n{'='*80}")
    print(f"DEBUG: Loading model and processor...")
    model, processor = load_model_and_processor()
    
    # Process each grid size
    all_results = {}
    
    for grid_size in Config.GRID_SIZES:
        print(f"\n{'='*80}")
        print(f"PROCESSING GRID SIZE: {grid_size}x{grid_size}")
        print(f"{'='*80}")
        
        grid_results = []
        
        for building_name in buildings_to_process:
            print(f"DEBUG: Processing building: {building_name}")
            metrics, predictions = evaluate_building(building_name, grid_size, model, processor)
            
            if metrics is not None:
                # Save individual building results
                save_building_results(building_name, grid_size, metrics, predictions)
                grid_results.append((building_name, metrics))
                
                # Print building summary
                print(f"\n{building_name} Results:")
                print(f"  Total Samples: {metrics['total_samples']}")
                print(f"  Unparseable: {metrics['unparseable_responses']}")
                print(f"  Grid Cell Accuracy: {metrics['grid_cell_accuracy']:.2%}")
                print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        
        all_results[grid_size] = grid_results
    
    # Save combined results
    if not Config.SINGLE_BUILDING_MODE:
        print(f"DEBUG: Saving combined results...")
        save_combined_results(all_results)
    
    # Print final summary
    print_summary(all_results)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {Config.OUTPUT_BASE_PATH}/{Config.TIMESTAMP}")
    
    print(f"DEBUG: main() complete")


if __name__ == "__main__":
    main()