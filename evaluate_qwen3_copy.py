# evaluate_qwen3.py

import os
import sys
import functools
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from datetime import datetime
from tqdm import tqdm
import re
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

class Config:
    
    # Mode settings
    SINGLE_BUILDING_MODE = False  # True for single building, False for all buildings
    SINGLE_BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"

    SKIP_BUILDINGS = ["Lausanne-Cathedrale-Notre-Dame"]
    
    # Debug settings
    DEBUG_MODE = False  # Set to True to print first few responses
    DEBUG_SAMPLES = 3
    
    # Paths
    SOURCE_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output"  # Original building data
    GRIDDED_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/grids_floorplan"  # Gridded floorplans
    ARROWS_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/arrows_floorplan"  # Arrows floorplans
    OUTPUT_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/qwen_evaluation"  # Results
    
    # Grid configuration
    GRID_SIZES = [10, 15, 20]  # List of grid sizes to evaluate: [10, 15, 20]
    
    # Arrows configuration
    ARROW_COUNTS = [10, 15, 20]  # List of arrow counts to evaluate: [10, 15, 20]
    
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
            "display_name": "Qwen3-VL-32B-Thinking"
        },
        "qwen3-vl-8b-thinking": {
            "model_name": "Qwen/Qwen3-VL-8B-Thinking",
            "display_name": "Qwen3-VL-8B-Thinking"
        },
        "qwen3-vl-32b-instruct": {
            "model_name": "Qwen/Qwen3-VL-32B-Instruct",
            "display_name": "Qwen3-VL-32B-Instruct"
        },
        "qwen3-vl-8b-instruct": {
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "display_name": "Qwen3-VL-8B-Instruct"
        }
    }
    
    # Generation settings
    MAX_NEW_TOKENS = 512
    USE_GREEDY = True
    TEMPERATURE = 0.1     # Only used if USE_GREEDY=False
    TOP_P = 0.95          # Only used if USE_GREEDY=False
    
    # vLLM specific settings
    MAX_MODEL_LEN = 32768  # Context length for vLLM
    TENSOR_PARALLEL_SIZE = 1  # Number of GPUs for tensor parallelism
    
    # Environment setup
    HF_CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"
    CUDA_DEVICE = "7"
    
    # Runtime
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = None
    
    @classmethod
    def get_building_source_dir(cls, building_name):
        """Get source directory for building data"""
        return f"{cls.SOURCE_BASE_PATH}/{building_name}"
    
    @classmethod
    def get_gridded_floorplan_path(cls, building_name, grid_size):
        """Get path to gridded floorplan"""
        return f"{cls.GRIDDED_BASE_PATH}/grid_size_{grid_size}/{building_name}_floorplan_gridded.jpg"
    
    @classmethod
    def get_ground_truth_csv_path(cls, building_name, grid_size):
        """Get path to pre-computed ground truth CSV"""
        return f"{cls.GRIDDED_BASE_PATH}/grid_size_{grid_size}_ground_truth/{building_name}_ground_truth.csv"
    
    @classmethod
    def get_arrows_floorplan_path(cls, building_name, arrow_count):
        """Get path to arrows floorplan"""
        return f"{cls.ARROWS_BASE_PATH}/arrows_{arrow_count}/{building_name}/{building_name}_arrows_visualization.jpg"
    
    @classmethod
    def get_arrow_mapping_path(cls, building_name, arrow_count):
        """Get path to arrow label mapping JSON"""
        return f"{cls.ARROWS_BASE_PATH}/arrows_{arrow_count}/{building_name}/arrow_label_mapping.json"
    
    @classmethod
    def setup_output_dir(cls, size_param=None, building_name=None):
        """Setup output directory structure"""
        base_dir = f"{cls.OUTPUT_BASE_PATH}/{cls.TIMESTAMP}"
        
        if size_param is not None:
            if cls.TASK_VARIANT == "grid_direction":
                base_dir = f"{base_dir}/grid_size_{size_param}"
            else:  # labeled_arrows
                base_dir = f"{base_dir}/arrows_{size_param}"
        
        if building_name is not None:
            base_dir = f"{base_dir}/{building_name}"
        
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    @classmethod
    def get_model_config(cls):
        return cls.MODEL_CONFIGS[cls.MODEL_VARIANT]

# Set environment variables
os.environ['HF_HOME'] = Config.HF_CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_DEVICE


def generate_column_label(col_index, num_cols):
    """
    Generate column label for a given index (A, B, C, ..., Z, AA, AB, ...)
    Matches the logic in create_grid_overlay.py
    """
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


def get_arrow_label_range(arrow_count):
    """Get the range of arrow labels (e.g., 'A to J' for 10 arrows)"""
    last_label = chr(65 + arrow_count - 1)  # A=65, so for 10 arrows: chr(74)='J'
    return f"A to {last_label}"


def validate_building_folder(building_name):
    """
    Validate that a building folder has all required files
    Returns (is_valid, missing_items)
    """
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


def validate_building_for_grid(building_name, grid_size):
    """
    Validate that a building has all required files for a specific grid size
    Returns (is_valid, reason)
    """
    # Check gridded floorplan
    floor_map_path = Config.get_gridded_floorplan_path(building_name, grid_size)
    if not os.path.exists(floor_map_path):
        return False, "missing_gridded_floorplan"
    
    # Check ground truth CSV
    ground_truth_path = Config.get_ground_truth_csv_path(building_name, grid_size)
    if not os.path.exists(ground_truth_path):
        return False, "missing_ground_truth_csv"
    
    # Check images folder exists and is not empty
    building_dir = Config.get_building_source_dir(building_name)
    images_dir = os.path.join(building_dir, 'images')
    
    if not os.path.exists(images_dir):
        return False, "missing_images_folder"
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        return False, "empty_images_folder"
    
    return True, None


def validate_building_for_arrows(building_name, arrow_count):
    """
    Validate that a building has all required files for arrows evaluation
    Returns (is_valid, reason)
    """
    # Check arrows floorplan
    arrows_floor_path = Config.get_arrows_floorplan_path(building_name, arrow_count)
    if not os.path.exists(arrows_floor_path):
        return False, "missing_arrows_floorplan"
    
    # Check arrow mapping JSON
    arrow_mapping_path = Config.get_arrow_mapping_path(building_name, arrow_count)
    if not os.path.exists(arrow_mapping_path):
        return False, "missing_arrow_mapping_json"
    
    # Check images folder exists and is not empty
    building_dir = Config.get_building_source_dir(building_name)
    images_dir = os.path.join(building_dir, 'images')
    
    if not os.path.exists(images_dir):
        return False, "missing_images_folder"
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        return False, "empty_images_folder"
    
    return True, None


def get_all_buildings():
    """
    Get list of all building folders and validate them
    Returns (valid_buildings, invalid_buildings_info)
    """
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
    ground_truth_csv_path = Config.get_ground_truth_csv_path(building_name, grid_size)
    
    if not os.path.exists(ground_truth_csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv_path}")
    
    # Load pre-computed ground truth
    df = pd.read_csv(ground_truth_csv_path)
    
    # Validate required columns
    required_columns = ['image_id', 'grid_cell', 'direction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
    
    return df


def load_arrow_mapping(building_name, arrow_count):
    """Load arrow label mapping JSON for arrows evaluation"""
    arrow_mapping_path = Config.get_arrow_mapping_path(building_name, arrow_count)
    
    if not os.path.exists(arrow_mapping_path):
        raise FileNotFoundError(f"Arrow mapping JSON not found: {arrow_mapping_path}")
    
    with open(arrow_mapping_path, 'r') as f:
        arrow_mapping = json.load(f)
    
    # Validate structure
    for label, info in arrow_mapping.items():
        required_keys = ['image_id', 'x', 'y', 'direction']
        missing_keys = [key for key in required_keys if key not in info]
        if missing_keys:
            raise ValueError(f"Arrow mapping for label {label} missing keys: {missing_keys}")
    
    return arrow_mapping


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


def get_zero_shot_prompt(size_param, task_variant):
    """Generate zero-shot prompt with Gothic architecture context"""
    if task_variant == "grid_direction":
        grid_size = size_param
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
        arrow_count = size_param
        label_range = get_arrow_label_range(arrow_count)
        
        return f"""You are analyzing a Gothic church floor plan with labeled arrows and an interior photograph from 12th-13th century France.

ARCHITECTURAL CONTEXT:
This is a Gothic church building with typical features including:
- Nave (central space), side aisles, transept (crossing), choir, apse
- Columns/pillars supporting pointed arches and ribbed vaults
- Large windows (often with tracery), clerestory windows
- Ambulatory (walkway around choir/apse in some churches)
- Stone walls, vaulted ceilings, decorative capitals

TASK: Identify which labeled arrow corresponds to the photograph's location and viewing direction.

ARROW INFORMATION:
- The floor plan contains {arrow_count} labeled arrows
- Arrow labels range from {label_range}
- Each arrow shows a specific position and viewing direction within the church

ANALYSIS STEPS:
1. IDENTIFY visible Gothic architectural features in the photograph:
   - Column/pillar positions and their spacing
   - Pointed arches and vault configurations
   - Window locations, sizes, and orientations
   - Wall positions and openings
   - Spatial arrangement (nave, aisle, transept, choir, etc.)

2. LOCATE these features on the floor plan with arrows:
   - Match column positions shown as dots/circles on plan
   - Identify window locations along walls
   - Find the corresponding section (nave, aisle, crossing, choir, apse)
   - Match the spatial layout pattern

3. MATCH to the correct arrow:
   - Find the arrow that points from a position matching where the photo was taken
   - Verify the arrow's direction matches what would be visible in the photograph
   - Consider both the location AND the viewing direction

IMPORTANT CONSTRAINTS:
- Only analyze architectural features CLEARLY VISIBLE in the photograph
- Each arrow represents a specific viewpoint - both position and direction matter
- Gothic churches typically have east-west orientation
- Do not assume features not visible in the photo
- Base your answer on concrete architectural elements you can identify
- The answer must be one of the {arrow_count} labeled arrows ({label_range})
- If uncertain, provide your best estimate but acknowledge this in reasoning

OUTPUT FORMAT (MANDATORY - YOU MUST PROVIDE BOTH FIELDS):
REASONING: [2-3 sentences identifying key Gothic features and matching them to an arrow]
ARROW_LABEL: [Single letter from {label_range}]

EXAMPLE OUTPUT:
REASONING: The photograph shows three cylindrical columns on the left supporting pointed arches, with a large Gothic window with tracery visible on the right. Arrow C in the north aisle matches this exact column spacing and eastward viewing direction.
ARROW_LABEL: C

Now analyze the provided floor plan and photograph. You MUST provide both fields (REASONING, ARROW_LABEL) even if uncertain."""


def get_few_shot_prompt(size_param, task_variant):
    """Generate few-shot prompt with Gothic architecture examples"""
    if task_variant == "grid_direction":
        grid_size = size_param
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
        arrow_count = size_param
        label_range = get_arrow_label_range(arrow_count)
        
        return f"""You are analyzing a Gothic church floor plan with labeled arrows and an interior photograph from 12th-13th century France.

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

ARROW INFORMATION:
- The floor plan contains {arrow_count} labeled arrows ({label_range})
- Each arrow shows position and viewing direction
- Match the photo to the correct arrow

TASK: Identify which arrow corresponds to the photo location and direction.

ANALYSIS STEPS:
1. Identify visible Gothic features in photo
2. Match to arrow positions on floor plan
3. Find arrow with matching location and viewing direction

OUTPUT FORMAT (MANDATORY):
REASONING: [2-3 sentences on Gothic features and arrow match]
ARROW_LABEL: [Letter from {label_range}]

You MUST provide both fields even if uncertain."""


def load_vllm_model(processor):
    model_config = Config.get_model_config()
    model_name = model_config["model_name"]
    
    print(f"\nLoading {model_config['display_name']} with vLLM...")
    print(f"Model path: {model_name}")
    print(f"Cache directory: {Config.HF_CACHE_DIR}")
    sys.stdout.flush()
    
    # Load model with vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        max_model_len=Config.MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": 2},  # 2 images: floor plan + photo
    )
    
    print(f"Model loaded successfully with vLLM")
    sys.stdout.flush()
    
    return llm


def prepare_vllm_request(processor, floor_plan_image, photo_image, prompt_text):
    """Prepare a single request for vLLM batch inference"""
    # Prepare messages with image placeholders
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for floor plan
                {"type": "image"},  # Placeholder for photo
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # Apply chat template to get prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": [floor_plan_image, photo_image]},
    }


def extract_structured_answer(response_text, task_variant):
    """Extract answer using multiple strategies with robust parsing"""
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
    return extract_structured_answer(response_text, task_variant)


def evaluate_building(building_name, size_param, llm, processor):
    """
    Evaluate a single building with a specific size (grid or arrow) using vLLM batching
    Returns (metrics, predictions_list) or (None, None) if error
    """
    print(f"\n{'='*60}")
    print(f"Building: {building_name}")
    if Config.TASK_VARIANT == "grid_direction":
        print(f"Grid Size: {size_param}x{size_param}")
    else:
        print(f"Arrow Count: {size_param}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Load ground truth based on task variant
    try:
        if Config.TASK_VARIANT == "grid_direction":
            ground_truth_data = load_ground_truth(building_name, size_param)
            print(f"Loaded {len(ground_truth_data)} images with ground truth")
        else:  # labeled_arrows
            ground_truth_data = load_arrow_mapping(building_name, size_param)
            print(f"Loaded {len(ground_truth_data)} arrow labels with ground truth")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR loading ground truth: {e}")
        sys.stdout.flush()
        return None, None
    
    # Get floorplan path based on task variant
    if Config.TASK_VARIANT == "grid_direction":
        floor_map_path = Config.get_gridded_floorplan_path(building_name, size_param)
    else:  # labeled_arrows
        floor_map_path = Config.get_arrows_floorplan_path(building_name, size_param)
    
    if not os.path.exists(floor_map_path):
        print(f"ERROR: Floorplan not found: {floor_map_path}")
        sys.stdout.flush()
        return None, None
    
    # Load floor plan image once
    floor_plan_image = Image.open(floor_map_path).convert("RGB")
    
    # Get prompt
    if Config.PROMPT_TYPE == "zero_shot":
        prompt_text = get_zero_shot_prompt(size_param, Config.TASK_VARIANT)
    else:
        prompt_text = get_few_shot_prompt(size_param, Config.TASK_VARIANT)
    
    # Get images directory
    building_dir = Config.get_building_source_dir(building_name)
    images_dir = os.path.join(building_dir, 'images')
    
    # Prepare all requests for batch inference
    print(f"Preparing batch requests...")
    sys.stdout.flush()
    
    batch_requests = []
    request_metadata = []  # Store metadata for each request
    
    if Config.TASK_VARIANT == "grid_direction":
        # ground_truth_data is a DataFrame
        for idx, row in ground_truth_data.iterrows():
            image_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                sys.stdout.flush()
                continue
            
            try:
                # Load photo image
                photo_image = Image.open(image_path).convert("RGB")
                
                # Prepare request
                request = prepare_vllm_request(
                    processor, floor_plan_image, photo_image, prompt_text
                )
                
                batch_requests.append(request)
                request_metadata.append({
                    'image_id': row['image_id'],
                    'ground_truth_cell': row['grid_cell'],
                    'ground_truth_direction': row['direction']
                })
                
            except Exception as e:
                print(f"Error preparing request for {row['image_id']}: {e}")
                sys.stdout.flush()
    
    else:  # labeled_arrows
        # ground_truth_data is a dict (arrow_mapping)
        for label, info in ground_truth_data.items():
            image_path = os.path.join(images_dir, f"{info['image_id']}.jpg")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                sys.stdout.flush()
                continue
            
            try:
                # Load photo image
                photo_image = Image.open(image_path).convert("RGB")
                
                # Prepare request
                request = prepare_vllm_request(
                    processor, floor_plan_image, photo_image, prompt_text
                )
                
                batch_requests.append(request)
                request_metadata.append({
                    'image_id': info['image_id'],
                    'ground_truth_label': label
                })
                
            except Exception as e:
                print(f"Error preparing request for {info['image_id']}: {e}")
                sys.stdout.flush()
    
    if not batch_requests:
        print("ERROR: No valid requests prepared")
        sys.stdout.flush()
        return None, None
    
    print(f"Prepared {len(batch_requests)} requests for batch inference")
    sys.stdout.flush()
    
    # Prepare sampling params
    if Config.USE_GREEDY:
        sampling_params = SamplingParams(
            max_tokens=Config.MAX_NEW_TOKENS,
            temperature=0.0,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
        )
    
    # Run batch inference
    print(f"Running batch inference on {len(batch_requests)} images...")
    sys.stdout.flush()
    
    try:
        outputs = llm.generate(
            batch_requests,
            sampling_params=sampling_params,
        )
    except Exception as e:
        print(f"ERROR during batch inference: {e}")
        sys.stdout.flush()
        return None, None
    
    print(f"Batch inference completed")
    sys.stdout.flush()
    
    # Process outputs
    predictions = []
    
    for output, metadata in zip(outputs, request_metadata):
        output_text = output.outputs[0].text if output.outputs else ""
        
        # Parse response
        parsed = parse_response(output_text, Config.TASK_VARIANT)
        
        # Validate extraction
        missing_fields = validate_extraction(parsed, Config.TASK_VARIANT)
        if missing_fields and Config.DEBUG_MODE:
            print(f"Image {metadata['image_id']}: Missing fields: {', '.join(missing_fields)}")
            sys.stdout.flush()
        
        # Add debug output for first few samples
        if Config.DEBUG_MODE and len(predictions) < Config.DEBUG_SAMPLES:
            print(f"\n{'='*60}")
            print(f"DEBUG - Image {metadata['image_id']}")
            print(f"{'='*60}")
            print(f"Raw Response:\n{output_text[:500]}...")
            print(f"\nParsed:")
            if Config.TASK_VARIANT == "grid_direction":
                print(f"  GRID_CELL: {parsed.get('grid_cell')}")
                print(f"  DIRECTION: {parsed.get('direction')}")
            else:
                print(f"  ARROW_LABEL: {parsed.get('arrow_label')}")
            print(f"  REASONING: {parsed.get('reasoning')[:100]}...")
            print(f"{'='*60}\n")
            sys.stdout.flush()
        
        # Add metadata to parsed result
        parsed['image_id'] = metadata['image_id']
        
        if Config.TASK_VARIANT == "grid_direction":
            parsed['ground_truth_cell'] = metadata['ground_truth_cell']
            parsed['ground_truth_direction'] = metadata['ground_truth_direction']
        else:
            parsed['ground_truth_label'] = metadata['ground_truth_label']
        
        predictions.append(parsed)
    
    print(f"Completed processing {len(predictions)} predictions")
    sys.stdout.flush()
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, Config.TASK_VARIANT)
    
    return metrics, predictions


def calculate_metrics(predictions, task_variant):
    """Calculate evaluation metrics from predictions (works for both grids and arrows)"""
    if task_variant == "grid_direction":
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
            true_cell = pred.get('ground_truth_cell')
            true_dir = pred.get('ground_truth_direction')
            pred_cell = pred.get('grid_cell')
            pred_dir = pred.get('direction')
            
            if not true_cell or not true_dir:
                continue
            
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
    
    else:  # labeled_arrows
        metrics = {
            'total_samples': len(predictions),
            'label_accuracy': 0,
            'unparseable_responses': 0
        }
        
        label_correct = 0
        
        for pred in predictions:
            true_label = pred.get('ground_truth_label')
            pred_label = pred.get('arrow_label')
            
            if not true_label:
                continue
            
            # Check if response is parseable
            if not pred_label:
                metrics['unparseable_responses'] += 1
                continue
            
            # Calculate metrics
            if pred_label == true_label:
                label_correct += 1
        
        # Calculate final metrics
        valid_samples = len(predictions) - metrics['unparseable_responses']
        
        if valid_samples > 0:
            metrics['label_accuracy'] = label_correct / valid_samples
    
    return metrics


def save_building_results(building_name, size_param, metrics, predictions):
    """Save results for a single building (works for both grids and arrows)"""
    output_dir = Config.setup_output_dir(size_param, building_name)
    
    model_config = Config.get_model_config()
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    full_metrics = {
        'building': building_name,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'model_variant': Config.MODEL_VARIANT,
        'model_name': model_config['model_name'],
        'timestamp': Config.TIMESTAMP,
        'metrics': metrics
    }
    
    if Config.TASK_VARIANT == "grid_direction":
        full_metrics['grid_size'] = size_param
    else:
        full_metrics['arrow_count'] = size_param
    
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'predictions.json')
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    if Config.TASK_VARIANT == "grid_direction":
        print(f"\nResults saved for {building_name} (grid {size_param}x{size_param}):")
    else:
        print(f"\nResults saved for {building_name} (arrows {size_param}):")
    print(f"  Metrics: {metrics_file}")
    print(f"  Predictions: {predictions_file}")
    sys.stdout.flush()
    
    return metrics_file, predictions_file


def save_combined_results(all_results):
    """
    Save combined results across all buildings and sizes (works for both grids and arrows)
    all_results: dict[size_param] -> list of (building_name, metrics)
    """
    output_dir = Config.setup_output_dir()
    
    combined_file = os.path.join(output_dir, 'combined_results.json')
    
    # Organize results
    combined_data = {
        'timestamp': Config.TIMESTAMP,
        'model_variant': Config.MODEL_VARIANT,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'summary': {}
    }
    
    if Config.TASK_VARIANT == "grid_direction":
        combined_data['grid_sizes'] = {}
        
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
    
    else:  # labeled_arrows
        combined_data['arrow_counts'] = {}
        
        # Add results for each arrow count
        for arrow_count, building_results in all_results.items():
            arrow_data = {
                'buildings': {},
                'aggregate_metrics': {}
            }
            
            # Collect metrics for aggregation
            all_metrics = {
                'label_accuracy': [],
                'unparseable_responses': []
            }
            
            total_samples = 0
            
            for building_name, metrics in building_results:
                arrow_data['buildings'][building_name] = metrics
                
                # Aggregate
                total_samples += metrics['total_samples']
                for key in all_metrics.keys():
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
            
            # Calculate aggregate statistics
            arrow_data['aggregate_metrics'] = {
                'total_buildings': len(building_results),
                'total_samples': total_samples,
                'mean_label_accuracy': float(np.mean(all_metrics['label_accuracy'])) if all_metrics['label_accuracy'] else 0,
                'total_unparseable_responses': sum(all_metrics['unparseable_responses'])
            }
            
            combined_data['arrow_counts'][f'arrows_{arrow_count}'] = arrow_data
    
    # Save
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\nCombined results saved: {combined_file}")
    sys.stdout.flush()
    
    return combined_file


def print_summary(all_results):
    """Print summary of evaluation results (works for both grids and arrows)"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    if Config.TASK_VARIANT == "grid_direction":
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
    
    else:  # labeled_arrows
        for arrow_count, building_results in all_results.items():
            print(f"\nArrow Count: {arrow_count}")
            print(f"Buildings evaluated: {len(building_results)}")
            
            # Aggregate metrics
            all_label_acc = [m['label_accuracy'] for _, m in building_results]
            
            print(f"Mean Label Accuracy: {np.mean(all_label_acc):.2%}")
            sys.stdout.flush()


def main():
    
    # Override print with auto-flush version
    global print
    print = functools.partial(print, flush=True)
    
    print(f"{'='*80}")
    print("QWEN3-VL EVALUATION (with vLLM)")
    print(f"{'='*80}")
    
    model_config = Config.get_model_config()
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'SINGLE BUILDING' if Config.SINGLE_BUILDING_MODE else 'ALL BUILDINGS'}")
    print(f"  Model: {model_config['display_name']}")
    print(f"  Task Variant: {Config.TASK_VARIANT}")
    print(f"  Prompt Type: {Config.PROMPT_TYPE}")
    
    if Config.TASK_VARIANT == "grid_direction":
        print(f"  Grid Sizes: {Config.GRID_SIZES}")
    else:
        print(f"  Arrow Counts: {Config.ARROW_COUNTS}")
    
    print(f"  Timestamp: {Config.TIMESTAMP}")
    print(f"  Using vLLM with batch inference")
    
    # Get buildings to process
    if Config.SINGLE_BUILDING_MODE:
        buildings_to_process = [Config.SINGLE_BUILDING_NAME]
        print(f"\nProcessing single building: {Config.SINGLE_BUILDING_NAME}")
    else:
        valid_buildings, invalid_buildings = get_all_buildings()
        buildings_to_process = valid_buildings

        # Filter out skip buildings
        buildings_to_process = [b for b in buildings_to_process if b not in Config.SKIP_BUILDINGS]
        
        print(f"\nBuilding validation:")
        print(f"  Valid buildings: {len(valid_buildings)}")
        print(f"  Skipped buildings: {len(Config.SKIP_BUILDINGS)}")
        print(f"  Invalid buildings: {len(invalid_buildings)}")
        
        if invalid_buildings:
            print(f"\nInvalid buildings (missing items):")
            for building, missing in list(invalid_buildings.items())[:10]:
                print(f"  - {building}: {', '.join(missing)}")
            if len(invalid_buildings) > 10:
                print(f"  ... and {len(invalid_buildings) - 10} more")
    
    # Load processor (needed for prompt formatting)
    print(f"\n{'='*80}")
    print("Loading processor...")
    model_config = Config.get_model_config()
    processor = AutoProcessor.from_pretrained(
        model_config["model_name"],
        trust_remote_code=True,
        cache_dir=Config.HF_CACHE_DIR
    )
    print("Processor loaded")
    
    # Load vLLM model once
    print(f"\n{'='*80}")
    llm = load_vllm_model(processor)
    
    # Process each size (grid or arrow)
    all_results = {}
    skip_summary = {}
    
    # Get size list based on task variant
    if Config.TASK_VARIANT == "grid_direction":
        size_list = Config.GRID_SIZES
        size_name = "grid size"
    else:
        size_list = Config.ARROW_COUNTS
        size_name = "arrow count"
    
    for size_param in size_list:
        print(f"\n{'='*80}")
        if Config.TASK_VARIANT == "grid_direction":
            print(f"PROCESSING GRID SIZE: {size_param}x{size_param}")
        else:
            print(f"PROCESSING ARROW COUNT: {size_param}")
        print(f"{'='*80}")
        
        # Validate buildings for this size
        print(f"\nValidating buildings for {size_name} {size_param}...")
        valid_for_size = []
        skipped_reasons = {}
        
        for building_name in buildings_to_process:
            if Config.TASK_VARIANT == "grid_direction":
                is_valid, reason = validate_building_for_grid(building_name, size_param)
            else:
                is_valid, reason = validate_building_for_arrows(building_name, size_param)
            
            if is_valid:
                valid_for_size.append(building_name)
            else:
                skipped_reasons[building_name] = reason
        
        print(f"  Valid: {len(valid_for_size)}")
        print(f"  Skipped: {len(skipped_reasons)}")
        
        if skipped_reasons:
            # Count skip reasons
            reason_counts = {}
            for reason in skipped_reasons.values():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            print(f"\n  Skip reasons:")
            for reason, count in reason_counts.items():
                print(f"    {reason}: {count}")
            
            # Store for final summary
            skip_summary[size_param] = skipped_reasons
        
        size_results = []
        
        for building_name in valid_for_size:
            metrics, predictions = evaluate_building(building_name, size_param, llm, processor)
            
            if metrics is not None:
                # Save individual building results
                save_building_results(building_name, size_param, metrics, predictions)
                size_results.append((building_name, metrics))
                
                # Print building summary
                print(f"\n{building_name} Results:")
                print(f"  Total Samples: {metrics['total_samples']}")
                print(f"  Unparseable: {metrics['unparseable_responses']}")
                
                if Config.TASK_VARIANT == "grid_direction":
                    print(f"  Grid Cell Accuracy: {metrics['grid_cell_accuracy']:.2%}")
                    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
                else:  # labeled_arrows
                    print(f"  Label Accuracy: {metrics['label_accuracy']:.2%}")
        
        all_results[size_param] = size_results
    
    # Save combined results
    if not Config.SINGLE_BUILDING_MODE:
        save_combined_results(all_results)
    
    # Print final summary
    print_summary(all_results)
    
    # Print skip summary
    if skip_summary:
        print(f"\n{'='*80}")
        print("SKIPPED BUILDINGS SUMMARY")
        print(f"{'='*80}")
        for size_param, skipped in skip_summary.items():
            if Config.TASK_VARIANT == "grid_direction":
                print(f"\nGrid Size {size_param}x{size_param}: {len(skipped)} skipped")
            else:
                print(f"\nArrow Count {size_param}: {len(skipped)} skipped")
            
            # Group by reason
            by_reason = {}
            for building, reason in skipped.items():
                by_reason.setdefault(reason, []).append(building)
            
            for reason, buildings in by_reason.items():
                print(f"  {reason}: {len(buildings)}")
                if Config.DEBUG_MODE and len(buildings) <= 5:
                    for b in buildings:
                        print(f"    - {b}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {Config.OUTPUT_BASE_PATH}/{Config.TIMESTAMP}")


if __name__ == "__main__":
    main()