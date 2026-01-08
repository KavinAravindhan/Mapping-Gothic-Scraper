# evaluate_gpt5.py

import os
import sys
import functools
import json
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import argparse

class Config:
    
    # Mode settings
    SINGLE_BUILDING_MODE = False  # True for single building, False for all buildings
    SINGLE_BUILDING_NAME = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    
    # Debug settings
    DEBUG_MODE = False  # Set to True to print batch info

    DRY_RUN = False  # Set by command line
    
    # Paths
    SOURCE_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output"  # Original building data
    GRIDDED_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/grids_floorplan"  # Gridded floorplans
    OUTPUT_BASE_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/gpt_evaluation"  # Results
    
    # Grid configuration
    GRID_SIZES = [10, 15, 20]  # List of grid sizes to evaluate: [10, 15, 20]
    
    # Task variant: "grid_direction" or "labeled_arrows"
    TASK_VARIANT = "grid_direction"
    
    # Prompt type: "zero_shot" or "few_shot"
    PROMPT_TYPE = "zero_shot"
    
    # Model Options
    MODEL_VARIANT = "gpt-5-mini"  # Options: "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano"
    
    # Model configuration
    MODEL_CONFIGS = {
        "gpt-5.1": {
            "model_name": "gpt-5.1",
            "display_name": "GPT-5.1",
            "reasoning_effort": "low",
            "verbosity": "medium"
        },
        "gpt-5": {
            "model_name": "gpt-5",
            "display_name": "GPT-5",
            "reasoning_effort": "low",
            "verbosity": "medium"
        },
        "gpt-5-mini": {
            "model_name": "gpt-5-mini",
            "display_name": "GPT-5-Mini",
            "reasoning_effort": "low",
            "verbosity": "low"
        },
        "gpt-5-nano": {
            "model_name": "gpt-5-nano",
            "display_name": "GPT-5-Nano",
            "reasoning_effort": "low",
            "verbosity": "low"
        }
    }
    
    # API settings
    OPENAI_API_KEY = None  # Will be loaded from .env
    MAX_OUTPUT_TOKENS = 512  # Matched to Qwen's MAX_NEW_TOKENS
    
    # Batch settings
    COMPLETION_WINDOW = "24h"
    BATCH_CHECK_INTERVAL = 60  # seconds
    
    # Runtime
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def load_env(cls):
        """Load environment variables from .env file"""
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip().strip('"').strip("'")  # Remove quotes
                        os.environ[key.strip()] = value
        
        cls.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")
    
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
    def setup_output_dir(cls, grid_size=None, building_name=None):
        """Setup output directory structure"""
        base_dir = f"{cls.OUTPUT_BASE_PATH}/{cls.TIMESTAMP}"
        
        if grid_size is not None:
            base_dir = f"{base_dir}/grid_size_{grid_size}"
        
        if building_name is not None:
            base_dir = f"{base_dir}/{building_name}"
        
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    @classmethod
    def get_model_config(cls):
        return cls.MODEL_CONFIGS[cls.MODEL_VARIANT]

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-5 Spatial Localization Evaluation")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Prepare batches but do not submit to API')
    return parser.parse_args()

# ==================== HELPER FUNCTIONS ====================

def generate_column_label(col_index, num_cols):
    """Generate column label for a given index (A, B, C, ..., Z, AA, AB, ...)"""
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
    """Validate that a building folder has all required files"""
    building_dir = Config.get_building_source_dir(building_name)
    missing_items = []
    
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
    """Get list of all building folders and validate them"""
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


def load_ground_truth(building_name, grid_size):
    """Load pre-computed ground truth CSV for specific grid size"""
    ground_truth_csv_path = Config.get_ground_truth_csv_path(building_name, grid_size)
    
    if not os.path.exists(ground_truth_csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv_path}")
    
    df = pd.read_csv(ground_truth_csv_path)
    
    required_columns = ['image_id', 'grid_cell', 'direction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
    
    return df


def encode_image_to_base64(image_path):
    """Encode image to base64 string for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ==================== PROMPTS - MATCHING QWEN ====================

def get_zero_shot_prompt(grid_size, task_variant):
    """Generate zero-shot prompt with Gothic architecture context (matches Qwen)"""
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
- FLOOR PLAN ORIENTATION: The floor plan is oriented with North at the top, South at the bottom, West on the left, and East on the right
- Grid directions (N, NE, E, SE, S, SW, W, NW) refer to the floor plan's coordinate system, not real-world cardinal directions
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
    """Generate few-shot prompt with Gothic architecture examples (matches Qwen)"""
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


# ==================== BATCH API FUNCTIONS ====================

def prepare_batch_requests_for_building(building_name, grid_size, ground_truth_df):
    """Prepare batch requests for a single building"""
    building_dir = Config.get_building_source_dir(building_name)
    floor_map_path = Config.get_gridded_floorplan_path(building_name, grid_size)
    
    if not os.path.exists(floor_map_path):
        raise FileNotFoundError(f"Gridded floorplan not found: {floor_map_path}")
    
    # Encode floor map once (shared across all images in this building)
    floor_map_base64 = encode_image_to_base64(floor_map_path)
    
    # Get prompt
    if Config.PROMPT_TYPE == "zero_shot":
        prompt_text = get_zero_shot_prompt(grid_size, Config.TASK_VARIANT)
    else:
        prompt_text = get_few_shot_prompt(grid_size, Config.TASK_VARIANT)
    
    # Get model config
    model_config = Config.get_model_config()
    
    # Prepare requests
    batch_requests = []
    images_dir = os.path.join(building_dir, 'images')
    
    for idx, row in ground_truth_df.iterrows():
        image_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            sys.stdout.flush()
            continue
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Responses API format
        request = {
            "custom_id": f"{building_name}|{row['image_id']}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_config['model_name'],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt_text
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{floor_map_base64}"
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        ]
                    }
                ],
                "reasoning": {
                    "effort": model_config['reasoning_effort']
                },
                "text": {
                    "verbosity": model_config['verbosity']
                },
                "max_output_tokens": Config.MAX_OUTPUT_TOKENS
            }
        }
        
        batch_requests.append(request)
    
    return batch_requests


def save_batch_file(requests, output_dir, building_name, grid_size, chunk_idx=1):
    """Save batch requests to JSONL file"""
    filename = f"batch_{building_name}_grid{grid_size}_chunk{chunk_idx}.jsonl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    if Config.DEBUG_MODE:
        print(f"Saved {len(requests)} requests to {filepath}")
        sys.stdout.flush()
    
    return filepath

def submit_batch(batch_filepath, building_name, grid_size):
    """Submit batch file to OpenAI API"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    print(f"Submitting batch for {building_name} (grid {grid_size}x{grid_size})...")
    sys.stdout.flush()
    
    # Upload file
    with open(batch_filepath, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose='batch'
        )
    
    # Create batch
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint='/v1/responses',
        completion_window=Config.COMPLETION_WINDOW
    )
    
    print(f"Batch submitted: {batch.id}")
    sys.stdout.flush()
    
    return batch.id


def check_batch_status(batch_id):
    """Check status of a batch job"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    batch = client.batches.retrieve(batch_id)
    return batch

def get_batch_errors(batch_id):
    """Get error details from a failed batch"""
    from openai import OpenAI
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    batch = client.batches.retrieve(batch_id)
    
    if hasattr(batch, 'errors') and batch.errors:
        print(f"Batch errors: {batch.errors}")
    
    return batch


def wait_for_batch_completion(batch_id, building_name):
    """Wait for batch to complete with progress updates"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    print(f"Waiting for batch {batch_id} ({building_name}) to complete...")
    sys.stdout.flush()
    
    while True:
        batch = client.batches.retrieve(batch_id)
        
        if batch.status == 'completed':
            print(f"Batch {batch_id} completed")
            sys.stdout.flush()
            return batch
        elif batch.status in ['failed', 'expired', 'cancelled']:
            print(f"Batch {batch_id} {batch.status}")
            get_batch_errors(batch_id)
            sys.stdout.flush()
            return None
        
        # Print progress if available
        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            print(f"Progress: {counts.completed}/{counts.total} completed")
            sys.stdout.flush()
        
        time.sleep(Config.BATCH_CHECK_INTERVAL)


def download_batch_results(batch_id, output_dir):
    """Download results from a completed batch"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != 'completed':
        print(f"Batch not completed. Status: {batch.status}")
        sys.stdout.flush()
        return None
    
    # Download results
    result_file_id = batch.output_file_id
    result_content = client.files.content(result_file_id)
    
    # Save to file
    output_filename = f"batch_results_{batch_id}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'wb') as f:
        f.write(result_content.content)
    
    print(f"Results downloaded: {output_path}")
    sys.stdout.flush()
    
    return output_path


# ==================== RESPONSE PARSING ====================

def extract_structured_answer(response_text, task_variant):
    """Extract answer using multiple strategies with robust parsing (matches Qwen)"""
    import re
    
    result = {
        'reasoning': '',
        'grid_cell': None,
        'direction': None,
        'arrow_label': None,
        'raw_response': response_text
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
                    if row_num >= 1:
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


def process_batch_results(results_file, ground_truth_df):
    """Process results from batch API"""
    predictions = []
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            
            # Extract custom_id: "building_name|image_id"
            custom_id = result['custom_id']
            parts = custom_id.split('|')
            if len(parts) != 2:
                continue
            
            building_name, image_id = parts
            
            # Extract response
            if result['response']['status_code'] == 200:
                body = result['response']['body']
                
                # Handle Responses API output format
                response_text = ""
                if 'output' in body:
                    for output_item in body['output']:
                        if output_item.get('type') == 'message':
                            for content_item in output_item.get('content', []):
                                if content_item.get('type') == 'output_text':
                                    response_text += content_item.get('text', '')
                elif 'choices' in body:
                    response_text = body['choices'][0]['message']['content']
                else:
                    response_text = str(body)
                
                parsed = extract_structured_answer(response_text, Config.TASK_VARIANT)
                parsed['image_id'] = image_id
                parsed['custom_id'] = custom_id
                
                # Add ground truth
                gt_row = ground_truth_df[ground_truth_df['image_id'] == image_id]
                if not gt_row.empty:
                    parsed['ground_truth_cell'] = gt_row.iloc[0]['grid_cell']
                    parsed['ground_truth_direction'] = gt_row.iloc[0]['direction']
                
                predictions.append(parsed)
            else:
                print(f"Error for {custom_id}: {result['response']['status_code']}")
                sys.stdout.flush()
    
    return predictions


# ==================== METRICS CALCULATION ====================

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


def calculate_metrics(predictions):
    """Calculate evaluation metrics from predictions"""
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
    
    return metrics


def save_building_results(building_name, grid_size, metrics, predictions):
    """Save results for a single building"""
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
    
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'predictions.json')
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Results saved for {building_name} (grid {grid_size}x{grid_size})")
    print(f"  Metrics: {metrics_file}")
    print(f"  Predictions: {predictions_file}")
    sys.stdout.flush()
    
    return metrics_file, predictions_file


def save_combined_results(all_results):
    """Save combined results across all buildings and grid sizes"""
    output_dir = Config.setup_output_dir()
    
    combined_file = os.path.join(output_dir, 'combined_results.json')
    
    combined_data = {
        'timestamp': Config.TIMESTAMP,
        'model_variant': Config.MODEL_VARIANT,
        'task_variant': Config.TASK_VARIANT,
        'prompt_type': Config.PROMPT_TYPE,
        'grid_sizes': {},
        'summary': {}
    }
    
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
    
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined results saved: {combined_file}")
    sys.stdout.flush()
    
    return combined_file


def print_summary(all_results):
    """Print summary of evaluation results"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    for grid_size, building_results in all_results.items():
        print(f"\nGrid Size: {grid_size}x{grid_size}")
        print(f"Buildings evaluated: {len(building_results)}")
        
        all_grid_acc = [m['grid_cell_accuracy'] for _, m in building_results]
        all_dir_acc = [m['direction_accuracy'] for _, m in building_results]
        all_exact_acc = [m['exact_match_accuracy'] for _, m in building_results]
        
        print(f"Mean Grid Cell Accuracy: {np.mean(all_grid_acc):.2%}")
        print(f"Mean Direction Accuracy: {np.mean(all_dir_acc):.2%}")
        print(f"Mean Exact Match Accuracy: {np.mean(all_exact_acc):.2%}")
        sys.stdout.flush()


# ==================== BUILDING EVALUATION ====================

def split_into_chunks(items, chunk_size=100):
    """Split list into chunks"""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def evaluate_building(building_name, grid_size):
    """Evaluate a single building with a specific grid size using batch API"""
    print(f"\n{'='*60}")
    print(f"Building: {building_name}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Load ground truth
    try:
        ground_truth_df = load_ground_truth(building_name, grid_size)
        print(f"Loaded {len(ground_truth_df)} images with ground truth")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR loading ground truth: {e}")
        sys.stdout.flush()
        return None, None
    
    # Setup output directory
    output_dir = Config.setup_output_dir(grid_size, building_name)
    
    # Prepare batch requests
    try:
        all_batch_requests = prepare_batch_requests_for_building(
            building_name, grid_size, ground_truth_df
        )
        print(f"Prepared {len(all_batch_requests)} batch requests")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR preparing batch: {e}")
        sys.stdout.flush()
        return None, None
    
    # Split into chunks if needed
    CHUNK_SIZE = 100
    request_chunks = list(split_into_chunks(all_batch_requests, CHUNK_SIZE))
    print(f"Split into {len(request_chunks)} batches ({CHUNK_SIZE} images per batch)")
    sys.stdout.flush()
    
    all_predictions = []
    
    # Process each chunk
    for chunk_idx, chunk_requests in enumerate(request_chunks, 1):
        print(f"\nProcessing batch chunk {chunk_idx}/{len(request_chunks)}...")
        sys.stdout.flush()
        
        # Save batch file
        batch_filepath = save_batch_file(
            chunk_requests, output_dir, building_name, grid_size, chunk_idx
        )
        
        # Submit batch (skip if dry-run)
        if Config.DRY_RUN:
            print(f"DRY-RUN: Would submit batch file: {batch_filepath}")
            sys.stdout.flush()
            continue
        
        try:
            batch_id = submit_batch(batch_filepath, building_name, grid_size)
        except Exception as e:
            print(f"ERROR submitting batch chunk {chunk_idx}: {e}")
            sys.stdout.flush()
            continue
        
        # Wait for completion
        batch = wait_for_batch_completion(batch_id, f"{building_name} (chunk {chunk_idx})")
        
        if batch is None:
            print(f"ERROR: Batch chunk {chunk_idx} did not complete successfully")
            sys.stdout.flush()
            continue
        
        # Download results
        results_file = download_batch_results(batch_id, output_dir)
        
        if results_file is None:
            continue
        
        # Process results
        chunk_predictions = process_batch_results(results_file, ground_truth_df)
        all_predictions.extend(chunk_predictions)
        print(f"Processed {len(chunk_predictions)} predictions from chunk {chunk_idx}")
        sys.stdout.flush()
    
    if Config.DRY_RUN:
        return None, None
    
    if not all_predictions:
        return None, None
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    sys.stdout.flush()
    
    # Calculate metrics on all predictions
    metrics = calculate_metrics(all_predictions)
    
    # Save results
    save_building_results(building_name, grid_size, metrics, all_predictions)
    
    # Print summary
    print(f"\n{building_name} Results:")
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Unparseable: {metrics['unparseable_responses']}")
    print(f"  Grid Cell Accuracy: {metrics['grid_cell_accuracy']:.2%}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    sys.stdout.flush()
    
    return metrics, all_predictions


# ==================== MAIN ====================

def main():
    # Parse arguments
    args = parse_args()
    Config.DRY_RUN = args.dry_run
    
    # Override print with auto-flush version
    global print
    print = functools.partial(print, flush=True)
    
    print(f"{'='*80}")
    print("GPT-5 EVALUATION")
    print(f"{'='*80}")
    
    # Load environment
    Config.load_env()
    
    model_config = Config.get_model_config()
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'SINGLE BUILDING' if Config.SINGLE_BUILDING_MODE else 'ALL BUILDINGS'}")
    print(f"  Model: {model_config['display_name']}")
    print(f"  Task Variant: {Config.TASK_VARIANT}")
    print(f"  Prompt Type: {Config.PROMPT_TYPE}")
    print(f"  Grid Sizes: {Config.GRID_SIZES}")
    print(f"  Timestamp: {Config.TIMESTAMP}")
    sys.stdout.flush()
    
    # Get buildings to process
    if Config.SINGLE_BUILDING_MODE:
        buildings_to_process = [Config.SINGLE_BUILDING_NAME]
        print(f"\nProcessing single building: {Config.SINGLE_BUILDING_NAME}")
        sys.stdout.flush()
    else:
        valid_buildings, invalid_buildings = get_all_buildings()
        buildings_to_process = valid_buildings
        
        print(f"\nBuilding validation:")
        print(f"  Valid buildings: {len(valid_buildings)}")
        print(f"  Invalid buildings: {len(invalid_buildings)}")
        sys.stdout.flush()
    
    # Process each grid size
    all_results = {}
    skip_summary = {}  # Track skipped buildings per grid size
    
    for grid_size in Config.GRID_SIZES:
        print(f"\n{'='*80}")
        print(f"PROCESSING GRID SIZE: {grid_size}x{grid_size}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # Validate buildings for this grid size
        print(f"\nValidating buildings for grid size {grid_size}...")
        valid_for_grid = []
        skipped_reasons = {}
        
        for building_name in buildings_to_process:
            is_valid, reason = validate_building_for_grid(building_name, grid_size)
            
            if is_valid:
                valid_for_grid.append(building_name)
            else:
                skipped_reasons[building_name] = reason
        
        print(f"  Valid: {len(valid_for_grid)}")
        print(f"  Skipped: {len(skipped_reasons)}")
        sys.stdout.flush()
        
        if skipped_reasons:
            # Count skip reasons
            reason_counts = {}
            for reason in skipped_reasons.values():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            print(f"\n  Skip reasons:")
            for reason, count in reason_counts.items():
                print(f"    {reason}: {count}")
            sys.stdout.flush()
            
            # Store for final summary
            skip_summary[grid_size] = skipped_reasons
        
        # Process valid buildings
        grid_results = []
        
        for building_name in valid_for_grid:
            metrics, predictions = evaluate_building(building_name, grid_size)
            
            if metrics is not None:
                grid_results.append((building_name, metrics))
        
        all_results[grid_size] = grid_results
    
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
        for grid_size, skipped in skip_summary.items():
            print(f"\nGrid Size {grid_size}x{grid_size}: {len(skipped)} skipped")
            
            # Group by reason
            by_reason = {}
            for building, reason in skipped.items():
                by_reason.setdefault(reason, []).append(building)
            
            for reason, buildings in by_reason.items():
                print(f"  {reason}: {len(buildings)}")
                if Config.DEBUG_MODE and len(buildings) <= 5:
                    for b in buildings:
                        print(f"    - {b}")
        sys.stdout.flush()
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {Config.OUTPUT_BASE_PATH}/{Config.TIMESTAMP}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()