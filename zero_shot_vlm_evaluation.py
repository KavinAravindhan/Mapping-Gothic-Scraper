"""
zero_shot_vlm_evaluation.py

This script evaluates vision-language models on spatial localization tasks.
Given a gridded floor plan and interior photos, models predict:
1. Grid cell location (A-J, 1-10)
2. Camera direction (N, NE, E, SE, S, SW, W, NW)

Supports multiple VLMs with easy model swapping.
"""

import torch
from PIL import Image
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Import model-specific libraries
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

MODEL_CONFIGS = {
    "qwen2-vl-7b": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "processor_id": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2vl"
    },
    "molmo-7b": {
        "model_id": "allenai/Molmo-7B-D-0924",
        "processor_id": "allenai/Molmo-7B-D-0924",
        "type": "molmo"
    },
    "internvl2-8b": {
        "model_id": "OpenGVLab/InternVL2-8B",
        "processor_id": "OpenGVLab/InternVL2-8B",
        "type": "internvl"
    },
    "llava-next-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "processor_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "type": "llava"
    },
    "cogvlm2-19b": {
        "model_id": "THUDM/cogvlm2-llama3-chinese-chat-19B",
        "processor_id": "THUDM/cogvlm2-llama3-chinese-chat-19B",
        "type": "cogvlm"
    },
    "deepseek-vl-7b": {
        "model_id": "deepseek-ai/deepseek-vl-7b-chat",
        "processor_id": "deepseek-ai/deepseek-vl-7b-chat",
        "type": "deepseek"
    }
}

# ==============================================================================
# PROMPT TEMPLATE
# ==============================================================================

ZERO_SHOT_PROMPT = """You are analyzing architectural photographs to determine their location on a floor plan.

I will show you:
1. A gridded floor plan with a 10x10 grid overlay (columns A-J, rows 1-10)
2. A photograph taken inside this building

Your task is to identify:
1. The grid cell where the photo was taken (format: [Letter][Number], e.g., C5)
2. The camera direction when the photo was taken (N, NE, E, SE, S, SW, W, NW)

Analyze the architectural features, spatial layout, and visual cues carefully.

Provide your answer in EXACTLY this format:
GRID_CELL: [Your answer]
DIRECTION: [Your answer]
REASONING: [Brief explanation of your reasoning]

Floor plan image is provided first, followed by the interior photograph."""

# ==============================================================================
# MODEL HANDLER CLASS
# ==============================================================================

class VLMHandler:
    """Handles loading and inference for different vision-language models."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the VLM handler.
        
        Args:
            model_name: Key from MODEL_CONFIGS
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.config = MODEL_CONFIGS[model_name]
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the model and processor based on model type."""
        print(f"Loading {self.model_name}...")
        
        model_type = self.config["type"]
        
        if model_type == "qwen2vl":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.config["processor_id"])
            
        elif model_type == "molmo":
            self.processor = AutoProcessor.from_pretrained(
                self.config["processor_id"],
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
        elif model_type == "internvl":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.config["processor_id"],
                trust_remote_code=True
            )
            
        elif model_type == "llava":
            self.processor = AutoProcessor.from_pretrained(self.config["processor_id"])
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
        elif model_type == "cogvlm":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["processor_id"],
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
        elif model_type == "deepseek":
            self.processor = AutoProcessor.from_pretrained(self.config["processor_id"])
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        print(f"✓ {self.model_name} loaded successfully")
        
    def generate_response(self, floorplan_path: str, photo_path: str) -> str:
        """
        Generate model response for a given floor plan and photo pair.
        
        Args:
            floorplan_path: Path to gridded floor plan image
            photo_path: Path to interior photograph
            
        Returns:
            Model's text response
        """
        # Load images
        floorplan_img = Image.open(floorplan_path).convert("RGB")
        photo_img = Image.open(photo_path).convert("RGB")
        
        model_type = self.config["type"]
        
        # Model-specific inference
        if model_type == "qwen2vl":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": floorplan_img},
                        {"type": "image", "image": photo_img},
                        {"type": "text", "text": ZERO_SHOT_PROMPT}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[floorplan_img, photo_img], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
        elif model_type == "molmo":
            inputs = self.processor.process(
                images=[floorplan_img, photo_img],
                text=ZERO_SHOT_PROMPT
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = self.model.generate_from_batch(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7
                )
            response = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
        elif model_type == "internvl":
            pixel_values = self.processor(images=[floorplan_img, photo_img], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            prompt = f"<image>\n<image>\n{ZERO_SHOT_PROMPT}"
            
            with torch.no_grad():
                response = self.model.chat(
                    self.processor.tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config={"max_new_tokens": 512}
                )
                
        elif model_type == "llava":
            prompt = f"USER: <image>\n<image>\n{ZERO_SHOT_PROMPT}\nASSISTANT:"
            inputs = self.processor(text=prompt, images=[floorplan_img, photo_img], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
        elif model_type == "cogvlm":
            inputs = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=ZERO_SHOT_PROMPT,
                images=[floorplan_img, photo_img]
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
        elif model_type == "deepseek":
            conversation = [
                {
                    "role": "user",
                    "content": f"<image>\n<image>\n{ZERO_SHOT_PROMPT}",
                    "images": [floorplan_img, photo_img]
                }
            ]
            inputs = self.processor(conversation, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        return response

# ==============================================================================
# RESULT PARSING
# ==============================================================================

def parse_model_output(response: str) -> Dict[str, str]:
    """
    Parse model output to extract grid cell, direction, and reasoning.
    
    Args:
        response: Raw model output text
        
    Returns:
        Dictionary with parsed components
    """
    result = {
        "grid_cell": "UNKNOWN",
        "direction": "UNKNOWN",
        "reasoning": "Could not parse response"
    }
    
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("GRID_CELL:"):
            result["grid_cell"] = line.split("GRID_CELL:")[1].strip()
        elif line.startswith("DIRECTION:"):
            result["direction"] = line.split("DIRECTION:")[1].strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split("REASONING:")[1].strip()
    
    return result

# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def evaluate_model_on_building(
    model_name: str,
    building_name: str,
    photo_ids: List[str],
    use_arrows: bool = False,
    output_dir: str = "evaluation_results"
):
    """
    Evaluate a VLM on all photos for a given building.
    
    Args:
        model_name: Key from MODEL_CONFIGS
        building_name: Name of the building (e.g., "Beaumont-sur-Oise-Eglise-Saint-Leonor")
        photo_ids: List of photo identifiers (e.g., ["001", "002", "003"])
        use_arrows: Whether to use arrows visualization or base floorplan
        output_dir: Directory to save results
    """
    # Initialize model
    handler = VLMHandler(model_name)
    handler.load_model()
    
    # Define paths
    base_dir = f"maps_output/{building_name}"
    
    if use_arrows:
        floorplan_path = f"{base_dir}/{building_name}_arrows_visualization.jpg"
        plan_type = "arrows"
    else:
        floorplan_path = f"{base_dir}/{building_name}_floorplan.jpg"
        plan_type = "base"
    
    # Check if floorplan exists
    if not os.path.exists(floorplan_path):
        print(f"Error: Floor plan not found at {floorplan_path}")
        return
    
    # Prepare results storage
    results = {
        "model": model_name,
        "building": building_name,
        "floorplan_type": plan_type,
        "timestamp": datetime.now().isoformat(),
        "predictions": []
    }
    
    # Evaluate each photo
    for photo_id in photo_ids:
        photo_path = f"{base_dir}/images/{building_name}_{photo_id}.jpg"
        
        if not os.path.exists(photo_path):
            print(f"Warning: Photo not found at {photo_path}, skipping...")
            continue
        
        print(f"\nEvaluating photo {photo_id}...")
        
        try:
            # Generate prediction
            response = handler.generate_response(floorplan_path, photo_path)
            parsed = parse_model_output(response)
            
            # Store result
            result_entry = {
                "photo_id": photo_id,
                "photo_path": photo_path,
                "raw_response": response,
                "predicted_grid_cell": parsed["grid_cell"],
                "predicted_direction": parsed["direction"],
                "reasoning": parsed["reasoning"]
            }
            
            results["predictions"].append(result_entry)
            
            print(f"  Grid Cell: {parsed['grid_cell']}")
            print(f"  Direction: {parsed['direction']}")
            
        except Exception as e:
            print(f"Error processing photo {photo_id}: {str(e)}")
            results["predictions"].append({
                "photo_id": photo_id,
                "photo_path": photo_path,
                "error": str(e)
            })
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/{model_name}_{building_name}_{plan_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Processed {len(results['predictions'])} photos")

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    
    # List your photo IDs (the numbers/identifiers after the building name)
    # Example: if you have Beaumont-sur-Oise-Eglise-Saint-Leonor_001.jpg, 
    # Beaumont-sur-Oise-Eglise-Saint-Leonor_002.jpg, etc.
    photo_ids = ["001", "002", "003", "004", "005"]  # MODIFY THIS
    
    # Choose which model to evaluate
    model_to_test = "qwen2-vl-7b"  # Change this to test different models
    
    # Choose whether to use arrows visualization or base floorplan
    use_arrows_plan = False  # Set to True to use arrows visualization
    
    # Run evaluation
    evaluate_model_on_building(
        model_name=model_to_test,
        building_name=building_name,
        photo_ids=photo_ids,
        use_arrows=use_arrows_plan,
        output_dir="evaluation_results"
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nTo test another model, change 'model_to_test' to one of:")
    for model_key in MODEL_CONFIGS.keys():
        print(f"  - {model_key}")