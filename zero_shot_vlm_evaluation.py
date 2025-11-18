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
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForImageTextToText  # Add this
)

# from deepseek_vl.models import DeepseekVLV2Processor

os.environ['HF_HOME'] = '/mnt/swordfish-pool2/kavin/cache'

# Set device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

# Model Configurations
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
    "internvl3.5-8b": {  # Much newer and better than InternVL2!
        "model_id": "OpenGVLab/InternVL3_5-8B-HF",
        "processor_id": "OpenGVLab/InternVL3_5-8B-HF",
        "type": "internvl35"
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
    "deepseek-vl2-tiny": {
        "model_id": "deepseek-ai/deepseek-vl2-tiny",
        "processor_id": "deepseek-ai/deepseek-vl2-tiny",
        "type": "deepseek_vl2"
    }
}


# Prompt Template
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
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
        elif model_type == "internvl35":
            # Choose one of these model_ids:
            #   "OpenGVLab/InternVL3_5-8B-HF"  or  "OpenGVLab/InternVL3-8B-hf"
            model_id = self.config["model_id"]

            # If you have flash-attn installed and working, set this to "flash_attention_2".
            # Otherwise, use "sdpa" (PyTorch SDPA) or omit the arg.
            ATTN_IMPL = "sdpa"  # or "flash_attention_2" if your env supports it

            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=ATTN_IMPL,  # <-- supported knob; keep, change, or drop
            ).eval()
                
        elif model_type == "llava":
            self.processor = AutoProcessor.from_pretrained(self.config["processor_id"])
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.config["model_id"],
                dtype=torch.float16,
                device_map="auto"
            )
            
        elif model_type == "cogvlm":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["processor_id"],
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_id"],
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
        elif model_type == "deepseek_vl2":
            model_id = self.config["model_id"]

            # Load the processor via Transformers with remote code
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            # Load the model; remote code exposes the right class under AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            ).eval()

            # Tokenizer is part of the processor
            self.tokenizer = self.processor.tokenizer
        
        print(f"{self.model_name} loaded successfully")
        
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
            # Molmo debugging and processing
            print(f"\n=== Molmo Debug Info ===")
            print(f"Floorplan image size: {floorplan_img.size}")
            print(f"Photo image size: {photo_img.size}")
            
            # Process images and text
            inputs = self.processor.process(
                images=[floorplan_img, photo_img],
                text=ZERO_SHOT_PROMPT
            )
            
            # Debug: check what the processor returns
            print(f"Type of inputs: {type(inputs)}")
            print(f"Inputs keys: {inputs.keys()}")
            for key, value in inputs.items():
                if value is not None:
                    print(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
                else:
                    print(f"  {key}: None")
            
            # Move inputs to device and add batch dimension (handling None values)
            processed_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    if isinstance(v, torch.Tensor):
                        # Add batch dimension and move to device
                        processed_inputs[k] = v.to(self.device).unsqueeze(0)
                        print(f"Processed {k}: shape={processed_inputs[k].shape}")
                    else:
                        processed_inputs[k] = v
                        print(f"Processed {k}: non-tensor, type={type(v)}")
                else:
                    print(f"Skipping {k}: is None")
            
            # Generate with Molmo's specific API
            print(f"\nGenerating with keys: {processed_inputs.keys()}")

            try:
                with torch.no_grad():
                    output = self.model.generate_from_batch(
                        processed_inputs,
                        GenerationConfig(
                            max_new_tokens=512, 
                            stop_strings="<|endoftext|>",
                            use_cache=True,  # Enable KV cache
                            do_sample=False,  # Greedy decoding
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        ),
                        tokenizer=self.processor.tokenizer
                    )
                
                print(f"Output shape: {output.shape}")
                print(f"Output type: {type(output)}")
                
                # Extract only the generated tokens
                if 'input_ids' in processed_inputs:
                    input_length = processed_inputs['input_ids'].size(1)
                    generated_tokens = output[0, input_length:]
                    print(f"Generated tokens shape: {generated_tokens.shape}")
                else:
                    generated_tokens = output[0]
                    print(f"Using full output, shape: {generated_tokens.shape}")
                
                response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"Response length: {len(response)}")
                print(f"=== End Molmo Debug ===\n")
                
            except Exception as e:
                print(f"ERROR during generation: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
        elif model_type == "internvl35":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": floorplan_img},
                    {"type": "image", "image": photo_img},
                    {"type": "text",  "text": ZERO_SHOT_PROMPT},
                ],
            }]

            # Tokenize & pack images via the processor’s chat template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

            # Decode only the generated continuation
            response = self.processor.decode(
                output_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
                
        elif model_type == "llava":
            # LLaVA-NeXT requires specific formatting for multiple images
            # Use separate image tokens for each image
            prompt = f"[INST] <image>\n<image>\n{ZERO_SHOT_PROMPT} [/INST]"
            
            inputs = self.processor(
                text=prompt, 
                images=[floorplan_img, photo_img], 
                return_tensors="pt",
                padding=True
            )
            
            # Move all inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode only the generated part (excluding the prompt)
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
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
            
        elif model_type == "deepseek_vl2":
            # Two-image conversation: floorplan first, interior photo second
            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\n<image>\n" + ZERO_SHOT_PROMPT,
                    "images": [floorplan_img, photo_img],  # PIL images are OK
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # Prepare inputs (handles batching, vision tiling, etc.)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=[floorplan_img, photo_img],
                force_batchify=True,
                system_prompt=""
            ).to(self.model.device)

            # DeepSeek-VL2 remote code exposes this helper on the model
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            with torch.no_grad():
                output_ids = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )

            response = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)


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
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("GRID_CELL:"):
            result["grid_cell"] = line.split("GRID_CELL:")[1].strip()
        elif line.startswith("DIRECTION:"):
            result["direction"] = line.split("DIRECTION:")[1].strip()
        elif line.startswith("REASONING:"):
            # Capture everything after REASONING: including subsequent lines
            reasoning_parts = [line.split("REASONING:")[1].strip()]
            # Get remaining lines as reasoning
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    reasoning_parts.append(lines[j].strip())
            result["reasoning"] = " ".join(reasoning_parts)
            break
    
    return result


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
        floorplan_path = f"{base_dir}/{building_name}_floorplan_arrows_gridded.jpg"
        plan_type = "arrows"
    else:
        floorplan_path = f"{base_dir}/{building_name}_floorplan_gridded.jpg"
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
        photo_path = f"{base_dir}/images/{photo_id}.jpg"
        
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
    
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(results['predictions'])} photos")

if __name__ == "__main__":

    building_name = "Beaumont-sur-Oise-Eglise-Saint-Leonor"
    photo_ids = ["1065_00010", "1065_00011", "1065_00028", "1065_00044", "1065_00033"]
    
    # model_to_test = "qwen2-vl-7b"
    # model_to_test = "molmo-7b"
    # model_to_test = "internvl3.5-8b"
    # model_to_test = "llava-next-7b"
    # model_to_test = "cogvlm2-19b"
    model_to_test = "deepseek-vl2-tiny"

    use_arrows_plan = True  # Set to True to use arrows visualization
    
    # Run evaluation
    evaluate_model_on_building(
        model_name=model_to_test,
        building_name=building_name,
        photo_ids=photo_ids,
        use_arrows=use_arrows_plan,
        output_dir="evaluation_results"
    )