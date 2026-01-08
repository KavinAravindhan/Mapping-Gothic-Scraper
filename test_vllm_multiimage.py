# test_vllm_multiimage.py

import os
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
CACHE_DIR = "/mnt/swordfish-pool2/kavin/cache"

FLOOR_PLAN_PATH = "/mnt/swordfish-pool2/kavin/maps_output_analysis/grids_floorplan/grid_size_10/Beaumont-sur-Oise-Eglise-Saint-Leonor_floorplan_gridded.jpg"
PHOTO_PATH = "/mnt/swordfish-pool2/kavin/maps_output/Beaumont-sur-Oise-Eglise-Saint-Leonor/images/1065_00001.jpg"

def test_multi_image_generate():
    """Test using llm.generate() with multi_modal_data - following vLLM docs pattern"""
    print("\n" + "="*60)
    print("TEST 1: Using llm.generate() method")
    print("="*60)
    
    # Load model
    print("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 2},  # Specify we want 2 images
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load images
    print("Loading images...")
    floor_plan = Image.open(FLOOR_PLAN_PATH).convert("RGB")
    photo = Image.open(PHOTO_PATH).convert("RGB")
    
    # Prepare messages with placeholders
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for first image
                {"type": "image"},  # Placeholder for second image
                {
                    "type": "text",
                    "text": "The first image is a Gothic church floor plan. The second image is an interior photograph. Describe what you see in both images."
                },
            ],
        }
    ]
    
    # Apply chat template to get prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"Generated prompt: {prompt[:200]}...")
    
    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )
    
    # Run inference with generate()
    try:
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": [floor_plan, photo]},
            },
            sampling_params=sampling_params,
        )
        
        print("\n✅ SUCCESS! Multi-image with generate() works!")
        print(f"\nResponse: {outputs[0].outputs[0].text}")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


def test_multi_image_chat():
    """Test using llm.chat() with image URLs - following vLLM docs pattern"""
    print("\n" + "="*60)
    print("TEST 2: Using llm.chat() method")
    print("="*60)
    
    # Load model
    print("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 2},
    )
    
    # Prepare messages for chat
    # Note: chat() can handle local file paths directly
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The first image is a Gothic church floor plan. The second is an interior photo. Describe both."},
                {"type": "image_url", "image_url": {"url": FLOOR_PLAN_PATH}},
                {"type": "image_url", "image_url": {"url": PHOTO_PATH}},
            ],
        }
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )
    
    try:
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
        )
        
        print("\n✅ SUCCESS! Multi-image with chat() works!")
        print(f"\nResponse: {outputs[0].outputs[0].text}")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


def test_concatenated_fallback():
    """Fallback: concatenate images side-by-side"""
    print("\n" + "="*60)
    print("TEST 3: Fallback - Concatenated image")
    print("="*60)
    
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1},
    )
    
    # Concatenate images
    floor_plan = Image.open(FLOOR_PLAN_PATH).convert("RGB")
    photo = Image.open(PHOTO_PATH).convert("RGB")
    
    width = floor_plan.width + photo.width
    height = max(floor_plan.height, photo.height)
    combined = Image.new('RGB', (width, height))
    combined.paste(floor_plan, (0, 0))
    combined.paste(photo, (floor_plan.width, 0))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This image has two parts side-by-side. Left is a Gothic church floor plan, right is an interior photo. Describe both."},
                {"type": "image_url", "image_url": {"url": "placeholder"}},  # Will be replaced
            ],
        }
    ]
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    try:
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": combined},
            },
            sampling_params=sampling_params,
        )
        
        print("\n✅ Concatenated image works!")
        print(f"\nResponse: {outputs[0].outputs[0].text}")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    
    print("\n" + "#"*60)
    print("# Testing vLLM Multi-Image Support for Qwen3-VL")
    print("#"*60)
    
    # Test 1: generate() method
    success_generate = test_multi_image_generate()
    
    if not success_generate:
        # Test 2: chat() method (alternative)
        print("\nTrying chat() method as alternative...")
        success_chat = test_multi_image_chat()
        
        if not success_chat:
            # Test 3: Fallback to concatenated
            print("\nTrying concatenated image as fallback...")
            test_concatenated_fallback()
    
    print("\n" + "#"*60)
    print("# Testing Complete")
    print("#"*60)


if __name__ == "__main__":
    main()