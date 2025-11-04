# Standard library imports
import base64
import csv
import json
import os
from pathlib import Path

# Third-party imports
from PIL import Image
import torch

# Try different import methods for Qwen
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print("✓ Using Qwen2VLForConditionalGeneration")
    QWEN_AVAILABLE = True
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        Qwen2VLForConditionalGeneration = AutoModelForVision2Seq
        print("✓ Using AutoModelForVision2Seq as fallback")
        QWEN_AVAILABLE = True
    except ImportError:
        print("✗ Error: Could not import Qwen model classes")
        QWEN_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    print("⚠ Warning: qwen_vl_utils not available, using fallback")
    QWEN_UTILS_AVAILABLE = False
    
    def process_vision_info(messages):
        """Fallback function if qwen_vl_utils is not available"""
        images = []
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        images.append(item['image'])
        return images, None


class QwenFloorPlanAnalyzer:
    """Analyze photos using Qwen VL model to identify location on floor plan"""
    
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initialize the Qwen model
        
        Args:
            model_name: HuggingFace model identifier
        """
        if not QWEN_AVAILABLE:
            raise ImportError(
                "Qwen model classes not available. Please install the required packages:\n"
                "pip install transformers>=4.37.0 accelerate --break-system-packages\n"
                "pip install qwen-vl-utils --break-system-packages"
            )
        
        print(f"Loading Qwen model: {model_name}")
        print("This may take a few minutes on first run...")
        
        try:
            # Load model and processor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            print("\nTroubleshooting:")
            print("1. Update transformers: pip install --upgrade transformers")
            print("2. Install required packages: pip install accelerate torch")
            print("3. Try a different model: Qwen/Qwen2-VL-2B-Instruct")
            raise
    
    def analyze_photo_location(self, gridded_floor_plan_path, photo_path):
        """
        Analyze a photo and determine its location on the gridded floor plan
        
        Args:
            gridded_floor_plan_path: Path to the floor plan with grid overlay
            photo_path: Path to the photo to analyze
            
        Returns:
            dict with 'grid_cell', 'direction', and 'raw_response'
        """
        print(f"\nAnalyzing photo: {photo_path}")
        
        # Create the prompt
        prompt = """You are analyzing architectural photographs and floor plans. 

I'm showing you two images:
1. A gridded floor plan of a building (with columns labeled A, B, C... and rows labeled 1, 2, 3...)
2. A photograph taken from somewhere inside this building

Your task is to identify:
1. Which grid cell (e.g., "C5", "A3") the photo was taken from
2. The direction the camera was facing when the photo was taken (N, NE, E, SE, S, SW, W, NW)

Please analyze the architectural features in the photo (columns, vaults, walls, windows) and match them to the floor plan.

Respond in this exact format:
GRID_CELL: [letter][number]
DIRECTION: [direction]
CONFIDENCE: [low/medium/high]
REASONING: [brief explanation]

Example:
GRID_CELL: C5
DIRECTION: NE
CONFIDENCE: high
REASONING: The photo shows three columns in a row which matches the layout at C5. The vault pattern and wall configuration indicate the camera is facing northeast."""

        # Prepare messages for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": gridded_floor_plan_path,
                    },
                    {
                        "type": "image",
                        "image": photo_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        print("Generating analysis...")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Raw response:\n{response}\n")
        
        # Parse the response
        result = self._parse_response(response)
        result['raw_response'] = response
        
        return result
    
    def _parse_response(self, response):
        """Parse the model's response to extract structured data"""
        result = {
            'grid_cell': None,
            'direction': None,
            'confidence': None,
            'reasoning': None
        }
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('GRID_CELL:'):
                result['grid_cell'] = line.split(':', 1)[1].strip()
            elif line.startswith('DIRECTION:'):
                result['direction'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                result['confidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    
    def test_single_image(self, gridded_floor_plan_path, photo_path, ground_truth=None):
        """
        Test with a single image - simplified for initial testing
        
        Args:
            gridded_floor_plan_path: Path to gridded floor plan
            photo_path: Path to single photo to test
            ground_truth: Optional dict with 'x', 'y', 'direction' for comparison
            
        Returns:
            dict with analysis results
        """
        print(f"\n{'='*60}")
        print(f"TESTING WITH SINGLE IMAGE")
        print(f"{'='*60}")
        print(f"Floor plan: {gridded_floor_plan_path}")
        print(f"Photo: {photo_path}")
        
        # Analyze the photo
        result = self.analyze_photo_location(gridded_floor_plan_path, photo_path)
        
        # Add ground truth if provided
        if ground_truth:
            result['ground_truth'] = ground_truth
        
        # Print results nicely
        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Grid Cell:  {result['grid_cell']}")
        print(f"Direction:  {result['direction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning:  {result['reasoning']}")
        
        if ground_truth:
            print(f"\n{'='*60}")
            print(f"GROUND TRUTH COMPARISON")
            print(f"{'='*60}")
            print(f"Ground Truth Direction: {ground_truth.get('direction', 'N/A')}")
            print(f"Ground Truth Position:  x={ground_truth.get('x', 'N/A')}, y={ground_truth.get('y', 'N/A')}")
            
            # Check if direction matches
            if result['direction'] == ground_truth.get('direction'):
                print(f"✓ Direction MATCHES!")
            else:
                print(f"✗ Direction DOES NOT MATCH")
        
        print(f"\n{'='*60}")
        print(f"RAW MODEL RESPONSE")
        print(f"{'='*60}")
        print(result['raw_response'])
        
        return result
    
    def analyze_building(self, building_dir, output_json='analysis_results.json'):
        """
        Analyze all photos for a single building
        
        Args:
            building_dir: Path to building directory (e.g., 'maps_output/Building-Name/')
            output_json: Name of output JSON file
        """
        building_path = Path(building_dir)
        building_name = building_path.name
        
        print(f"\n{'='*60}")
        print(f"Analyzing building: {building_name}")
        print(f"{'='*60}")
        
        # Find the gridded floor plan
        gridded_floorplan = building_path / f"{building_name}_floorplan_gridded.jpg"
        if not gridded_floorplan.exists():
            print(f"✗ Error: Gridded floor plan not found at {gridded_floorplan}")
            return None
        
        # Load ground truth data
        data_json = building_path / 'data.json'
        ground_truth = {}
        if data_json.exists():
            with open(data_json, 'r') as f:
                data = json.load(f)
                for overlay in data['overlays']:
                    ground_truth[overlay['image_id']] = {
                        'x': overlay['x'],
                        'y': overlay['y'],
                        'direction': overlay['direction']
                    }
        
        # Find all images
        images_dir = building_path / 'images'
        if not images_dir.exists():
            print(f"✗ Error: Images directory not found at {images_dir}")
            return None
        
        image_files = list(images_dir.glob('*.jpg'))
        print(f"Found {len(image_files)} images to analyze\n")
        
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}]")
            image_id = image_path.stem
            
            try:
                # Analyze the photo
                analysis = self.analyze_photo_location(str(gridded_floorplan), str(image_path))
                
                # Add metadata
                analysis['image_id'] = image_id
                analysis['image_path'] = str(image_path)
                
                # Add ground truth if available
                if image_id in ground_truth:
                    analysis['ground_truth'] = ground_truth[image_id]
                
                results.append(analysis)
                
                print(f"✓ Grid: {analysis['grid_cell']}, Direction: {analysis['direction']}")
                if image_id in ground_truth:
                    print(f"  Ground truth direction: {ground_truth[image_id]['direction']}")
                
            except Exception as e:
                print(f"✗ Error analyzing {image_id}: {str(e)}")
                results.append({
                    'image_id': image_id,
                    'error': str(e)
                })
        
        # Save results
        output_path = building_path / output_json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved results to: {output_path}")
        
        return results


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...\n")
    
    dependencies = {
        'transformers': False,
        'torch': False,
        'PIL': False,
        'qwen_vl_utils': False
    }
    
    try:
        import transformers
        dependencies['transformers'] = True
        print(f"✓ transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed")
    
    try:
        import torch
        dependencies['torch'] = True
        print(f"✓ torch version: {torch.__version__}")
    except ImportError:
        print("✗ torch not installed")
    
    try:
        from PIL import Image
        dependencies['PIL'] = True
        print(f"✓ PIL available")
    except ImportError:
        print("✗ PIL not installed")
    
    try:
        import qwen_vl_utils
        dependencies['qwen_vl_utils'] = True
        print(f"✓ qwen_vl_utils available")
    except ImportError:
        print("✗ qwen_vl_utils not installed")
    
    print("\n" + "="*60)
    
    if not all(dependencies.values()):
        print("Missing dependencies detected. Install with:")
        print("pip install transformers>=4.37.0 torch pillow qwen-vl-utils accelerate --break-system-packages")
        return False
    else:
        print("✓ All dependencies installed!")
        return True


def main():
    """Example usage"""
    
    # Check dependencies first
    if not check_dependencies():
        print("\n✗ Please install missing dependencies before running.")
        return
    
    # ========================================
    # CONFIGURATION - EDIT THESE PATHS
    # ========================================
    
    # Choose mode: 'single' for testing one image, 'full' for all images
    MODE = 'single'  # Change to 'full' to analyze all images in a building
    
    # Building directory
    BUILDING_DIR = 'maps_output/Beaumont-sur-Oise-Eglise-Saint-Leonor'
    
    # For single image testing - specify the image filename
    TEST_IMAGE_FILENAME = '1065_00010.jpg'  # Change this to test different images
    
    # Model to use (uncomment the one you want)
    # MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # More accurate, slower
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # Faster, less accurate
    
    # ========================================
    # END CONFIGURATION
    # ========================================
    
    # Initialize the analyzer
    print("\nInitializing Qwen model...")
    try:
        analyzer = QwenFloorPlanAnalyzer(model_name=MODEL_NAME)
    except Exception as e:
        print(f"\n✗ Failed to initialize model: {str(e)}")
        return
    
    if MODE == 'single':
        # ========================================
        # SINGLE IMAGE TEST MODE
        # ========================================
        
        building_path = Path(BUILDING_DIR)
        building_name = building_path.name
        
        # Construct paths
        gridded_floor_plan = building_path / f"{building_name}_floorplan_gridded.jpg"
        photo_path = building_path / 'images' / TEST_IMAGE_FILENAME
        
        # Check if files exist
        if not gridded_floor_plan.exists():
            print(f"✗ Error: Gridded floor plan not found at {gridded_floor_plan}")
            return
        
        if not photo_path.exists():
            print(f"✗ Error: Photo not found at {photo_path}")
            print(f"\nAvailable images in {building_path / 'images'}:")
            for img in (building_path / 'images').glob('*.jpg'):
                print(f"  - {img.name}")
            return
        
        # Load ground truth if available
        data_json = building_path / 'data.json'
        ground_truth = None
        if data_json.exists():
            with open(data_json, 'r') as f:
                data = json.load(f)
                image_id = Path(TEST_IMAGE_FILENAME).stem
                for overlay in data['overlays']:
                    if overlay['image_id'] == image_id:
                        ground_truth = {
                            'x': overlay['x'],
                            'y': overlay['y'],
                            'direction': overlay['direction']
                        }
                        break
        
        # Run single image test
        result = analyzer.test_single_image(
            str(gridded_floor_plan),
            str(photo_path),
            ground_truth=ground_truth
        )
        
        # Save result
        output_file = 'single_image_test_result.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Result saved to: {output_file}")
        
    else:
        # ========================================
        # FULL BUILDING ANALYSIS MODE
        # ========================================
        
        results = analyzer.analyze_building(
            building_dir=BUILDING_DIR,
            output_json='qwen_analysis_results.json'
        )
        
        # Print summary
        if results:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"Total images analyzed: {len(results)}")
            
            successful = [r for r in results if 'grid_cell' in r and r['grid_cell']]
            print(f"Successful analyses: {len(successful)}")
            
            # Compare with ground truth where available
            correct_directions = 0
            total_with_gt = 0
            for r in results:
                if 'ground_truth' in r and 'direction' in r:
                    total_with_gt += 1
                    if r['direction'] == r['ground_truth']['direction']:
                        correct_directions += 1
            
            if total_with_gt > 0:
                accuracy = (correct_directions / total_with_gt) * 100
                print(f"Direction accuracy: {correct_directions}/{total_with_gt} ({accuracy:.1f}%)")


if __name__ == "__main__":
    main()