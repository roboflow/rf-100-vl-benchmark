import os
import json
import glob
import base64
import numpy as np
import cv2
from PIL import Image
import time
import concurrent.futures
import threading
import logging
import random
import re
import supervision as sv
from tqdm import tqdm
from io import BytesIO
from google import genai
from google.genai import types
import argparse
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set your API key here or use environment variable
API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize Gemini client (making sure API_KEY is not None)
if API_KEY is None:
    raise ValueError("API_KEY is not set. Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=API_KEY)

# Select the model
MODEL_ID = "gemini-2.0-flash"

# System instructions for bounding boxes
#SYSTEM_PROMPT = """
#Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
#If an object is present multiple times, detect all instances.
#"""

SYSTEM_PROMPT = """Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Detect all instances of all objects requested by prompt."""

# Safety settings
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

# Configuration
MODEL_ID = "gemini-2.0-flash"  # Using Gemini Flash model
OUTPUT_DIR = "gemini_detection_results"
MERGED_OUTPUT_DIR = "gemini_detection_merged"
VISUALIZE_DIR = "gemini_visualized_predictions"
MAX_WORKERS = 12  # Maximum number of parallel workers for image processing
REQUEST_LIMIT = 120  # Maximum number of API requests per minute (Gemini limit)
MAX_RETRIES = 3  # Maximum number of retries for a single image
RETRY_DELAY_BASE = 2  # Base delay for exponential backoff (seconds)
RETRY_DELAY_MAX = 60  # Maximum retry delay (1 minute)


# Thread-safe rate limiter for API requests with jitter
class RateLimiter:
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period  # in seconds
        self.calls = []
        self.lock = threading.Lock()
        
    def __call__(self):
        with self.lock:
            now = time.time()
            # Remove calls older than the period
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # If we've reached the maximum number of calls in the period, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0, 0.1 * sleep_time)
                    time.sleep(sleep_time + jitter)
                    now = time.time()
                self.calls = self.calls[1:]  # Remove the oldest call
                
            # Add the current call time
            self.calls.append(now)

# Create a rate limiter instance
rate_limiter = RateLimiter(REQUEST_LIMIT)

def is_rate_limit_error(error_msg):
    """Check if an error message indicates a rate limit issue"""
    rate_limit_indicators = [
        "rate limit", "ratelimit", "too many requests", 
        "429", "throttl", "quota exceeded", "limit exceeded"
    ]
    error_lower = str(error_msg).lower()
    return any(indicator in error_lower for indicator in rate_limit_indicators)

def parse_json(json_output):
    """Parse the JSON content from the Gemini response"""
    # Parsing out the markdown fencing if present
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    
    # If no ```json marker was found, just use the raw text
    return json_output

def inference_with_retry(image_path, categories):
    """Run inference with retry logic for rate limits"""
    retries = 0
    
    # Create category prompt
    category_prompt = ", ".join(categories)
    full_prompt = f"Detect the 2d bounding boxes of the following objects: {category_prompt}"
    
    while retries <= MAX_RETRIES:
        try:
            # Apply rate limiting
            rate_limiter()
            
            # Load and resize image using the tutorial approach
            im = Image.open(BytesIO(open(image_path, "rb").read()))
            original_width, original_height = im.size
            
            # Resize image using thumbnail method as in the tutorial
            im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            input_width, input_height = im.size
            
            # Run model to find bounding boxes
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[full_prompt, im],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    safety_settings=safety_settings,
                    max_output_tokens=8192,
                )
            )
            
            return response.text, input_width, input_height, original_width, original_height, None

        except Exception as e:
            error_msg = str(e)
            retries += 1
            
            # Determine error type for logging
            if is_rate_limit_error(error_msg):
                error_type = "rate limit"
            else:
                error_type = "API"
            
            # If we've exceeded max retries, give up
            if retries > MAX_RETRIES:
                logger.warning(f"Maximum retries ({MAX_RETRIES}) exceeded for {os.path.basename(image_path)}")
                return None, f"{error_type.capitalize()} error exceeded after {MAX_RETRIES} retries: {error_msg}"
            
            # Calculate delay with exponential backoff and jitter
            delay = min(RETRY_DELAY_MAX, RETRY_DELAY_BASE * (2 ** (retries - 1)))
            jitter = random.uniform(0, 0.1 * delay)
            wait_time = delay + jitter
            
            logger.info(f"{error_type.capitalize()} error for {os.path.basename(image_path)}: {error_msg[:100]}... Retry {retries}/{MAX_RETRIES} after {wait_time:.2f}s")
            time.sleep(wait_time)
    
    return None, None, None, None, None, "Maximum retries exceeded"

def convert_coordinates_to_coco_format(boxes, image_id, original_width, original_height, 
                                       input_width, input_height, categories_dict):
    """Convert Gemini bounding box format to COCO format"""
    coco_annotations = []
    
    for idx, box in enumerate(boxes):
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("box_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label = box.get("label", "unknown")
        
        # Find category_id from the label
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label.lower():
                category_id = cat_id
                break
        
        # Skip if no matching category found
        if category_id is None:
            continue
            
        # Gemini format: [y1, x1, y2, x2] where coordinates are normalized to 1000
        # We need to convert to absolute coordinates in the original image
        y1, x1, y2, x2 = bbox_2d
        
        # Convert from normalized coordinates (0-1000) to absolute in input resolution
        abs_y1 = int(y1 * input_height / 1000)
        abs_x1 = int(x1 * input_width / 1000)
        abs_y2 = int(y2 * input_height / 1000)
        abs_x2 = int(x2 * input_width / 1000)
        
        # Adjust for correct ordering if needed
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        # Convert to original image resolution
        x1_orig = int(abs_x1 * original_width / input_width)
        y1_orig = int(abs_y1 * original_height / input_height)
        x2_orig = int(abs_x2 * original_width / input_width)
        y2_orig = int(abs_y2 * original_height / input_height)
        
        # Calculate width and height for COCO format
        width = x2_orig - x1_orig
        height = y2_orig - y1_orig
        
        # Create COCO annotation
        coco_annotation = {
            "id": idx,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1_orig, y1_orig, width, height],
            "area": width * height,
            "segmentation": [],
            "iscrowd": 0,
            "score": 1.0  # Gemini doesn't provide confidence scores
        }
        
        coco_annotations.append(coco_annotation)
    
    return coco_annotations

def visualize_predictions(image_path, boxes, input_width, input_height, 
                        original_width, original_height, categories_dict, save_path):
    """Visualize prediction boxes on the image using supervision library"""
    # Read the image using OpenCV
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning(f"Could not read image at {image_path}. Skipping visualization.")
        return
    
    xyxy = []
    class_ids = []
    confidences = []
    labels = []
    
    for box in boxes:
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("box_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label_text = box.get("label", "unknown")
        
        # Find category_id from the label
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label_text.lower():
                category_id = cat_id
                break
        
        # Skip if no matching category found
        if category_id is None:
            # Use a default category ID for unmatched labels
            category_id = -1
        
        # Gemini format: [y1, x1, y2, x2] where coordinates are normalized to 1000
        y1, x1, y2, x2 = bbox_2d
        
        # Convert from normalized coordinates (0-1000) to absolute in input resolution
        abs_y1 = int(y1 * input_height / 1000)
        abs_x1 = int(x1 * input_width / 1000)
        abs_y2 = int(y2 * input_height / 1000)
        abs_x2 = int(x2 * input_width / 1000)
        
        # Adjust for correct ordering if needed
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        # Convert to original image resolution
        x1_orig = int(abs_x1 * original_width / input_width)
        y1_orig = int(abs_y1 * original_height / input_height)
        x2_orig = int(abs_x2 * original_width / input_width)
        y2_orig = int(abs_y2 * original_height / input_height)
        
        xyxy.append([x1_orig, y1_orig, x2_orig, y2_orig])
        class_ids.append(category_id)
        confidences.append(1.0)  # Gemini doesn't provide confidence scores
        labels.append(label_text)
    
    # If no boxes to visualize, save the original image
    if not xyxy:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        return
    
    # Convert to numpy arrays
    xyxy = np.array(xyxy, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    confidences = np.array(confidences, dtype=np.float32)
    
    # Create supervision detections object
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids
    )
    
    # Create annotators
    box_annotator = sv.BoxAnnotator(
        color=sv.Color.ROBOFLOW, 
        thickness=2
    )
    
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.ROBOFLOW,
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1
    )
    
    # Draw bounding boxes
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    
    # Draw labels
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    
    # Save the annotated image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, annotated_frame)

def load_existing_results(results_file):
    """Load existing results from a file if it exists"""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing results file {results_file}. Creating backup.")
            # Create a backup of the corrupted file
            backup_file = f"{results_file}.bak.{int(time.time())}"
            os.rename(results_file, backup_file)
            return []
    return []

def get_processed_image_ids(existing_results):
    """Get the set of image IDs that have already been processed"""
    return set(annotation["image_id"] for annotation in existing_results)

def analyze_existing_results(existing_results, all_image_ids):
    """Analyze existing results to find complete and incomplete results"""
    # Group annotations by image_id
    annotations_by_image = {}
    for annotation in existing_results:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # Find images with annotations
    images_with_annotations = set(annotations_by_image.keys())
    
    # Find images without annotations
    images_without_annotations = all_image_ids - images_with_annotations
    
    return images_with_annotations, images_without_annotations

def process_image(args):
    """Process a single image - this function will be run in parallel"""
    (image_info, image_id, file_name, original_height, original_width, 
     test_folder, categories, categories_dict, results_dir, vis_dir, 
     processed_image_ids) = args
    
    # Skip if this image has already been processed
    if image_id in processed_image_ids:
        logger.info(f"Image {image_id} ({file_name}) already processed. Skipping.")
        return None, None
    
    # Find image file
    image_path = os.path.join(test_folder, file_name)
    if not os.path.exists(image_path):
        # Try to find in nested folders
        image_files = glob.glob(os.path.join(test_folder, "**", file_name), recursive=True)
        if image_files:
            image_path = image_files[0]
        else:
            return None, f"Image file not found: {file_name}"
    
    try:
        logger.info(f"Processing image: {file_name} (ID: {image_id})")
        
        # Run inference with retry logic
        response, input_width, input_height, original_width, original_height, error = inference_with_retry(
            image_path, categories
        )
        
        if response:
            # Parse the JSON response
            json_content = parse_json(response)
            try:
                boxes = json.loads(json_content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for {file_name}: {json_content}")
                return None, f"Failed to parse JSON for {file_name}"
            
            # Convert to COCO format
            coco_annotations = convert_coordinates_to_coco_format(
                boxes, image_id, original_width, original_height,
                input_width, input_height, categories_dict
            )
            
            # If we got valid annotations
            if coco_annotations:
                # Visualize predictions
                vis_save_path = os.path.join(vis_dir, file_name)
                visualize_predictions(
                    image_path, boxes, input_width, input_height, 
                    original_width, original_height, categories_dict, vis_save_path
                )
                
                return coco_annotations, None
            else:
                logger.warning(f"Got response but no valid annotations for {file_name}")
                return None, f"No valid annotations for {file_name}"
        else:
            return None, f"Failed to get response for {file_name}: {error}"
            
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}", exc_info=True)
        return None, f"Error processing {file_name}: {e}"

def process_dataset(dataset_dir):
    """Process a single dataset with smarter handling of existing results"""
    dataset_name = os.path.basename(dataset_dir)
    
    # Define output directories
    results_dir = os.path.join(OUTPUT_DIR, dataset_name)
    merged_results_dir = os.path.join(MERGED_OUTPUT_DIR, dataset_name)
    vis_dir = os.path.join(VISUALIZE_DIR, dataset_name)
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(merged_results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    test_folder = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_folder):
        logger.error(f"Test folder not found in {dataset_dir}")
        return None
        
    # Find annotation file
    annotation_file = glob.glob(os.path.join(test_folder, "*_annotations.coco.json"))
    if not annotation_file:
        logger.error(f"Annotation file not found in {test_folder}")
        return None
        
    annotation_file = annotation_file[0]
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Extract category names
    categories = annotations.get("categories", [])
    category_names = [cat["name"] for cat in categories]
    categories_dict = {cat["id"]: cat["name"] for cat in categories}
    
    # Get all images
    images = annotations.get("images", [])
    all_image_ids = {image_info["id"] for image_info in images}

    
    # Load existing results
    results_file = os.path.join(results_dir, f"gemini_detection_results.json")
    existing_results = load_existing_results(results_file)
    
    # Get all processed image IDs
    processed_image_ids = get_processed_image_ids(existing_results)
    
    # Analyze existing results to find complete and incomplete annotations
    images_with_annotations, images_without_annotations = analyze_existing_results(
        existing_results, all_image_ids)
    
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Looking for {len(category_names)} categories: {category_names}")
    logger.info(f"Total images: {len(images)}")
    logger.info(f"Images with existing annotations: {len(images_with_annotations)}")
    logger.info(f"Images needing processing: {len(images_without_annotations)}")
    
    # If all images already have annotations, just copy to merged folder
    if len(images_without_annotations) == 0:
        logger.info(f"All images already have valid annotations for dataset {dataset_name}. Nothing to do.")
        
        # Copy existing results to merged folder
        merged_results_file = os.path.join(merged_results_dir, f"gemini_detection_results.json")
        with open(merged_results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
            
        return results_file, existing_results
    
    # Create arguments for parallel processing
    args_list = [
        (image_info, image_info["id"], image_info["file_name"], 
         image_info["height"], image_info["width"], test_folder,  category_names, categories_dict, results_dir, vis_dir, processed_image_ids)
        for image_info in images
    ]
    
    # Initialize new results with existing ones
    new_results = existing_results.copy()
    
    # Only process images that have no annotations yet or have incomplete annotations
    remaining_args = [args for args in args_list if args[1] in images_without_annotations]
    
    # Process remaining images in parallel with tqdm progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_image, args) for args in remaining_args]
        
        with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="image") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()
                if result:
                    new_results.extend(result)
                if error:
                    pbar.write(error)
                pbar.update(1)
    
    # Save results to both original and merged directories
    with open(results_file, 'w') as f:
        json.dump(new_results, f, indent=2)
        
    merged_results_file = os.path.join(merged_results_dir, f"gemini_detection_results.json")
    with open(merged_results_file, 'w') as f:
        json.dump(new_results, f, indent=2)
        
    logger.info(f"Saved results to {results_file} and {merged_results_file}")
    logger.info(f"Final results: {len(new_results)} annotations")
    
    return results_file, new_results

def validate_merged_results():
    """Validate and merge all results at the end to ensure completeness"""
    logger.info("Validating and finalizing merged results...")
    
    dataset_dirs = [d for d in glob.glob("rf100_datasets/*") 
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, "test"))]
    
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        
        # Get paths
        original_results_file = os.path.join(OUTPUT_DIR, dataset_name, "gemini_detection_results.json")
        merged_results_file = os.path.join(MERGED_OUTPUT_DIR, dataset_name, "gemini_detection_results.json")
        
        # Ensure the merged directory exists
        os.makedirs(os.path.dirname(merged_results_file), exist_ok=True)
        
        # Load original results if they exist
        original_results = []
        if os.path.exists(original_results_file):
            try:
                with open(original_results_file, 'r') as f:
                    original_results = json.load(f)
                logger.info(f"Loaded {len(original_results)} annotations from original results for {dataset_name}")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse original results file for {dataset_name}")
        
        # If the merged file doesn't exist or is empty, copy the original
        if not os.path.exists(merged_results_file) or os.path.getsize(merged_results_file) == 0:
            with open(merged_results_file, 'w') as f:
                json.dump(original_results, f, indent=2)
            logger.info(f"Created merged results file for {dataset_name} with {len(original_results)} annotations")
    
    logger.info("Validation and finalization of merged results complete")

def main():
    """Main function to process all datasets"""
    parser = argparse.ArgumentParser(description='Process datasets for object detection')
    parser.add_argument('--with_instructions', action='store_true', 
                        help='Use custom instructions instead of default prompt')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZE_DIR, exist_ok=True)
    
    # Find all dataset directories
    dataset_dirs = [d for d in glob.glob("rf100_datasets/*") 
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, "test"))]
    
    if not dataset_dirs:
        logger.error("No datasets found. Make sure datasets are in the correct directory with 'test' subdirectories.")
        return
        
    logger.info(f"Found {len(dataset_dirs)} datasets")
    
    # Process each dataset
    for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets", unit="dataset"):
        try:
            logger.info(f"Starting processing dataset: {os.path.basename(dataset_dir)}")
            process_dataset(dataset_dir)
            logger.info(f"Finished processing dataset: {os.path.basename(dataset_dir)}")
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_dir}: {e}", exc_info=True)
    
    # Final validation to ensure all results are properly merged
    validate_merged_results()
    
    logger.info("All datasets processed successfully!")

if __name__ == "__main__":
    main()