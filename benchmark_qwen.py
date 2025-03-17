import os
import json
import glob
import base64
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import time
from datetime import datetime
from openai import OpenAI
from qwen_vl_utils import smart_resize
import supervision as sv
from tqdm import tqdm
import concurrent.futures
import threading

# Set your API key here or use environment variable
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'
API_KEY = os.getenv('DASHSCOPE_API_KEY')

# Configuration
MODEL_ID = "qwen2.5-vl-72b-instruct"
MIN_PIXELS = 512*28*28
MAX_PIXELS = 2048*28*28
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT_DIR = "qwen_detection_results"
VISUALIZE_DIR = "qwen_visualized_predictions"
MAX_WORKERS = 16  # Maximum number of parallel workers for image processing
REQUEST_LIMIT = 120  # Maximum number of API requests per minute (adjust based on API limits)

# Thread-safe rate limiter for API requests
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
                    time.sleep(sleep_time)
                    now = time.time()
                self.calls = self.calls[1:]  # Remove the oldest call
                
            # Add the current call time
            self.calls.append(now)

# Create a rate limiter instance
rate_limiter = RateLimiter(REQUEST_LIMIT)

def encode_image(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def inference_with_api(image_path, prompt, sys_prompt=SYSTEM_PROMPT, model_id=MODEL_ID, 
                       min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Run inference using Alibaba Dashscope API for Qwen2.5 VL"""
    # Apply rate limiting
    rate_limiter()
    
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return None

def parse_qwen_response(response):
    """Parse the JSON response from Qwen model"""
    if not response:
        return []
    
    # Extract JSON content from the response
    try:
        # If the response has markdown-style JSON fencing
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0].strip()
        # If it's just a JSON string
        else:
            json_text = response.strip()
            
        # Remove any trailing commas before parsing
        json_text = json_text.replace(",\n]", "\n]").replace(",]", "]")
        
        # Parse the JSON
        boxes = json.loads(json_text)
        return boxes
    except Exception as e:
        print(f"Failed to parse response: {e}")
        print(f"Response: {response}")
        return []

def convert_to_coco_format(qwen_predictions, image_id, original_width, original_height, 
                           input_width, input_height, categories_dict):
    """Convert Qwen detection results to COCO format"""
    coco_annotations = []
    
    # Parse the Qwen predictions
    boxes = parse_qwen_response(qwen_predictions)
    
    for idx, box in enumerate(boxes):
        bbox_2d = box.get("bbox_2d", None)
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
            
        # Convert coordinates from input resolution to original image resolution
        x1, y1, x2, y2 = bbox_2d
        x1 = int(x1 / input_width * original_width)
        y1 = int(y1 / input_height * original_height)
        x2 = int(x2 / input_width * original_width)
        y2 = int(y2 / input_height * original_height)
        
        # COCO format uses [x, y, width, height]
        width = x2 - x1
        height = y2 - y1
        
        # Create COCO annotation
        coco_annotation = {
            "id": idx,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "segmentation": [],
            "iscrowd": 0,
            "score": 1.0  # Qwen doesn't provide confidence scores
        }
        
        coco_annotations.append(coco_annotation)
    
    return coco_annotations, boxes

def visualize_predictions(image_path, boxes, input_width, input_height, 
                          original_width, original_height, categories_dict, save_path):
    """Visualize prediction boxes on the image using supervision library"""
    # Read the image using OpenCV
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"WARNING: Could not read image at {image_path}. Skipping visualization.")
        return
    
    xyxy = []
    class_ids = []
    confidences = []
    labels = []
    
    for box in boxes:
        bbox_2d = box.get("bbox_2d", None)
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
        
        # Convert coordinates from input resolution to original image resolution
        x1, y1, x2, y2 = bbox_2d
        x1 = int(x1 / input_width * original_width)
        y1 = int(y1 / input_height * original_height)
        x2 = int(x2 / input_width * original_width)
        y2 = int(y2 / input_height * original_height)
        
        xyxy.append([x1, y1, x2, y2])
        class_ids.append(category_id)
        confidences.append(1.0)  # Qwen doesn't provide confidence scores
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

def process_image(args):
    """Process a single image - this function will be run in parallel"""
    (image_info, image_id, file_name, original_height, original_width, 
     test_folder, prompt, categories_dict, results_dir, vis_dir) = args
    
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
        # Open image and get dimensions
        image = Image.open(image_path)
        width, height = image.size
        
        # Smart resize for API
        input_height, input_width = smart_resize(height, width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        
        # Run inference
        response = inference_with_api(image_path, prompt)
        
        if response:
            # Convert to COCO format and get original boxes for visualization
            coco_annotations, original_boxes = convert_to_coco_format(
                response, image_id, original_width, original_height,
                input_width, input_height, categories_dict
            )
            
            # Visualize predictions
            vis_save_path = os.path.join(vis_dir, file_name)
            visualize_predictions(
                image_path, original_boxes, input_width, input_height, 
                original_width, original_height, categories_dict, vis_save_path
            )
            
            return coco_annotations, None
        else:
            return None, f"Failed to get response for {file_name}"
            
    except Exception as e:
        return None, f"Error processing {file_name}: {e}"

def process_dataset(dataset_dir):
    """Process a single dataset"""
    test_folder = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_folder):
        print(f"Test folder not found in {dataset_dir}")
        return
        
    # Find annotation file
    annotation_file = glob.glob(os.path.join(test_folder, "*_annotations.coco.json"))
    if not annotation_file:
        print(f"Annotation file not found in {test_folder}")
        return
        
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
    
    # Create prompt with all category names
    category_prompt = ", ".join(category_names)
    prompt = f"Outline the position of each {category_prompt} and output all the coordinates in JSON format."
    
    # Prepare results
    dataset_name = os.path.basename(dataset_dir)
    results_dir = os.path.join(OUTPUT_DIR, dataset_name)
    vis_dir = os.path.join(VISUALIZE_DIR, dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize COCO results
    coco_results = []
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Looking for {len(category_names)} categories: {category_names}")
    print(f"Found {len(images)} images to process")
    
    # Create arguments for parallel processing
    args_list = [
        (image_info, image_info["id"], image_info["file_name"], 
         image_info["height"], image_info["width"], test_folder, 
         prompt, categories_dict, results_dir, vis_dir)
        for image_info in images
    ]
    
    # Process images in parallel with tqdm progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_image, args) for args in args_list]
        
        with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="image") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()
                if result:
                    coco_results.extend(result)
                if error:
                    pbar.write(error)
                pbar.update(1)
    
    # Save results
    results_file = os.path.join(results_dir, f"qwen_detection_results.json")
    with open(results_file, 'w') as f:
        json.dump(coco_results, f, indent=2)
        
    print(f"Saved results to {results_file}")
    return results_file, coco_results

def main():
    """Main function to process all datasets"""
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZE_DIR, exist_ok=True)
    
    dataset_dirs = [d for d in glob.glob("rf100_datasets/*") if os.path.isdir(d) and os.path.exists(os.path.join(d, "test"))]
    
    if not dataset_dirs:
        print("No datasets found. Make sure datasets are in the correct directory with 'test' subdirectories.")
        return
        
    print(f"Found {len(dataset_dirs)} datasets")
    
    # Process each dataset with a progress bar
    for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets", unit="dataset"):
        try:
            process_dataset(dataset_dir)
        except Exception as e:
            print(f"Error processing dataset {dataset_dir}: {e}")
    
    print("All datasets processed successfully!")

if __name__ == "__main__":
    main()