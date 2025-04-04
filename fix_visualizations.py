import os
import json
import glob
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
import argparse
from tqdm import tqdm

def calculate_resize_dimensions(original_width, original_height, target_size=768):
    """
    Calculate dimensions when resizing to fit target_size while preserving aspect ratio.
    Also returns padding information.
    """
    # Calculate resize dimensions preserving aspect ratio
    if original_width > original_height:
        # Landscape image
        resized_width = target_size
        resized_height = int(original_height * target_size / original_width)
    else:
        # Portrait image
        resized_height = target_size
        resized_width = int(original_width * target_size / original_height)
    
    # Calculate padding
    pad_top = (target_size - resized_height) // 2
    pad_bottom = target_size - resized_height - pad_top
    pad_left = (target_size - resized_width) // 2
    pad_right = target_size - resized_width - pad_left
    
    return {
        'resized_width': resized_width,
        'resized_height': resized_height,
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
        'pad_left': pad_left,
        'pad_right': pad_right
    }

def convert_model_to_original_coordinates(detection, original_width, original_height):
    """
    Convert coordinates from the model space (768x768) back to original image coordinates,
    properly accounting for aspect ratio preservation and padding.
    """
    # Get model detection coordinates (COCO format: [x, y, w, h])
    x, y, w, h = detection["bbox"]
    
    # Calculate resize dimensions and padding
    resize_info = calculate_resize_dimensions(original_width, original_height)
    
    # Remove padding (transform from padded 768x768 to resized image space)
    x_in_resized = x - resize_info['pad_left']
    y_in_resized = y - resize_info['pad_top']
    
    # Handle edge cases where the detection is in the padding area
    if x_in_resized < 0:
        w = max(0, w + x_in_resized)
        x_in_resized = 0
    
    if y_in_resized < 0:
        h = max(0, h + y_in_resized)
        y_in_resized = 0
    
    # Scale from resized image back to original image coordinates
    x_in_original = x_in_resized * original_width / resize_info['resized_width']
    y_in_original = y_in_resized * original_height / resize_info['resized_height']
    w_in_original = w * original_width / resize_info['resized_width']
    h_in_original = h * original_height / resize_info['resized_height']
    
    return [x_in_original, y_in_original, w_in_original, h_in_original]

def visualize_detection(image_path, detections, output_path, original_dimensions=None):
    """
    Visualize detection boxes on the image using supervision library with aspect-ratio
    aware coordinate transformation.
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"WARNING: Could not read image at {image_path}. Skipping visualization.")
        return False
    
    # Get the actual dimensions of the image
    img_height, img_width = frame.shape[:2]
    
    # Prepare data for supervision
    xyxy = []
    confidences = []
    class_ids = []
    labels = []
    
    # Create a mapping for category names to class IDs
    categories = {}
    for detection in detections:
        category_name = detection.get("category_name", "unknown")
        if category_name not in categories:
            categories[category_name] = len(categories)
    
    # Process each detection
    for detection in detections:
        # Get the category name and class ID
        category_name = detection.get("category_name", "unknown")
        class_id = categories.get(category_name, -1)
        
        # Transform the coordinates with aspect ratio awareness
        try:
            # For detections directly from model (need to be converted)
            bbox = detection.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            
            # Convert from model space to original image space
            x, y, w, h = convert_model_to_original_coordinates(
                detection, img_width, img_height
            )
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Add to visualization data
            xyxy.append([x, y, x + w, y + h])
            confidences.append(float(detection.get("score", 1.0)))
            class_ids.append(class_id)
            labels.append(category_name)
        except Exception as e:
            print(f"Error processing detection {detection}: {e}")
    
    # If no valid boxes to visualize, save the original image
    if not xyxy:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        return False
    
    # Convert to numpy arrays
    xyxy = np.array(xyxy, dtype=np.float32)
    confidences = np.array(confidences, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    
    # Create supervision detections object
    sv_detections = sv.Detections(
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
        detections=sv_detections
    )
    
    # Draw labels
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=sv_detections,
        labels=labels
    )
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_frame)
    return True

def visualize_with_debug(image_path, detections, output_path):
    """
    Visualize detection with additional debug information to help diagnose coordinate issues.
    Creates side-by-side comparison of different coordinate transformation methods.
    """
    # Read the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"WARNING: Could not read image at {image_path}. Skipping visualization.")
        return False
    
    img_height, img_width = original_image.shape[:2]
    
    # Create three copies of the image
    images = {
        "Original": original_image.copy(),
        "Direct Scaling": original_image.copy(),
        "Aspect-Aware": original_image.copy()
    }
    
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
    
    # Process each detection with different methods
    for method_name, image in images.items():
        if method_name == "Original":
            # Skip drawing on the original image
            continue
        
        xyxy = []
        confidences = []
        class_ids = []
        labels = []
        
        # Create a mapping for category names to class IDs
        categories = {}
        for detection in detections:
            category_name = detection.get("category_name", "unknown")
            if category_name not in categories:
                categories[category_name] = len(categories)
        
        # Process each detection
        for detection in detections:
            category_name = detection.get("category_name", "unknown")
            class_id = categories.get(category_name, -1)
            bbox = detection.get("bbox", [0, 0, 0, 0])
            
            if len(bbox) != 4:
                continue
            
            if method_name == "Direct Scaling":
                # Original method from your code
                x, y, w, h = bbox
            else:  # "Aspect-Aware"
                # New aspect-aware transformation
                x, y, w, h = convert_model_to_original_coordinates(
                    detection, img_width, img_height
                )
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            xyxy.append([x, y, x + w, y + h])
            confidences.append(float(detection.get("score", 1.0)))
            class_ids.append(class_id)
            labels.append(f"{category_name} ({method_name})")
        
        # If no valid boxes to visualize, continue
        if not xyxy:
            continue
        
        # Convert to numpy arrays
        xyxy = np.array(xyxy, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        # Create supervision detections object
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )
        
        # Draw bounding boxes
        image = box_annotator.annotate(
            scene=image,
            detections=sv_detections
        )
        
        # Draw labels
        image = label_annotator.annotate(
            scene=image,
            detections=sv_detections,
            labels=labels
        )
        
        # Update the image in the dictionary
        images[method_name] = image
    
    # Create a concatenated debug image
    direct_scaling_img = images["Direct Scaling"]
    aspect_aware_img = images["Aspect-Aware"]
    
    # Add method name to the top of each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(direct_scaling_img, "Direct Scaling", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(aspect_aware_img, "Aspect-Aware", (10, 30), font, 1, (0, 0, 255), 2)
    
    # Concatenate horizontally
    debug_image = np.hstack((direct_scaling_img, aspect_aware_img))
    
    # Ensure the directory exists
    debug_dir = os.path.dirname(output_path)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save the debug image
    debug_output_path = output_path.replace(".jpg", "_debug.jpg")
    cv2.imwrite(debug_output_path, debug_image)
    
    # Also save the aspect-aware version as the main output
    cv2.imwrite(output_path, aspect_aware_img)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate corrected visualizations from existing prediction files')
    parser.add_argument('--predictions-dir', type=str, default="4.5-results", 
                      help='Directory containing prediction JSON files')
    parser.add_argument('--dataset-root', type=str, default="rf100_datasets",
                      help='Root directory containing all datasets')
    parser.add_argument('--output-dir', type=str, default="4.5-correct-visualizations",
                      help='Directory to save corrected visualizations')
    parser.add_argument('--filter', type=str, default=None,
                      help='Optional filter to process only specific datasets (comma-separated)')
    parser.add_argument('--debug', action='store_true',
                      help='Create side-by-side debug visualizations showing both methods')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all prediction files
    prediction_files = glob.glob(os.path.join(args.predictions_dir, "*_prediction.json"))
    print(f"Found {len(prediction_files)} prediction files")
    
    # Apply filter if specified
    if args.filter:
        filter_items = [f.strip() for f in args.filter.split(",")]
        prediction_files = [f for f in prediction_files if any(item in f for item in filter_items)]
        print(f"After filtering: {len(prediction_files)} prediction files")
    
    # Process each prediction file
    successful_count = 0
    for pred_file in tqdm(prediction_files, desc="Processing predictions"):
        try:
            # Load prediction data
            with open(pred_file, 'r') as f:
                prediction_data = json.load(f)
            
            # Extract dataset and image information
            dataset_name = prediction_data.get("dataset", "unknown")
            image_filename = prediction_data.get("image", "unknown.jpg")
            detections = prediction_data.get("detections", [])
            
            # Check if we have any detections
            if not detections:
                print(f"No detections found for {dataset_name} - {image_filename}")
                continue
            
            # Find the image file in the dataset
            dataset_dir = os.path.join(args.dataset_root, dataset_name)
            test_folder = os.path.join(dataset_dir, "test")
            
            # First look for exact filename match
            image_path = os.path.join(test_folder, image_filename)
            
            # If not found, search recursively
            if not os.path.exists(image_path):
                image_files = glob.glob(os.path.join(test_folder, "**", image_filename), recursive=True)
                if image_files:
                    image_path = image_files[0]
                else:
                    print(f"Image file not found for {dataset_name} - {image_filename}")
                    continue
            
            # Create output directory for this dataset
            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Output visualization path
            output_path = os.path.join(dataset_output_dir, f"{image_filename}_corrected.jpg")
            
            # Generate visualization with debug option if specified
            if args.debug:
                success = visualize_with_debug(image_path, detections, output_path)
            else:
                success = visualize_detection(image_path, detections, output_path)
                
            if success:
                successful_count += 1
                
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {len(prediction_files)} prediction files")
    print(f"Successfully generated {successful_count} visualizations")
    print(f"Visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()