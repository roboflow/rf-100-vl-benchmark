import os
import re
import json
import time
import glob
import base64
import cv2
import supervision as sv
from pathlib import Path
import numpy as np

#########################
# 1) DETECTION PARSING  #
#########################
def parse_output_to_detections(output_text, original_image_dimensions):
    """
    Robust parser for an output that looks like:
    ```json
    [
      {"bbox": [x1, y1, x2, y2], "score": *confidence_score_0-1*, "label": *one_of_the_classes*},
      ...
    ]
    ```
    1) Removes code fences.
    2) Fixes missing colons in `bbox [...]`.
    3) Parses the entire string as JSON (expecting a top-level list).
    4) If JSON parsing fails (e.g. due to truncation), it falls back to extracting individual JSON objects.
    5) Iterates each item; if one is malformed, it is skipped. Others are still accepted.

    Returns a list of detections, each:
      {"bbox": [x1, y1, x2, y2], "score": float, "category_name": str}
    """
    import re  # Make sure `re` is imported (should be at top-level too, but just in case).
    # Remove code fences and fix "bbox" formatting
    text_clean = re.sub(r'```(?:json)?\s*', '', output_text)
    text_clean = text_clean.replace('```', '')
    text_clean = re.sub(r'"bbox\s*\[(.*?)\]', r'"bbox":[\1]', text_clean)

    detections = []
    data = []

    # Try to parse as pure JSON first
    try:
        data = json.loads(text_clean)
        if not isinstance(data, list):
            print("Top-level JSON is not a list; skipping.")
            return detections
    except json.JSONDecodeError:
        print("Malformed JSON detected, attempting to salvage valid detections.")
        # Fallback: extract individual JSON objects using a regex.
        object_strings = re.findall(r'\{[^}]+\}', text_clean)
        for obj_str in object_strings:
            try:
                obj = json.loads(obj_str)
                data.append(obj)
            except json.JSONDecodeError:
                print(f"Skipping malformed object: {obj_str}")
        print("Salvaged the following json objects: ", data)
        #save output_text to a file
        if not os.path.exists("output_text.txt"):
            with open("output_text.txt", "w") as f:
                f.write("")
        #append the output_text to the file
        with open("output_text.txt", "a") as f:
            f.write(output_text)

    # Process each item in data (which should now be a list of dicts)
    for item in data:
        try:
            if not isinstance(item, dict):
                print(f"Skipping item (not a dict): {item}")
                continue

            bbox = item.get("bbox", [])
            if len(bbox) != 4:
                print(f"Skipping invalid bbox length: {bbox}")
                continue

            # Convert bbox coordinates to floats
            # The model might still be outputting [x1, y1, w, h], so check and convert if needed
            x1, y1, x2_or_w, y2_or_h = map(float, bbox)

            # Check if the format is [x1, y1, w, h] by comparing the last two values
            # If x2_or_w < x1 or y2_or_h < y1, it's likely [x1, y1, w, h]
            if x2_or_w < x1 or y2_or_h < y1:
                # It's in the old format, convert from [x1, y1, w, h] to [x1, y1, x2, y2]
                x2 = x1 + x2_or_w
                y2 = y1 + y2_or_h
            else:
                # It's already in [x1, y1, x2, y2] format
                x2 = x2_or_w
                y2 = y2_or_h

            # Validate bounding box
            if x2 < x1 or y2 < y1:
                print(f"Skipping invalid bbox coordinates: {bbox}")
                continue
            if x2 == x1 or y2 == y1:
                print(f"Skipping zero dimension bbox: {bbox}")
                continue

            label = item.get("label", "unknown")
            if label == "unknown":
                print(f"Skipping item (label is unknown): {item}")
                continue

            # Scale bbox back to original image dimensions
            orig_w, orig_h = original_image_dimensions
            x1 = x1 * orig_w / 768
            y1 = y1 * orig_h / 768
            x2 = x2 * orig_w / 768
            y2 = y2 * orig_h / 768

            detections.append({
                "bbox": [x1, y1, x2, y2],  # Now in x1, y1, x2, y2 format
                "score": 1.0,
                "category_name": label
            })

        except Exception as e:
            print(f"Skipping item due to error: {item}, error: {e}")
            # Save the item to a file for debugging
            if not os.path.exists("failed_detections.txt"):
                with open("failed_detections.txt", "w") as f:
                    f.write("")
            with open("failed_detections.txt", "a") as f:
                f.write(str(item) + "\n")

    return detections

def visualize_detections(image_path, detections, output_path):
    """
    Visualize detection boxes on the image using supervision library.
    
    Args:
        image_path (str): Path to the original image
        detections (list): List of detection dictionaries with bbox (x1,y1,x2,y2), score, and category_name
        output_path (str): Path to save the visualized image
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"WARNING: Could not read image at {image_path}. Skipping visualization.")
        return
    
    # Prepare data for supervision
    xyxy = []
    confidences = []
    class_ids = []
    labels = []
    
    # Create a mapping for category names to class IDs
    categories = {}
    for i, detection in enumerate(detections):
        category_name = detection["category_name"]
        if category_name not in categories:
            categories[category_name] = len(categories)
    
    # Use bbox directly since they're already in [x1, y1, x2, y2] format
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        xyxy.append([x1, y1, x2, y2])
        confidences.append(float(detection["score"]))
        class_id = categories[detection["category_name"]]
        class_ids.append(class_id)
        labels.append(detection["category_name"])
    
    # If no boxes to visualize, save the original image
    if not xyxy:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        return
    
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
    print(f"Visualization saved to: {output_path}")

#################################
# 2) IMAGE ENCODING HELPER FUNC #
#################################
def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes the image file into a base64 string.
    """
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return b64_string


#########################
# 3) MAIN LOGIC         #
#########################
def main():
    # ---------------------------------------------
    # NEW: Migrate from global openai.* to client
    # ---------------------------------------------
    from openai import OpenAI

    # Create an OpenAI client instance
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")  # or specify directly
        # You can also set base_url, default_headers, etc. here if needed
    )

    dataset_root = "rf100_datasets"

    results_dir = "4.5-alt-results"
    vis_dir = "4.5-alt-visualizations"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    all_dataset_times = []
    results_log = []
    num_datasets_processed = 0
    num_valid_results = 0

    # List all subdirectories under dataset_root
    for dataset_name in os.listdir(dataset_root):
        #check if the prediction file already exists
        #out_json_path = os.path.join(results_dir, f"{dataset_name}_prediction.json")
        #if os.path.exists(out_json_path):
        #    print(f"Prediction file already exists for {dataset_name}, skipping.")
        #    continue

        dataset_dir = os.path.join(dataset_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue  # Skip files that are not directories

        test_folder = os.path.join(dataset_dir, "test")
        if not os.path.isdir(test_folder):
            continue

        annotations_path = os.path.join(test_folder, "_annotations.coco.json")
        if not os.path.isfile(annotations_path):
            print(f"No _annotations.coco.json for dataset: {dataset_name}, skipping.")
            continue

        # Parse the categories from the annotations file
        with open(annotations_path, "r") as f:
            coco_data = json.load(f)
        categories = coco_data.get("categories", [])
        all_classes = [cat["name"] for cat in categories]

        # Find an image in the test folder (just pick one)
        image_paths = glob.glob(os.path.join(test_folder, "*.*"))
        if not image_paths:
            print(f"No images found in {test_folder}, skipping dataset {dataset_name}.")
            continue

        image_path = image_paths[0]

        # Original image dimensions from the first entry in coco_data["images"]
        # (Make sure it matches the actual image you are choosing to read!)
        orig_w = coco_data["images"][0]["width"]
        orig_h = coco_data["images"][0]["height"]
        original_image_dimensions = (orig_w, orig_h)

        # Read the image, resize to 768x768, then save it to a new file for the prompt
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image at {image_path}, skipping dataset {dataset_name}.")
            continue
        image_resized = cv2.resize(image, (768, 768))
        resized_image_path = os.path.join(test_folder, "resized_image.jpg")
        cv2.imwrite(resized_image_path, image_resized)

        # Encode the resized image
        base64_image = encode_image_to_base64(resized_image_path)

        categories_string = ", ".join([cat["name"] for cat in categories])

        # Update prompt to indicate the new format
        prompt_text = (
            f"Detect each of the following objects: {categories_string}, in the image. "
            "The image resolution is 768x768 pixels. Your response should be only the list of their locations coordinates in format [x1, y1, x2, y2] where x1,y1 is the top left corner and x2,y2 is the bottom right corner, EXACT same as below:\n"
            "[{\"bbox\":[x1,y1,x2,y2],\"label\":\"*one of the classes above*\"}, ...]"
        )

        content_items = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
            },
        ]

        messages_payload = [
            {"role": "user", "content": content_items}
        ]

        # Save the messages_payload to a file (for debugging/inspection)
        with open("messages_payload.json", "w") as f:
            json.dump(messages_payload, f, indent=2)

        # --------------------------------------------------
        # Example of calling the new openai v1.0.0+ library
        # --------------------------------------------------
        start_time = time.time()
        try:
            num_datasets_processed += 1
            # Replaces: openai.ChatCompletion.create(...)
            # Now: client.chat.completions.create(...)
            response = client.chat.completions.create(
                model="gpt-4.5-preview",  # Replace with the valid model you're using
                messages=messages_payload,
                max_tokens=10000,
            )
            end_time = time.time()
            elapsed = end_time - start_time
            all_dataset_times.append(elapsed)

            # `response` is a pydantic model with .choices, etc.
            if response.choices:
                output_text = response.choices[0].message.content
                detections = parse_output_to_detections(output_text, original_image_dimensions)

                # Check if we have at least 1 valid detection
                valid = len(detections) > 0
                if valid:
                    num_valid_results += 1

                    # Create a visualization for the valid detection
                    dataset_vis_dir = os.path.join(vis_dir, dataset_name)
                    if not os.path.exists(dataset_vis_dir):
                        os.makedirs(dataset_vis_dir)
                    vis_output_path = os.path.join(dataset_vis_dir, f"{os.path.basename(image_path)}_detection.jpg")
                    
                    # Visualize the detection on the original image
                    visualize_detections(image_path, detections, vis_output_path)

                # Save or log the detection result
                detection_result = {
                    "dataset": dataset_name,
                    "image": os.path.basename(image_path),
                    "time_sec": elapsed,
                    "raw_text_output": output_text,
                    "detections": detections
                }
                results_log.append(detection_result)

                # Save predictions to a file
                out_json_path = os.path.join("4.5-alt-results", f"{dataset_name}_prediction.json")
                if not os.path.exists("4.5-alt-results"):
                    os.makedirs("4.5-alt-results")
                try:
                    with open(out_json_path, "w") as out_f:
                        json.dump(detection_result, out_f, indent=2)
                except Exception as e:
                    print(f"Failed to save prediction file for {dataset_name}: {e}")

                # Show progress
                print(f"\n=== {dataset_name} ===")
                print(f"Time: {elapsed:.3f}s; Valid detection(s): {valid} ({len(detections)} items)")
                print(f"Predictions saved to: {out_json_path}")
            else:
                print(f"No response received from model for {dataset_name}.")

        except Exception as e:
            print(f"Error during OpenAI API call for dataset {dataset_name}: {e}")
            continue

    # Compute and print the average detection time across all datasets
    if all_dataset_times:
        avg_time = sum(all_dataset_times) / len(all_dataset_times)
        print(f"\n=== SUMMARY ===")
        print(f"Processed {num_datasets_processed} dataset(s).")
        print(f"Successful results (>=1 detection): {num_valid_results}")
        print(f"Average time for 1-image detection: {avg_time:.4f} seconds")
    else:
        print("No datasets were processed; unable to compute average time.")

    # Optionally, save the entire results_log to a single JSON
    with open("all_detections_summary.json", "w") as f:
        json.dump(results_log, f, indent=2)


if __name__ == "__main__":
    main()