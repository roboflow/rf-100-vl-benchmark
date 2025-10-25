import re
import time
import json
import random
import cv2
import numpy as np
import supervision as sv
import os
from utils.shared_eval_utils import is_rate_limit_error
from google.genai import types
MAX_WORKERS = 16
REQUEST_LIMIT = 9000
MAX_RETRIES = 3
RETRY_DELAY_BASE = 8
RETRY_DELAY_MAX = 60

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

def visualize_few_shot_ground_truth(train_folder, categories_dict, output_vis_dir, logger):
    """
    Visualizes the ground truth for a specified number of training images
    after converting their COCO annotations to Gemini format and back.

    This helps verify the coordinate conversion process used for few-shot examples.

    Args:
        train_folder (str): Path to the training data folder containing images
                            and '_annotations.coco.json'.
        categories_dict (dict): Mapping from category ID to category name.
        output_vis_dir (str): The base directory where visualization subfolders
                              for datasets are created. The few-shot examples
                              will be saved in a subdirectory here.
    """

    annotation_file = os.path.join(train_folder, "_annotations.coco.json")

    with open(annotation_file, 'r') as f:
        train_data = json.load(f)

    images_by_id = {img['id']: img for img in train_data.get('images', [])}
    annotations_by_image = {}
    for ann in train_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    annotated_image_ids = list(annotations_by_image.keys())

    annotated_image_ids.sort()

    few_shot_vis_save_dir = os.path.join(output_vis_dir, "few_shot_ground_truth_examples")
    os.makedirs(few_shot_vis_save_dir, exist_ok=True)
    logger.info(f"Saving few-shot example visualizations to: {few_shot_vis_save_dir}")

    visualized_count = 0
    for img_id in annotated_image_ids:

        img_info = images_by_id[img_id]
        original_width = img_info['width']
        original_height = img_info['height']
        image_file_name = img_info['file_name']
        image_path = os.path.join(train_folder, image_file_name)

        #Don't save the image if it's already in the few_shot_vis_save_dir
        if os.path.exists(os.path.join(few_shot_vis_save_dir, f"gt_{image_file_name}")):
            continue

        example_annotations = annotations_by_image[img_id]

        gemini_formatted_boxes_for_vis = []
        for ann in example_annotations:
            coco_bbox = ann['bbox']
            cat_id = ann['category_id']
            cat_name = categories_dict.get(cat_id, "unknown_gt")

            gemini_bbox_normalized = convert_coco_to_gemini_format(coco_bbox, original_width, original_height)

            gemini_formatted_boxes_for_vis.append({
                "label": cat_name,
                "box_2d": gemini_bbox_normalized
            })

        save_vis_path = os.path.join(few_shot_vis_save_dir, f"gt_{image_file_name}")

        visualize_predictions(
            image_path=image_path,
            boxes=gemini_formatted_boxes_for_vis,
            input_width=original_width,
            input_height=original_height,
            original_width=original_width,
            original_height=original_height,
            categories_dict=categories_dict,
            save_path=save_vis_path
        )
        logger.info(f"Saved few-shot ground truth visualization for: {image_file_name}")
        visualized_count += 1

    logger.info(f"Successfully visualized {visualized_count} few-shot ground truth examples.")

def format_ground_truth_for_model(annotations_for_image, categories_dict, width, height):
    """
    Formats ground truth annotations for a single image into the 
    JSON string expected in the 'model' turn of a few-shot prompt.
    """
    output_boxes = []
    for ann in annotations_for_image:
        cat_id = ann['category_id']
        cat_name = categories_dict.get(cat_id, "unknown")
        
        gemini_bbox = convert_coco_to_gemini_format(ann['bbox'], width, height)
        
        output_boxes.append({
            "label": cat_name,
            "box_2d": gemini_bbox 
        })
        
    json_string = json.dumps(output_boxes, indent=2) 
    
    formatted_string = f"```json\n{json_string}\n```"
    
    return formatted_string

def parse_json(json_output, logger):
    """Parse the JSON content from the Gemini response"""
    if not json_output:
        return []
        
    try:
        if "```json" in json_output:
            json_text = json_output.split("```json")[1].split("```")[0].strip()
        elif "```" in json_output:
            pattern = r"```(?:\w+)?\s*([\s\S]+?)\s*```"
            matches = re.findall(pattern, json_output)
            if matches:
                json_text = matches[0].strip()
            else:
                json_text = json_output.strip()
        else:
            json_text = json_output.strip()
            
        json_text = re.sub(r'\s+', ' ', json_text)
        json_text = json_text.replace(",]", "]").replace(", ]", "]")
        
        try:
            boxes = json.loads(json_text)
            if not isinstance(boxes, list):
                logger.warning(f"Unexpected JSON format (not a list): {type(boxes)}")
                return []
            return boxes
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        logger.debug(f"Response: {json_output}")
        return []

def convert_coco_to_gemini_format(bbox, width, height):
    """Convert COCO bbox [x, y, w, h] to Gemini format [ymin, xmin, ymax, xmax]"""
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    
    xmin = int(xmin * 1000 / width)
    ymin = int(ymin * 1000 / height)
    xmax = int(xmax * 1000 / width)
    ymax = int(ymax * 1000 / height)
    
    return [ymin, xmin, ymax, xmax]

def inference_with_retry(contents, system_prompt, model_id, logger, rate_limiter, client):
    """
    Run inference with retry logic for rate limits, accepting the full 'contents' list.

    Args:
        contents (list[Content]): The list of Content objects for the API call.
        system_prompt (str): The system instruction for the model.
        model_id (str): The ID of the Gemini model to use.

    Returns:
        tuple[str | None, str | None]: (response_text, error_message)
                                       response_text is None if an error occurred.
                                       error_message is None if successful.
    """
    retries = 0

    while retries <= MAX_RETRIES:
        try:
            rate_limiter()
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    safety_settings=safety_settings,
                    max_output_tokens=8192,
                ),
            )

            response_text = getattr(response, 'text', None)
            if response_text is None:
                 logger.warning(f"API response for model {model_id} did not contain a 'text' attribute.")
                 try:
                      if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                           response_text = response.candidates[0].content.parts[0].text
                           logger.info("Recovered response text from candidate parts.")
                 except Exception as e:
                      logger.error(f"Could not extract text from response parts: {e}. Returning empty.")
                      response_text = ""

            return response_text, None

        except Exception as e:
            error_msg = str(e)
            retries += 1

            error_type = "rate limit" if is_rate_limit_error(error_msg) else "API"

            if retries > MAX_RETRIES:
                error_detail = f"{error_type.capitalize()} error exceeded after {MAX_RETRIES} retries: {error_msg}"
                logger.warning(f"{error_detail[:500]}...")
                return None, error_detail

            delay = min(RETRY_DELAY_MAX, RETRY_DELAY_BASE * (2 ** (retries - 1)))
            jitter = random.uniform(0, 0.1 * delay)
            wait_time = delay + jitter

            logger.info(f"{error_type.capitalize()} error encountered. Retry {retries}/{MAX_RETRIES} after {wait_time:.2f}s. Details: {error_msg[:200]}...")
            time.sleep(wait_time)

    return None, "Maximum retries exceeded without specific error capture (logic error)"

def convert_coordinates_to_coco_format(boxes, image_id, original_width, original_height, 
                                     input_width, input_height, categories_dict):
    """Convert Gemini bounding box format to COCO format"""
    coco_annotations = []
    
    for idx, box in enumerate(boxes):
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("box_2d", box.get("bbox", box.get("bounding_box", box.get("bounding_box_2d", box.get("bbox_2d", None)))))
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label = box.get("label", box.get("class", box.get("name", box.get("class_label", box.get("class_name", "unknown")))))
        
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label.lower():
                category_id = cat_id
                break
        
        if category_id is None:
            continue
            
        y1, x1, y2, x2 = bbox_2d
        
        abs_y1 = int(y1 * input_height / 1000)
        abs_x1 = int(x1 * input_width / 1000)
        abs_y2 = int(y2 * input_height / 1000)
        abs_x2 = int(x2 * input_width / 1000)
        
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        x1_orig = int(abs_x1 * original_width / input_width)
        y1_orig = int(abs_y1 * original_height / input_height)
        x2_orig = int(abs_x2 * original_width / input_width)
        y2_orig = int(abs_y2 * original_height / input_height)
        
        width = x2_orig - x1_orig
        height = y2_orig - y1_orig
        
        coco_annotation = {
            "id": idx,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1_orig, y1_orig, width, height],
            "area": width * height,
            "segmentation": [],
            "iscrowd": 0,
            "score": 1.0
        }
        
        coco_annotations.append(coco_annotation)
    
    return coco_annotations

def visualize_predictions(image_path, boxes, input_width, input_height, 
                        original_width, original_height, categories_dict, save_path):
    """Visualize prediction boxes on the image using supervision library"""
    frame = cv2.imread(image_path)
    
    xyxy = []
    class_ids = []
    confidences = []
    labels = []
    
    for box in boxes:
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("box_2d", box.get("bbox", box.get("bounding_box", box.get("bounding_box_2d", box.get("bbox_2d", None)))))
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label_text = box.get("label", box.get("class", box.get("name", box.get("class_label", box.get("class_name", "unknown")))))
        
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label_text.lower():
                category_id = cat_id
                break
        
        if category_id is None:
            category_id = -1
        
        y1, x1, y2, x2 = bbox_2d
        
        abs_y1 = int(y1 * input_height / 1000)
        abs_x1 = int(x1 * input_width / 1000)
        abs_y2 = int(y2 * input_height / 1000)
        abs_x2 = int(x2 * input_width / 1000)
        
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        x1_orig = int(abs_x1 * original_width / input_width)
        y1_orig = int(abs_y1 * original_height / input_height)
        x2_orig = int(abs_x2 * original_width / input_width)
        y2_orig = int(abs_y2 * original_height / input_height)
        
        xyxy.append([x1_orig, y1_orig, x2_orig, y2_orig])
        class_ids.append(category_id)
        confidences.append(1.0)
        labels.append(label_text)
    
    if not xyxy:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        return
    
    xyxy = np.array(xyxy, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    confidences = np.array(confidences, dtype=np.float32)
    
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids
    )
    
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
    
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, annotated_frame)

def load_existing_results(results_file, logger):
    """Load existing results from a file if it exists"""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing results file {results_file}. Creating backup.")
            backup_file = f"{results_file}.bak.{int(time.time())}"
            os.rename(results_file, backup_file)
            return []
    return []

def get_processed_image_ids(existing_results):
    """Get the set of image IDs that have already been processed"""
    return set(annotation["image_id"] for annotation in existing_results)