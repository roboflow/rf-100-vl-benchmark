import os
import json
import base64
import numpy as np
import cv2
import time
import random
import re
import supervision as sv
from openai import OpenAI
import pickle
from utils.shared_eval_utils import *

MODEL_ID = "qwen2.5-vl-72b-instruct"
MIN_PIXELS = 4*28*28
MAX_PIXELS = 12800*28*28
SYSTEM_PROMPT = "You are a helpful assistant capable of object detection."
MAX_WORKERS = 4
REQUEST_LIMIT = 100
MAX_RETRIES = 2
RETRY_DELAY_BASE = 8
RETRY_DELAY_MAX = 300

def load_processing_status(status_file, logger):
    """Load the processing status using pickle"""
    if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
        try:
            with open(status_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading status: {e}")
            return {}
    return {}

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

def save_processing_status(status_map, status_file, logger):
    """Save the processing status using pickle"""
    temp_file = status_file + ".tmp"
    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(temp_file, 'wb') as f:
            pickle.dump(status_map, f)
        os.replace(temp_file, status_file)
        return True
    except Exception as e:
        logger.error(f"Error saving status: {e}")
        print(f"Error saving status: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    
def is_valid_json(json_text):
    """Check if a string is valid JSON"""
    try:
        if not json_text or not json_text.strip().startswith(('[', '{')):
            return False
        json.loads(json_text)
        return True
    except json.JSONDecodeError:
        return False

def encode_image(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def is_timeout_error(error_msg):
    """Check if the error is a timeout error"""
    return "RequestTimeOut" in str(error_msg) or "timed out" in str(error_msg).lower()

def inference_with_retry(image_path, prompt, sys_prompt=SYSTEM_PROMPT, model_id=MODEL_ID,
                        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, logger=None, rate_limiter=None, API_KEY=None):
    """
    Run inference with retry logic, but DO NOT retry if the error indicates
    'data_inspection_failed'.
    """
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            rate_limiter()
            base64_image = encode_image(image_path)
            if base64_image is None:
                 raise Exception("Failed to encode image")

            client = OpenAI(
                api_key=API_KEY,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
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
            completion = client.chat.completions.create(model=model_id, messages=messages)
            return completion.choices[0].message.content, None
        except Exception as e:
            error_msg = str(e)

            is_data_inspection_error = "'code': 'data_inspection_failed'" in error_msg or \
                                       "data_inspection_failed" in error_msg

            if is_data_inspection_error:
                logger.error(f"Data inspection failed for {os.path.basename(image_path)}. Input image might be corrupted or invalid for the model. No retry will be attempted. Error: {error_msg}")
                return None, f"Data inspection failed: {error_msg}"

            retries += 1
            error_type = "API"

            if is_rate_limit_error(error_msg): error_type = "rate limit"
            elif is_timeout_error(error_msg): error_type = "timeout"

            if retries > MAX_RETRIES:
                logger.warning(f"Max retries ({MAX_RETRIES}) exceeded for {os.path.basename(image_path)} (Last error type: {error_type})")
                return None, f"{error_type.capitalize()} error after {MAX_RETRIES} retries: {error_msg}"

            delay = min(RETRY_DELAY_MAX, RETRY_DELAY_BASE * (2 ** (retries - 1)))
            jitter = random.uniform(0, 0.1 * delay)
            wait_time = delay + jitter

            logger.info(f"{error_type.capitalize()} error for {os.path.basename(image_path)}: {error_msg[:100]}... Retry {retries}/{MAX_RETRIES} after {wait_time:.2f}s")
            time.sleep(wait_time)

    logger.error(f"Exited retry loop unexpectedly for {os.path.basename(image_path)}")
    return None, "Maximum retries logic failed or loop exited unexpectedly"

def convert_to_coco_format(all_parsed_boxes, image_id, original_width, original_height,
                           input_width, input_height, prompted_name_to_id_map, logger):
    """
    Convert a list of aggregated, parsed Qwen detection boxes (from multiple single-class calls)
    to COCO format using the prompted name mapping.

    Args:
        all_parsed_boxes (list): List of dicts, where each dict is a parsed box object
                                 (e.g., {'bbox_2d': [x1,y1,x2,y2], 'label': 'cat'}).
        image_id (int): The ID of the image these annotations belong to.
        original_width (int): Original width of the image.
        original_height (int): Original height of the image.
        input_width (int): Width of the image used as input to the model.
        input_height (int): Height of the image used as input to the model.
        prompted_name_to_id_map (dict): Mapping from prompted name (lowercase) to original category ID.

    Returns:
        tuple: A tuple containing:
            - coco_annotations (list): List of annotations in COCO format.
            - detected_labels_for_vis (list): List of dicts containing box info and Qwen labels
                                              for visualization purposes.
    """
    coco_annotations = []
    detected_labels_for_vis = []

    for idx, box in enumerate(all_parsed_boxes):
        if not isinstance(box, dict):
            logger.warning(f"Skipping invalid box entry (not a dict) in aggregated list for image {image_id}: {box}")
            continue

        bbox_2d = box.get("bbox_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            logger.warning(f"Skipping aggregated box with invalid 'bbox_2d' for image {image_id}: {box}")
            continue

        qwen_label_text = box.get("label", "unknown").strip()
        if qwen_label_text == "unknown":
            logger.warning(f"Skipping aggregated box with 'unknown' label for image {image_id}: {box}")
            continue

        category_id = None
        matched_prompted_name = None
        qwen_label_lower = qwen_label_text.lower()
        if qwen_label_lower in prompted_name_to_id_map:
             category_id = prompted_name_to_id_map[qwen_label_lower]
             for pn, cid in prompted_name_to_id_map.items():
                 if cid == category_id and pn == qwen_label_lower:
                     matched_prompted_name = pn
                     break
        if category_id is None:
            sorted_prompted_names = sorted(prompted_name_to_id_map.keys(), key=len, reverse=True)
            for prompted_name_lower in sorted_prompted_names:
                if prompted_name_lower in qwen_label_lower:
                    category_id = prompted_name_to_id_map[prompted_name_lower]
                    matched_prompted_name = prompted_name_lower # The key from the map
                    logger.debug(f"Substring match found: Qwen label '{qwen_label_text}' contained prompted name '{prompted_name_lower}' -> CatID {category_id}")
                    break

        if category_id is None:
            logger.warning(f"Aggregated box label '{qwen_label_text}' not found in prompted names mapping for image {image_id}. Skipping box.")
            continue

        try:
            x1, y1, x2, y2 = map(float, bbox_2d)
            x1 = int(x1 / input_width * original_width)
            y1 = int(y1 / input_height * original_height)
            x2 = int(x2 / input_width * original_width)
            y2 = int(y2 / input_height * original_height)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width, x2)
            y2 = min(original_height, y2)

            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid box coordinates after scaling/clamping for label '{qwen_label_text}': [{x1},{y1},{x2},{y2}]. Skipping.")
                continue

        except (ValueError, TypeError) as coord_err:
            logger.warning(f"Invalid coordinate types in bbox_2d {bbox_2d} for label '{qwen_label_text}': {coord_err}. Skipping.")
            continue

        width = x2 - x1
        height = y2 - y1

        coco_annotation = {
            "id": len(coco_annotations),
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "segmentation": [],
            "iscrowd": 0,
            "score": 1.0
        }
        coco_annotations.append(coco_annotation)

        detected_labels_for_vis.append({
            "bbox_orig": [x1, y1, x2, y2],
            "qwen_label": qwen_label_text,
            "category_id": category_id
        })

    return coco_annotations, detected_labels_for_vis

def parse_qwen_response(response, logger):
    """Parse the JSON response from Qwen model, handling markdown and potential issues"""
    if not response:
        return []

    json_text = ""
    try:
        if "```json" in response:
            json_text = response.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            match = re.search(r"```(?:.*\n)?([\s\S]+?)```", response)
            if match:
                json_text = match.group(1).strip()
            else:
                json_text = response.strip()
        else:
            json_text = response.strip()

        json_text = re.sub(r",\s*(\]|\})", r"\1", json_text)

        if not is_valid_json(json_text):
            logger.warning(f"Invalid JSON structure detected in response: {json_text[:200]}...")
            match = re.search(r'(\[.*\]|\{.*\})', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
                if not is_valid_json(json_text):
                    logger.warning(f"Could not extract valid JSON even after searching: {json_text[:200]}...")
                    return []
            else:
                 logger.warning(f"No JSON list/object found in the response: {json_text[:200]}...")
                 return []

        parsed_data = json.loads(json_text)

        if isinstance(parsed_data, list):
            boxes = parsed_data
        elif isinstance(parsed_data, dict):
             potential_keys = ['boxes', 'detections', 'objects', 'annotations']
             found = False
             for key in potential_keys:
                 if key in parsed_data and isinstance(parsed_data[key], list):
                     boxes = parsed_data[key]
                     found = True
                     break
             if not found:
                 logger.warning(f"Parsed JSON is a dict, but no expected list key found: {list(parsed_data.keys())}")
                 return []
        else:
            logger.warning(f"Parsed JSON is not a list or expected dict format: {type(parsed_data)}")
            return []

        return boxes

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        logger.debug(f"Problematic JSON text: {json_text}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {e}")
        logger.debug(f"Original Response: {response}")
        return []

def visualize_predictions(image_path, detected_labels_for_vis, save_path, logger):
    frame = cv2.imread(image_path)
    if frame is None:
        raise Exception("Failed to read image")

    if not detected_labels_for_vis:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        logger.info(f"No boxes to visualize for {os.path.basename(image_path)}, saved original.")
        return

    xyxy = []
    class_ids = []
    labels = []

    for item in detected_labels_for_vis:
        x1, y1, x2, y2 = item["bbox_orig"]
        xyxy.append([x1, y1, x2, y2])
        class_ids.append(item["category_id"])
        labels.append(item["qwen_label"])

    xyxy = np.array(xyxy, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    confidences = np.ones(len(xyxy), dtype=np.float32)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1
    )

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, annotated_frame)
    logger.debug(f"Saved visualization to {save_path}")

def visualize_predictions_og(image_path, boxes, input_width, input_height, 
                          original_width, original_height, categories_dict, save_path, logger):
    """Visualize prediction boxes on the image using supervision library"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise Exception("Failed to read image")
    
    xyxy = []
    class_ids = []
    confidences = []
    labels = []
    
    for box in boxes:
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("bbox_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label_text = box.get("label", "unknown")
        
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label_text.lower():
                category_id = cat_id
                break
        
        if category_id is None:
            category_id = -1
        
        x1, y1, x2, y2 = bbox_2d
        x1 = int(x1 / input_width * original_width)
        y1 = int(y1 / input_height * original_height)
        x2 = int(x2 / input_width * original_width)
        y2 = int(y2 / input_height * original_height)
        
        xyxy.append([x1, y1, x2, y2])
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

def process_image_single_class(args_tuple, logger):
    """
    Process a single image, running inference only for classes not yet marked
    as processed in the status map. Returns annotations and a list of
    successfully processed class IDs for this run.
    """
    (image_info, image_id, file_name, original_height, original_width,
     img_folder, prompted_name_to_id_map, id_to_prompted_name_map,
     results_dir, vis_dir,
     image_processing_status) = args_tuple

    image_path = os.path.join(img_folder, file_name)
    image_path = os.path.join(img_folder, file_name)
    if not os.path.exists(image_path):
        logger.error(f"Image file not found at expected path: {image_path}")
        return None, f"Image file not found: {file_name} in {img_folder}", image_id, []

    successfully_processed_classes = []
    all_parsed_boxes_for_image = []
    any_class_inference_failed = False
    classes_skipped = 0
    classes_attempted = 0

    try:
        input_height, input_width = smart_resize(original_height, original_width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        logger.info(f"Processing image: {file_name} (ID: {image_id}) - Checking class statuses.")

        num_classes_total = len(id_to_prompted_name_map)
        for class_idx, (cat_id, prompted_name) in enumerate(id_to_prompted_name_map.items()):
            if image_processing_status.get(cat_id, False):
                logger.debug(f"Image {image_id}, Class {cat_id} ('{prompted_name}') already processed. Skipping.")
                classes_skipped += 1
                continue

            classes_attempted += 1
            logger.debug(f"Image {image_id} ({file_name}): Querying class {class_idx+1}/{num_classes_total} '{prompted_name}' (ID: {cat_id})")
            single_class_prompt = f"Locate every {prompted_name} in the image and output the coordinates in JSON format."

            response, error = inference_with_retry(image_path, single_class_prompt, logger=logger, rate_limiter=rate_limiter, API_KEY=API_KEY)

            if error is None:
                successfully_processed_classes.append(cat_id)

                if response:
                    parsed_boxes_for_class = parse_qwen_response(response, logger)
                    if parsed_boxes_for_class:
                        original_label_count = len(parsed_boxes_for_class)
                        filtered_boxes_for_class = []
                        prompted_name_lower = prompted_name.lower()
                        for box in parsed_boxes_for_class:
                            qwen_label = box.get("label", "").strip().lower()
                            if qwen_label == prompted_name_lower:
                                filtered_boxes_for_class.append(box)

                        all_parsed_boxes_for_image.extend(filtered_boxes_for_class)

            else:
                logger.error(f"Failed inference for class '{prompted_name}' (ID: {cat_id}) on image {image_id}: {error}")
                any_class_inference_failed = True

        logger.info(f"Image {image_id}: Attempted {classes_attempted} classes, Skipped {classes_skipped}, Failed: {any_class_inference_failed}")

        vis_save_path = os.path.join(vis_dir, file_name)

        if all_parsed_boxes_for_image:
            logger.info(f"Image {image_id} ({file_name}): Aggregated {len(all_parsed_boxes_for_image)} relevant boxes from newly processed classes.")
            coco_annotations, detected_labels_for_vis = convert_to_coco_format(
                all_parsed_boxes_for_image,
                image_id, original_width, original_height,
                input_width, input_height,
                prompted_name_to_id_map, logger
            )

            if coco_annotations:
                visualize_predictions(image_path, detected_labels_for_vis, vis_save_path, logger)
                error_msg_out = "Some classes failed inference but results generated" if any_class_inference_failed else None
                return coco_annotations, error_msg_out, image_id, successfully_processed_classes
            else:
                logger.warning(f"Processed classes for {file_name}, got boxes, but no valid COCO annotations derived after conversion.")
                visualize_predictions(image_path, [], vis_save_path, logger)
                return [], f"No valid COCO annotations derived for {file_name} in this run", image_id, successfully_processed_classes
        else:
            final_error_msg = f"All attempted class inferences failed for {file_name}" if any_class_inference_failed and classes_attempted > 0 else f"No relevant objects found for attempted classes in {file_name}"
            logger.warning(final_error_msg)
            visualize_predictions(image_path, [], vis_save_path, logger)
            return None, final_error_msg, image_id, successfully_processed_classes

    except Exception as e:
        logger.error(f"Error processing image {file_name} (ID: {image_id}) during per-class loop or aggregation: {e}", exc_info=True)
        try:
            vis_save_path = os.path.join(vis_dir, file_name)
            visualize_predictions(image_path, [], vis_save_path, logger)
        except Exception as vis_e:
            logger.error(f"Additionally failed to save visualization for errored image {file_name}: {vis_e}")
        return None, f"Error processing {file_name}: {e}", image_id, []