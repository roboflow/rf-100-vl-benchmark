import os
import json
import glob
from PIL import Image
import concurrent.futures
import threading
import logging
import random
import re
from tqdm import tqdm
from io import BytesIO
from google import genai
import argparse
import random
import pickle
from google.genai.types import Content, Part
import argparse
import re
import copy
from utils.gemini_eval_utils import *
from utils.shared_eval_utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv('GOOGLE_API_KEY')

client = genai.Client(api_key=API_KEY)

rate_limiter = RateLimiter(REQUEST_LIMIT)

def create_few_shot_prompt_for_class(train_folder, categories_dict, target_class_name):
    """
    Creates class-specific few-shot examples as a list of alternating user/model Content objects.
    Only includes annotations for the target_class_name in the examples.

    Args:
        train_folder (str): Path to the training data folder.
        categories_dict (dict): Mapping from category ID to category name.
        target_class_name (str): The specific class name to filter examples for.

    Returns:
        list[Content]: A list of google.genai.types.Content objects containing
                       multi-turn examples relevant ONLY to the target_class_name.
                       Returns empty list if annotations/images are insufficient
                       or no examples contain the target class.
    """
    annotation_file = os.path.join(train_folder, "_annotations.coco.json")
    if not os.path.exists(annotation_file):
        logger.warning(f"Few-shot annotation file not found: {annotation_file} for class {target_class_name}")
        return []

    with open(annotation_file, 'r') as f:
        train_data = json.load(f)

    images_by_id = {img['id']: img for img in train_data.get('images', [])}
    all_annotations = train_data.get('annotations', [])
    target_category_id = None
    for cat_id, cat_name in categories_dict.items():
        if cat_name == target_class_name:
            target_category_id = cat_id
            break

    if target_category_id is None:
        logger.warning(f"Target class '{target_class_name}' not found in categories_dict.")
        return []

    annotations_by_image = {}
    for ann in all_annotations:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    few_shot_contents = []

    images_with_target_class = set()
    for ann in all_annotations:
        if ann['category_id'] == target_category_id:
            images_with_target_class.add(ann['image_id'])

    if not images_with_target_class:
        raise Exception(f"No training images found containing the target class '{target_class_name}'. Cannot create few-shot examples.")

    image_ids_to_process = list(images_with_target_class)

    examples_added = 0
    for img_id in image_ids_to_process:
        if img_id not in images_by_id:
            raise Exception(f"Image ID {img_id} expected but not found in image list for class {target_class_name}. Skipping.")

        if img_id not in annotations_by_image:
            raise Exception(f"Image ID {img_id} has no annotations entry for class {target_class_name}. Skipping.")

        img_info = images_by_id[img_id]
        width = img_info['width']
        height = img_info['height']
        image_path = os.path.join(train_folder, img_info['file_name'])

        if not os.path.exists(image_path):
            raise Exception(f"Few-shot image file not found: {image_path} for class {target_class_name}. Skipping example.")

        filtered_annotations = [
            ann for ann in annotations_by_image[img_id]
            if ann['category_id'] == target_category_id
        ]

        if not filtered_annotations:
            raise Exception(f"Image {img_info['file_name']} (ID: {img_id}) was expected to have '{target_class_name}' but filtering yielded no annotations. Skipping example.")

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        user_prompt_text = f"Detect all 2d bounding boxes of {target_class_name}."
        current_text_part = Part(text=user_prompt_text)

        user_content = Content(role="user", parts=[current_text_part, image_part])

        model_response_text = format_ground_truth_for_model(filtered_annotations, categories_dict, width, height)
        model_part = Part(text=model_response_text)

        model_content = Content(role="model", parts=[model_part])

        few_shot_contents.append(user_content)
        few_shot_contents.append(model_content)
        examples_added += 1
        # Optional: Limit the number of examples per class
        # if examples_added >= MAX_FEW_SHOT_EXAMPLES_PER_CLASS:
        #    break

    logger.info(f"Successfully created {examples_added} few-shot example turns for class '{target_class_name}'.")
    return few_shot_contents

def precompute_prompts_for_dataset(dataset_dir, categories, categories_dict):
    """
    Precompute prompts, generating CLASS-SPECIFIC multi-turn few-shot examples.
    Does NOT use the main README for instructions; assumes class-specific
    instructions are loaded later in process_image.

    Args:
        dataset_dir (str): Path to the dataset directory.
        categories (list[str]): List of category names.
        categories_dict (dict): Mapping from category ID to category name.

    Returns:
        dict: A nested dictionary storing prompts:
              prompts[mode][class_name] = (system_prompt, class_specific_few_shot_list)
              'class_specific_few_shot_list' is None for basic/instructions modes.
    """
    dataset_name = os.path.basename(dataset_dir)
    train_folder = os.path.join(dataset_dir, "train")
    logger.info(f"Precomputing class-specific prompts for dataset: {dataset_name}")

    base_system_prompt = "Return bounding boxes as a JSON array with labels. Never return masks or code fencing."

    prompts = {
        'basic': {},
        'instructions': {},
        'few_shot': {},
        'combined': {}
    }

    for class_name in categories:
        logger.debug(f"Generating few-shot examples for class: {class_name}")
        class_few_shot_list = create_few_shot_prompt_for_class(
            train_folder, categories_dict, class_name
        )

        if class_few_shot_list:
            for content_item in class_few_shot_list:
                if content_item.role == "model":
                    model_text = None
                    for part in content_item.parts:
                        if hasattr(part, 'text') and part.text:
                            model_text = part.text
                            break
                    
                    assert model_text is not None, \
                        f"[Assertion Error] Model turn for class '{class_name}' few-shot example has no text part."

                    parsed_boxes = []
                    try:
                        json_text = model_text
                        if "```json" in model_text:
                            json_text = model_text.split("```json", 1)[1].split("```", 1)[0].strip()
                        elif "```" in model_text:
                            match = re.search(r"```(?:\w*\s*)?([\s\S]+?)```", model_text)
                            if match:
                                json_text = match.group(1).strip()

                        parsed_boxes = json.loads(json_text)
                        assert isinstance(parsed_boxes, list), \
                            f"[Assertion Error] Parsed JSON in model turn for '{class_name}' is not a list."

                    except json.JSONDecodeError as e:
                        assert False, \
                            (f"[Assertion Error] Failed to parse JSON in model turn for class '{class_name}'. Error: {e}. "
                             f"Content: {model_text[:500]}...")
                    except Exception as e:
                         assert False, f"[Assertion Error] Unexpected error during JSON parsing/checking for '{class_name}': {e}"

                    for box in parsed_boxes:
                        assert isinstance(box, dict), \
                            f"[Assertion Error] Item in parsed box list for '{class_name}' is not a dict: {box}"
                        assert "label" in box, \
                            f"[Assertion Error] Box in model turn for '{class_name}' is missing 'label': {box}"
                        
                        assert box["label"] == class_name, \
                            f"[Assertion Error] Class Exclusivity Failed for '{class_name}': " \
                            f"Found label '{box['label']}' in few-shot example ground truth. " \
                            f"Expected only '{class_name}'."

        prompts['basic'][class_name] = (base_system_prompt, None)
        prompts['instructions'][class_name] = (base_system_prompt, None)
        prompts['few_shot'][class_name] = (base_system_prompt, class_few_shot_list)
        prompts['combined'][class_name] = (base_system_prompt, class_few_shot_list)

    return prompts

def process_image(args):
    """
    Process a single image by making one API call PER CLASS.
    Uses CLASS-SPECIFIC multi-turn few-shot prompts and instructions if enabled.
    Instructions are prepended to the first user turn.
    Maintains class-level status tracking in status.pkl.
    """

    (image_info, image_id, file_name,
     test_folder, categories, categories_dict, results_dir, vis_dir,
     prompts, few_shot, just_instructions, combined,
     status_file_lock, model_id, full_dataset_dir) = args

    dataset_name = os.path.basename(results_dir)
    dataset_dir = full_dataset_dir

    status_file = os.path.join(results_dir, "status.pkl")
    status_dict = {}
    image_key = str(image_id)

    with status_file_lock:
        try:
            if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
                with open(status_file, "rb") as sf:
                    status_dict = pickle.load(sf)
            if not isinstance(status_dict, dict):
                 logger.warning(f"Status file {status_file} contained non-dict data. Resetting.")
                 status_dict = {}
        except (EOFError, pickle.UnpicklingError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Error reading status file {status_file} ({e}). Treating as empty.")
            status_dict = {}
    
    image_status = status_dict.get(image_key, {})
    if not isinstance(image_status, dict): image_status = {}

    image_path = os.path.join(test_folder, file_name)
    if not os.path.exists(image_path):
        raise Exception(f"Image file not found: {file_name} (ID: {image_id})")
    
    with open(image_path, "rb") as f: img_bytes_content = f.read()
    im = Image.open(BytesIO(img_bytes_content))
    original_width, original_height = im.size
    input_width, input_height = im.size
    img_byte_arr_for_api = BytesIO()
    if im.mode == 'RGBA': im = im.convert('RGB')
    im.save(img_byte_arr_for_api, format='JPEG')
    img_byte_arr_for_api = img_byte_arr_for_api.getvalue()
    final_image_part = Part.from_bytes(data=img_byte_arr_for_api, mime_type="image/jpeg")

    if combined: mode = 'combined'
    elif few_shot: mode = 'few_shot'
    elif just_instructions: mode = 'instructions'
    else: mode = 'basic'

    boxes_all = []
    class_errors = []
    any_class_processed_this_run = False
    any_class_succeeded_previously = any(status is True for status in image_status.values())

    for cls in categories:
        if image_status.get(cls, False) is True:
            logger.debug(f"Skipping API call for class '{cls}' in image {file_name} (ID: {image_id}) - Already marked True.")
            continue

        any_class_processed_this_run = True
        logger.info(f"INITIATING API call for class '{cls}' in image {file_name} (ID: {image_id}). Mode: {mode}")

        system_prompt, class_few_shot_list = prompts[mode][cls]

        class_instructions = ""
        if mode in ['instructions', 'combined']:
            instruction_file_path = os.path.join(dataset_dir, "class_instructions", f"{cls}.txt")
            with open(instruction_file_path, 'r') as f_instr:
                loaded_instr = f_instr.read().strip()
                instruction_parts = loaded_instr.split("\n\n\n")
                if instruction_parts:
                    class_instructions = instruction_parts[-1].strip() # We found that dataset description is not helpful, so we only use class description
                else:
                    raise ValueError(f"No instructions found for class '{cls}' in {instruction_file_path}")

        contents_for_cls = []

        if class_few_shot_list:
            copied_few_shot_list = copy.deepcopy(class_few_shot_list)
            contents_for_cls.extend(copied_few_shot_list)

        final_detection_command = f"Detect all 2d bounding boxes of {cls}."
        final_text_part = Part(text=final_detection_command)
        final_image_part = Part.from_bytes(data=img_bytes_content, mime_type="image/jpeg")
        final_user_turn_parts = [final_text_part, final_image_part]
        final_user_turn = Content(role="user", parts=final_user_turn_parts)

        contents_for_cls.append(final_user_turn)

        if class_instructions and contents_for_cls:
            if contents_for_cls:
                 first_content = contents_for_cls[0]
                 if first_content.role == "user":
                     text_part_modified = False
                     for i, part in enumerate(first_content.parts):
                         if hasattr(part, 'text') and isinstance(part.text, str):
                              original_text = part.text
                              part.text = f"{original_text}\n\nUse the following annotator instructions to improve detection accuracy:\n{class_instructions}\n"
                              text_part_modified = True
                              logger.debug(f"Appended instructions to existing text part {i} in the first user turn for class '{cls}'.")
                              break
                     if not text_part_modified:
                         raise ValueError(f"Failed to prepend instructions to the first user turn for class '{cls}'.")
                 else:
                      raise ValueError(f"Unexpected content role in first user turn for class '{cls}'.")
        elif class_instructions and not contents_for_cls:
             raise ValueError(f"Class instructions loaded for '{cls}', but contents_for_cls is empty. Cannot prepend instructions.")

        response_text, err = inference_with_retry(
            contents=contents_for_cls,
            system_prompt=system_prompt,
            model_id=model_id,
            logger=logger,
            rate_limiter=rate_limiter,
            client=client
        )

        if err:
            logger.warning(f"API Error detecting class '{cls}' in {file_name}: {err}")
            image_status[cls] = False
            class_errors.append(f"Class '{cls}': {err[:200]}...")
        elif response_text is None:
             error_message = f"API returned None response for class '{cls}' in image {file_name}."
             logger.warning(error_message)
             image_status[cls] = False
             class_errors.append(f"Class '{cls}': None response")
        else:
            image_status[cls] = True
            parsed_boxes = parse_json(response_text, logger)
            if not parsed_boxes:
                 logger.info(f"Image {image_id}, Class '{cls}': OK, but no boxes detected/parsed.")
            else:
                 for box in parsed_boxes:
                     if isinstance(box, dict):
                          label_keys = ["label", "class", "name", "class_label", "class_name"]
                          existing_label = next((box[k] for k in label_keys if k in box), None)
                          if not existing_label:
                               box["label"] = cls
                          elif cls.lower() not in str(existing_label).lower():
                               logger.warning(f"Image {image_id}, Class '{cls}': Model possibly returned box with mismatched label '{existing_label}'. Keeping model label.")
                 boxes_all.extend(parsed_boxes)

    if any_class_processed_this_run:
        with status_file_lock:
            current_status_dict = {}
            try:
                 if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
                     with open(status_file, "rb") as sf_read:
                         re_read_status = pickle.load(sf_read)
                         if isinstance(re_read_status, dict): current_status_dict = re_read_status
                         else: current_status_dict = status_dict
            except Exception as read_err:
                 logger.warning(f"Error re-reading status file {status_file} before write ({read_err}). Using local updates.")
                 current_status_dict = status_dict

            current_status_dict[image_key] = image_status

            try:
                with open(status_file, "wb") as sf_write:
                    pickle.dump(current_status_dict, sf_write)
            except Exception as write_e:
                 error_message = f"CRITICAL: Failed to write status update for {image_id} to {status_file}: {write_e}"
                 logger.error(error_message)
                 return None, error_message

    final_success_status = any(status is True for status in image_status.values())

    if not final_success_status:
        error_message = f"All class detections failed or were skipped for {file_name}. Errors this run: {'; '.join(class_errors)}"
        logger.warning(error_message)
        return None, error_message

    coco_annotations = []
    if boxes_all:
        try:
            coco_annotations = convert_coordinates_to_coco_format(
                boxes_all, image_id, original_width, original_height,
                input_width, input_height, categories_dict
            )
            if not coco_annotations and boxes_all:
                 logger.warning(f"Image {image_id}: Had {len(boxes_all)} raw boxes, but none converted to COCO.")
        except Exception as convert_err:
            logger.error(f"Error converting boxes to COCO for {image_id}: {convert_err}", exc_info=True)
            return None, f"Error converting boxes for {image_id}: {convert_err}"
    elif any_class_succeeded_previously and not any_class_processed_this_run:
         logger.info(f"Image {image_id}: All classes previously completed. No new boxes.")
         coco_annotations = []
    else:
         logger.info(f"Image {image_id}: Class processing completed, but no valid boxes detected.")
         coco_annotations = []

    try:
        vis_save_path = os.path.join(vis_dir, file_name)
        visualize_predictions(
            image_path, boxes_all if boxes_all else [],
            input_width, input_height,
            original_width, original_height, categories_dict, vis_save_path
        )
    except Exception as e:
        logger.error(f"Failed to visualize predictions for {file_name} (ID: {image_id}): {e}")
        print(f"Failed to visualize predictions for {file_name} (ID: {image_id}): {e}")

    return coco_annotations, None

def process_dataset(dataset_dir, few_shot, just_instructions, combined, model_id, output_dir_root, visualize_dir_root, full_dataset_dir):
    """
    Process a single dataset using one API call PER CLASS per image.
    Uses class-level status checking via process_image.
    Adopts multiclass features like GT visualization and result consolidation.
    MODIFIED: Loads existing results and merges new results intelligently.
    """
    dataset_name = os.path.basename(dataset_dir)
    results_dir = os.path.join(output_dir_root, dataset_name)
    vis_dir = os.path.join(visualize_dir_root, dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    test_folder = os.path.join(dataset_dir, "test")
    train_folder = os.path.join(dataset_dir, "train")

    if not os.path.exists(test_folder):
        logger.error(f"Test folder not found in {dataset_dir}")
        raise FileNotFoundError(f"Test folder not found in {dataset_dir}")
    if (few_shot or combined) and not os.path.exists(train_folder):
         raise FileNotFoundError(f"Train folder needed for few-shot/combined prompts not found in {dataset_dir}")

    annotation_file = glob.glob(os.path.join(test_folder, "*_annotations.coco.json"))
    if not annotation_file:
        logger.error(f"Annotation file not found in {test_folder}")
        raise FileNotFoundError(f"Annotation file not found in {test_folder}")
    annotation_file = annotation_file[0]

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    categories = annotations.get("categories", [])
    category_names = [cat["name"] for cat in categories]
    categories_dict = {cat["id"]: cat["name"] for cat in categories}

    if os.path.exists(train_folder):
        visualize_few_shot_ground_truth(
            train_folder=train_folder,
            categories_dict=categories_dict,
            output_vis_dir=vis_dir,
            logger=logger
        )
    else:
        raise Exception(f"Train folder not found in {dataset_dir}")

    prompts = precompute_prompts_for_dataset(dataset_dir, category_names, categories_dict) # Assumes train folder exists if needed

    images = annotations.get("images", [])
    if not images:
        logger.warning(f"No test images found in annotation file for dataset {dataset_name}. Skipping test processing.")
        return None, [], 0, 0

    results_file = os.path.join(results_dir, f"gemini_detection_results.json")

    logger.info(f"Loading existing results from {results_file} if available.")
    existing_results_list = load_existing_results(results_file, logger)
    all_results_map = {}
    image_ids_in_dataset = {img['id'] for img in images}
    for img_id in image_ids_in_dataset:
         all_results_map[img_id] = []
    for ann in existing_results_list:
         img_id = ann.get('image_id')
         if img_id in all_results_map:
            all_results_map[img_id].append(ann)
         else:
            raise Exception(f"Image ID {img_id} not found in dataset {dataset_name}")
    logger.info(f"Loaded {len(existing_results_list)} annotations from file, mapped to {len(all_results_map)} images.")

    processed_count = 0
    error_count = 0
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Found {len(images)} images in annotations.")
    logger.info(f"Results will be stored in {results_dir}")
    logger.info(f"Class-level status file: {os.path.join(results_dir, 'status.pkl')}")

    status_file_lock = threading.Lock()

    args_list = [
        (image_info, image_info["id"], image_info["file_name"],
         test_folder,
         category_names,
         categories_dict, results_dir, vis_dir,
         prompts, few_shot, just_instructions, combined,
         status_file_lock, model_id, full_dataset_dir)
        for image_info in images
    ]

    temp_run_results = {}

    total_images_to_process = len(args_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, args): args[1] for args in args_list}
        with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="image", leave=False) as pbar:
            for future in concurrent.futures.as_completed(futures):
                image_id = futures[future]
                processed_count += 1
                try:
                    result, error = future.result()
                    temp_run_results[image_id] = (result, error)
                    if error:
                        error_count += 1
                        logger.warning(f"Image ID {image_id}: Error during processing - {error}")

                except Exception as exc:
                    error_count += 1
                    error_msg = f"Image ID {image_id}: Exception during future execution: {exc}"
                    logger.error(error_msg, exc_info=True)
                    temp_run_results[image_id] = (None, error_msg)

                pbar.update(1)

    logger.info("Merging results from current run with existing results...")
    for image_id, (run_result, run_error) in temp_run_results.items():
        if run_error is None and run_result is not None:
            processed_class_ids_in_result = {ann['category_id'] for ann in run_result}

            existing_anns = all_results_map.get(image_id, [])

            preserved_anns = [
                ann for ann in existing_anns
                if ann.get('category_id') not in processed_class_ids_in_result
            ]

            merged_anns = preserved_anns + run_result

            all_results_map[image_id] = merged_anns
            logger.debug(f"Image ID {image_id}: Merged {len(run_result)} new annotations with {len(preserved_anns)} preserved annotations.")

        elif run_error is not None:
            logger.warning(f"Image ID {image_id}: Skipping results update due to error in current run: {run_error}")

    final_results_list = []
    seen_annotation_keys = set()
    for img_id in sorted(all_results_map.keys()):
        annotations_list = all_results_map[img_id]
        if annotations_list:
            for ann in annotations_list:
                 bbox_tuple = tuple(ann.get('bbox', []))
                 ann_key = (img_id, ann.get('category_id'), bbox_tuple)

                 if ann_key not in seen_annotation_keys:
                      final_results_list.append(ann)
                      seen_annotation_keys.add(ann_key)
                 else:
                     logger.debug(f"Skipping duplicate annotation after merge: {ann_key}")

    with open(results_file, 'w') as f:
        json.dump(final_results_list, f, indent=2)

    logger.info(f"Saved updated results ({len(final_results_list)} unique annotations) to {results_file}")
    logger.info(f"Total images checked/attempted for {dataset_name}: {processed_count}/{total_images_to_process}")
    logger.info(f"Image-level errors/failures reported in this run: {error_count}")

    return results_file, final_results_list, processed_count, error_count


def main():
    """Main function to process all datasets"""
    parser = argparse.ArgumentParser(description='Process datasets for object detection (Class-by-Class)')
    parser.add_argument('--dataset_location', type=str, default="rf100_subtest",
                        help='Location of the dataset to process')
    parser.add_argument('--just_instructions', action='store_true',
                        help='Use custom instructions instead of default prompt')
    parser.add_argument('--few_shot', action='store_true',
                        help='Use few-shot examples in prompt (multi-turn)')
    parser.add_argument('--combined', action='store_true',
                        help='Use both instructions and few-shot examples (multi-turn)')
    parser.add_argument('--model_id', type=str, default="gemini-2.5-pro-preview-03-25",
                        help='The Gemini model ID to use for inference')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional custom root directory to save results and visualizations')
    args = parser.parse_args()

    if args.combined:
        eval_mode_str = "combined"
    elif args.few_shot:
        eval_mode_str = "few_shot"
    elif args.just_instructions:
        eval_mode_str = "instructions"
    else:
        eval_mode_str = "basic"

    log_file_name_base = f"gemini_detection_{args.model_id}_{eval_mode_str}_per_class"

    if args.save_dir:
        output_dir_root = os.path.join(args.save_dir, f"results_{eval_mode_str}_per_class")
        visualize_dir_root = os.path.join(args.save_dir, f"visualizations_{eval_mode_str}_per_class")
        log_file = os.path.join(args.save_dir, f"{log_file_name_base}.log")
        print(f"Using custom save directory: {args.save_dir}")
    else:
        output_dir_root = f"results_{args.model_id}_{eval_mode_str}_per_class"
        visualize_dir_root = f"visualizations_{args.model_id}_{eval_mode_str}_per_class"
        log_file = f"{log_file_name_base}.log"
        print(f"Using default save directories: {output_dir_root}, {visualize_dir_root}")

    os.makedirs(output_dir_root, exist_ok=True)
    os.makedirs(visualize_dir_root, exist_ok=True)
    if args.save_dir:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    all_dataset_dirs = sorted([d for d in glob.glob(args.dataset_location + "/*")
                               if os.path.isdir(d) and os.path.exists(os.path.join(d, "test"))])

    if not all_dataset_dirs:
        logger.error("No datasets found. Check the path pattern used in glob.glob.")
        return

    logger.info(f"Found {len(all_dataset_dirs)} total datasets. Will process all.")

    dataset_stats = {}

    for dataset_dir in tqdm(all_dataset_dirs, desc="Processing datasets", unit="dataset"):
        dataset_name = os.path.basename(dataset_dir)
        full_dataset_dir = os.path.join(args.dataset_location, dataset_name)

        logger.info(f"Starting processing dataset: {dataset_name}")
        _, _, processed, errored = process_dataset(
            dataset_dir, args.few_shot, args.just_instructions,
            args.combined, args.model_id,
            output_dir_root, visualize_dir_root, full_dataset_dir
        )
        dataset_stats[dataset_name] = {"processed": processed, "errors": errored}
        logger.info(f"Finished processing dataset: {dataset_name}")

    logger.info("="*30 + " Processing Summary " + "="*30)
    total_processed_all = 0
    total_errors_all = 0
    successful_datasets = 0
    failed_datasets = 0

    for name, stats in dataset_stats.items():
        if "status" in stats and ("Skipped" in stats["status"] or "Failed" in stats["status"]):
             logger.warning(f"Dataset '{name}': {stats['status']}")
             failed_datasets += 1
             continue

        processed = stats.get('processed', 0)
        errors = stats.get('errors', 0)

        total_processed_all += processed
        total_errors_all += errors
        successful_datasets += 1

        logger.info(f"Dataset '{name}': Checked/Attempted {processed} images. Image-level errors: {errors}.")

    logger.info("-"*78)
    logger.info(f"Overall across {successful_datasets} successfully processed datasets (out of {len(dataset_stats)} total):")
    logger.info(f"Total image checks/attempts: {total_processed_all}")
    logger.info(f"Total image-level errors/failures: {total_errors_all}")
    if failed_datasets > 0:
        logger.warning(f"{failed_datasets} datasets encountered critical errors or were skipped.")
    logger.info("="*78)
    if failed_datasets == 0:
        logger.info("All datasets processed.")
    else:
        logger.warning("Processing complete, but some datasets encountered errors.")

if __name__ == "__main__":
    main()