import os
import json
import glob
from PIL import Image
import concurrent.futures
import threading
import logging
import random
from tqdm import tqdm
from io import BytesIO
from google import genai
import argparse
import random
import pickle
from google.genai.types import Content, Part
from utils.gemini_eval_utils import *
from utils.shared_eval_utils import *
import copy

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

def create_few_shot_prompt(train_folder, categories_dict):
    """
    Creates few-shot examples as a list of alternating user/model Content objects
    using all available annotated images in the training folder. Also prepares data
    to visualize the first 3 examples.
    
    Args:
        train_folder (str): Path to the training data folder.
        categories_dict (dict): Mapping from category ID to category name.

    Returns:
        tuple[list[Content], list[dict]]: 
            - A list of google.genai.types.Content objects for the few-shot sequence.
            - A list of dictionaries containing data for visualizing the first 3 examples.
              Returns empty lists if annotations or images are insufficient.
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

    few_shot_contents = []

    annotated_image_ids = list(annotations_by_image.keys())
    
    category_prompt = ", ".join(categories_dict.values())

    examples_added = 0
    for img_id in annotated_image_ids:
            
        img_info = images_by_id[img_id]
        width = img_info['width']
        height = img_info['height']
        image_path = os.path.join(train_folder, img_info['file_name'])

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        user_prompt_text_for_example = f"Detect the 2d bounding boxes of the following objects: {category_prompt}."
        text_part_for_example = Part(text=user_prompt_text_for_example)

        user_content = Content(role="user", parts=[text_part_for_example, image_part])

        example_annotations = annotations_by_image[img_id]
        
        model_response_text = format_ground_truth_for_model(example_annotations, categories_dict, width, height)
        model_part = Part(text=model_response_text)
        
        model_content = Content(role="model", parts=[model_part])

        few_shot_contents.append(user_content)
        few_shot_contents.append(model_content)
        examples_added += 1

    logger.info(f"Successfully created {examples_added} few-shot example turns (x2 Content objects) using all available train images.")
    return few_shot_contents

def precompute_prompts_for_dataset(dataset_dir, categories, categories_dict):
    """Precompute prompts, including multi-turn few-shot."""
    train_folder = os.path.join(dataset_dir, "train")
    
    readme_path = os.path.join(dataset_dir, "README.dataset.txt")
    instructions = ""
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            instructions = f.read().strip()
    
    category_prompt = ", ".join(categories)
    basic_system_prompt = "Return bounding boxes as a JSON array with labels. Never return masks or code fencing."
    final_query_text_basic = f"Detect the 2d bounding boxes of the following objects: {category_prompt}."

    instructions_system_prompt = basic_system_prompt
    final_query_text_instructions = f"Detect the 2d bounding boxes of the following objects: {category_prompt}.\n\nUse the following annotator instructions to improve detection accuracy:\n{instructions}\n"
    
    few_shot_contents_list = create_few_shot_prompt(train_folder, categories_dict) 
    few_shot_system_prompt = basic_system_prompt
    final_query_text_few_shot = final_query_text_basic

    combined_system_prompt = basic_system_prompt
    final_query_text_combined = final_query_text_basic

    prompts = {
        'basic': (basic_system_prompt, None, final_query_text_basic, None),
        'instructions': (instructions_system_prompt, None, final_query_text_instructions, instructions),
        'few_shot': (few_shot_system_prompt, few_shot_contents_list, final_query_text_few_shot, None),
        'combined': (combined_system_prompt, few_shot_contents_list, final_query_text_combined, instructions)
    }

    return prompts

def process_image(args):
    """Process a single image using multi-turn few-shot if applicable."""
    (image_info, image_id, file_name,
     test_folder, categories, categories_dict, results_dir, vis_dir,
     prompts, few_shot, just_instructions, combined,
     status_file_lock, model_id) = args 

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

    if status_dict.get(image_key) is True:
        logger.info(f"Skipping image {file_name} (ID: {image_id}) - Already marked True in status.pkl.")
        return None, None

    image_path = os.path.join(test_folder, file_name)
    
    with open(image_path, "rb") as f: img_bytes_content = f.read()
    im = Image.open(BytesIO(img_bytes_content))
    original_width, original_height = im.size
    input_width, input_height = im.size
    img_byte_arr_for_api = BytesIO()
    if im.mode == 'RGBA': im = im.convert('RGB')
    im.save(img_byte_arr_for_api, format='JPEG')
    img_byte_arr_for_api = img_byte_arr_for_api.getvalue()
    final_image_part = Part.from_bytes(data=img_byte_arr_for_api, mime_type="image/jpeg")

    current_attempt_status = False
    coco_annotations = []
    error_message_for_return = None
    boxes_all = []

    try:
        system_prompt = None
        contents = []

        if combined: mode = 'combined'
        elif few_shot: mode = 'few_shot'
        elif just_instructions: mode = 'instructions'
        else: mode = 'basic'

        system_prompt, few_shot_content_list_orig, final_query_text, instructions_text = prompts[mode]

        if few_shot_content_list_orig:
            contents = list(few_shot_content_list_orig)

            if mode == 'combined' and instructions_text:
                if contents and contents[0].role == "user":
                    original_first_content = contents[0]
                    new_first_parts = []
                    text_part_modified = False
                    for part in original_first_content.parts:
                        if hasattr(part, 'text') and isinstance(part.text, str) and not text_part_modified:
                            modified_text = f"{part.text}\n\nUse the following annotator instructions to improve detection accuracy:\n{instructions_text}\n"
                            new_first_parts.append(Part(text=modified_text))
                            text_part_modified = True
                        else:
                            new_first_parts.append(part)

                    contents[0] = Content(role="user", parts=new_first_parts)
                    logger.debug(f"Modified first user turn for image {image_id} using selective copying.")

        final_text_part = Part(text=final_query_text)
        final_user_turn_parts = [final_text_part, final_image_part]
        final_user_turn = Content(role="user", parts=final_user_turn_parts)

        contents.append(final_user_turn)

        logger.info(f"INITIATING API call for image {file_name} (ID: {image_id}) using {len(contents)} content turns.")

        response_text, err = inference_with_retry(
            contents=contents,
            system_prompt=system_prompt,
            model_id=model_id,
            logger=logger,
            rate_limiter=rate_limiter,
            client=client
        )

        if err:
            error_message = f"API Error processing image {file_name} (ID: {image_id}): {err}"
            logger.warning(error_message)
            current_attempt_status = False
            error_message_for_return = error_message
        elif response_text is None:
            error_message = f"API returned None response for image {file_name} (ID: {image_id})."
            logger.warning(error_message)
            current_attempt_status = False
            error_message_for_return = error_message
        else:
            current_attempt_status = True
            boxes_all = parse_json(response_text, logger)

            if not boxes_all:
                 logger.info(f"Image {image_id} ({file_name}): API call successful, but no valid boxes detected/parsed from response: {response_text[:100]}...")
            else:
                 coco_annotations = convert_coordinates_to_coco_format(
                     boxes_all, image_id, original_width, original_height,
                     input_width, input_height, categories_dict
                 )
                 if not coco_annotations:
                      logger.warning(f"Image {image_id} ({file_name}): Parsed {len(boxes_all)} boxes, but none converted to COCO format. Response: {response_text[:100]}...")
                      print(f"Image {image_id} ({file_name}): Parsed {len(boxes_all)} boxes, but none converted to COCO format. Response: {response_text[:100]}...")

    except Exception as e:
        error_message = f"Unexpected error processing {file_name} (ID: {image_id}): {e}"
        logger.error(error_message, exc_info=True)
        current_attempt_status = False
        error_message_for_return = error_message

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

    with status_file_lock:
        final_status_dict = {}
        try:
             if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
                  with open(status_file, "rb") as sf_read:
                      final_status_dict = pickle.load(sf_read)
                  if not isinstance(final_status_dict, dict): final_status_dict = {}
        except Exception as read_err:
             logger.warning(f"Error reading status file before final write for {image_id}: {read_err}. Using local status dict.")
             final_status_dict = status_dict
        
        final_status_dict[image_key] = current_attempt_status
        
        try:
            with open(status_file, "wb") as sf_write:
                pickle.dump(final_status_dict, sf_write)
        except Exception as write_e:
             logger.error(f"Failed to write final status for {image_id} to {status_file}: {write_e}")
             if not error_message_for_return:
                 error_message_for_return = f"Failed to write final status to {status_file}: {write_e}"

    if current_attempt_status:
        return coco_annotations, None
    else:
        return None, error_message_for_return

def process_dataset(dataset_dir, few_shot, just_instructions, combined, model_id, output_dir_root, visualize_dir_root):
    """Process a single dataset with one API call per image, using status file."""
    dataset_name = os.path.basename(dataset_dir)

    results_dir = os.path.join(output_dir_root, dataset_name)
    vis_dir = os.path.join(visualize_dir_root, dataset_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    test_folder = os.path.join(dataset_dir, "test")
    train_folder = os.path.join(dataset_dir, "train")

    annotation_file = glob.glob(os.path.join(test_folder, "*_annotations.coco.json"))
    annotation_file = annotation_file[0]

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    categories = annotations.get("categories", [])
    category_names = [cat["name"] for cat in categories]
    categories_dict = {cat["id"]: cat["name"] for cat in categories}

    visualize_few_shot_ground_truth(
        train_folder=train_folder,
        categories_dict=categories_dict,
        output_vis_dir=vis_dir,
        logger=logger
    )

    prompts = precompute_prompts_for_dataset(dataset_dir, category_names, categories_dict)

    images = annotations.get("images", [])

    results_file = os.path.join(results_dir, f"gemini_detection_results.json")
    existing_results = load_existing_results(results_file, logger)
    
    processed_count = 0
    error_count = 0
    skipped_count = 0

    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Found {len(images)} images in annotations.")
    logger.info(f"Results will be stored in {results_dir}")
    logger.info(f"Status file: {os.path.join(results_dir, 'status.pkl')}")

    status_file_lock = threading.Lock()

    args_list = [
        (image_info, image_info["id"], image_info["file_name"],
         test_folder,
         category_names, categories_dict, results_dir, vis_dir,
         prompts, few_shot, just_instructions, combined,
         status_file_lock, model_id)
        for image_info in images
    ]
    
    results_map = {}
    for ann in existing_results:
        img_id = ann['image_id']
        if img_id not in results_map:
            results_map[img_id] = []
        results_map[img_id].append(ann)
    
    total_images_to_process = len(args_list) 

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, args): args[1] for args in args_list} 

        with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="image", leave=False) as pbar:
            for future in concurrent.futures.as_completed(futures):
                image_id = futures[future]
                processed_count += 1
                try:
                    result, error = future.result() 
                    
                    if error:
                        error_count += 1
                        pbar.write(f"Image ID {image_id}: Error - {error}") 
                        results_map.setdefault(image_id, []) 
                    elif result is None and error is None:
                        skipped_count += 1
                        results_map.setdefault(image_id, []) 
                        logger.debug(f"Image ID {image_id}: Confirmed skip based on status.")
                    elif result is not None and error is None: 
                        results_map[image_id] = result 
                        logger.debug(f"Updated results for image {image_id} with {len(result)} annotations (Status: True).")

                except Exception as exc:
                    error_count += 1 
                    logger.error(f"Image ID {image_id} generated an exception during future processing: {exc}", exc_info=True)
                    pbar.write(f"Image ID {image_id}: Worker exception - {exc}") 
                    results_map.setdefault(image_id, []) 
                
                pbar.update(1)

    final_results_list = []
    for img_id, annotations_list in results_map.items():
        if annotations_list:
            unique_anns = []
            seen_ann_ids = set()
            for ann in annotations_list:
                 ann_tuple = (ann.get('category_id'), tuple(ann.get('bbox', [])))
                 if ann_tuple not in seen_ann_ids:
                      unique_anns.append(ann)
                      seen_ann_ids.add(ann_tuple)
            final_results_list.extend(unique_anns)

    try: 
        with open(results_file, 'w') as f:
            json.dump(final_results_list, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to write final results file {results_file}: {e}")
        raise Exception(f"Failed to write final results file {results_file}: {e}")

    logger.info(f"Saved results to {results_file}")
    logger.info(f"Final results for {dataset_name}: {len(final_results_list)} unique annotations")
    logger.info(f"Total images checked/attempted for {dataset_name}: {processed_count}/{total_images_to_process}")
    logger.info(f"Images skipped due to existing 'True' status: {skipped_count}")
    logger.info(f"Image-level errors/failures reported: {error_count}")
    
    return results_file, final_results_list, processed_count, error_count, skipped_count

def main():
    """Main function to process all datasets"""
    parser = argparse.ArgumentParser(description='Process datasets for object detection')
    parser.add_argument('--dataset_location', type=str, default="rf100-vl",
                        help='Location of the dataset to process')
    parser.add_argument('--just_instructions', action='store_true',
                        help='Use custom instructions instead of default prompt')
    parser.add_argument('--few_shot', action='store_true',
                        help='Use few-shot examples in prompt')
    parser.add_argument('--combined', action='store_true',
                        help='Use both instructions and few-shot examples')
    parser.add_argument('--model_id', type=str, default= "gemini-2.5-pro-preview-03-25",
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

    log_file_name_base = f"gemini_detection_{args.model_id}_{eval_mode_str}"

    if args.save_dir:
        output_dir_root = os.path.join(args.save_dir, "results")
        visualize_dir_root = os.path.join(args.save_dir, "visualizations")
        log_file = os.path.join(args.save_dir, f"{log_file_name_base}.log")
        print(f"Using custom save directory: {args.save_dir}")
    else:
        output_dir_root = f"results_{args.model_id}_{eval_mode_str}"
        visualize_dir_root = f"visualizations_{args.model_id}_{eval_mode_str}"
        log_file = f"{log_file_name_base}.log"
        print(f"Using default save directories: {output_dir_root}, {visualize_dir_root}")

    os.makedirs(output_dir_root, exist_ok=True)
    os.makedirs(visualize_dir_root, exist_ok=True)
    if args.save_dir:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    all_dataset_dirs = sorted([d for d in glob.glob(args.dataset_location + "/*")
                               if os.path.isdir(d) and os.path.exists(os.path.join(d, "test"))])

    logger.info(f"Found {len(all_dataset_dirs)} total datasets. Will process all.")

    dataset_stats = {}

    for dataset_dir in tqdm(all_dataset_dirs, desc="Processing datasets", unit="dataset"): 
        dataset_name = os.path.basename(dataset_dir)
        logger.info(f"Starting processing dataset: {dataset_name}")
        _, _, processed, errored, skipped = process_dataset(
            dataset_dir, args.few_shot, args.just_instructions,
            args.combined, args.model_id,
            output_dir_root, visualize_dir_root
        )
        dataset_stats[dataset_name] = {"processed": processed, "errors": errored, "skipped": skipped}
        logger.info(f"Finished processing dataset: {dataset_name}")

    logger.info("="*30 + " Processing Summary " + "="*30)
    total_processed_all = 0
    total_errors_all = 0
    total_skipped_all = 0
    successful_datasets = 0
    failed_datasets = 0

    for name, stats in dataset_stats.items():
        if "status" in stats and "Skipped" in stats["status"]:
             logger.warning(f"Dataset '{name}': {stats['status']}")
             failed_datasets += 1
             continue
        if "status" in stats and "Failed" in stats["status"]:
             logger.error(f"Dataset '{name}': {stats['status']}")
             failed_datasets += 1
             continue
        
        processed = stats.get('processed', 0)
        errors = stats.get('errors', 0)
        skipped = stats.get('skipped', 0)
        
        total_processed_all += processed
        total_errors_all += errors
        total_skipped_all += skipped
        successful_datasets += 1

        logger.info(f"Dataset '{name}': Checked/Attempted {processed} images. Skipped: {skipped}. Errors: {errors}.")
    
    logger.info("-"*78)
    logger.info(f"Overall across {successful_datasets} successfully processed datasets (out of {len(dataset_stats)} total):")
    logger.info(f"Total image checks/attempts: {total_processed_all}")
    logger.info(f"Total images skipped (status True): {total_skipped_all}")
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