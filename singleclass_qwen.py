import os
import json
import glob
from PIL import Image
import time
from qwen_vl_utils import smart_resize
from tqdm import tqdm
import concurrent.futures
import logging
import random
import argparse
from utils.qwen_eval_utils import *
from utils.shared_eval_utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rf100_qwen_single_class_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv('DASHSCOPE_API_KEY')

BASE_RESULTS_DIR = "rf100_qwen_single_class_detection_results_test"
BASE_VISUALIZE_DIR = "rf100_qwen_single_class_visualized_predictions_test"

STATUS_FILENAME_PATTERN = "processing_status_{model}_{suffix}.pkl"

rate_limiter = RateLimiter(REQUEST_LIMIT)

def process_image(args_tuple):
    """
    Process a single image for RF100, running inference for each class not yet marked
    as processed in the status map. Returns annotations and a list of
    successfully processed class IDs for this run. (Adapted from OdinW)
    """
    (image_info, image_id, file_name, original_height, original_width,
     test_folder,
     prompted_name_to_id_map, id_to_prompted_name_map,
     results_dir, vis_dir,
     image_processing_status,
     base_instructions
     ) = args_tuple

    image_path = os.path.join(test_folder, file_name)
    if not os.path.exists(image_path):
        image_files = glob.glob(os.path.join(test_folder, "**", file_name), recursive=True)
        if image_files:
            image_path = image_files[0]
        else:
            raise Exception(f"Image file not found: {file_name} in {test_folder} or subdirectories.")

    successfully_processed_classes = []
    all_parsed_boxes_for_image = []
    any_class_inference_failed = False
    classes_skipped = 0
    classes_attempted = 0
    vis_save_path = os.path.join(vis_dir, file_name)

    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size

        if abs(original_width - image_info['width']) > 1 or abs(original_height - image_info['height']) > 1:
             raise Exception(f"Mismatch between actual image dimensions ({original_width}x{original_height}) and annotation dimensions ({image_info['width']}x{image_info['height']}) for {file_name}. Using actual dimensions.")

        input_height, input_width = smart_resize(original_height, original_width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        logger.info(f"Processing image: {file_name} (ID: {image_id}) Original: {original_width}x{original_height} -> Input: {input_width}x{input_height}. Checking class statuses.")

        num_classes_total = len(id_to_prompted_name_map)
        for class_idx, (cat_id, prompted_name) in enumerate(id_to_prompted_name_map.items()):

            if image_processing_status.get(cat_id, False):
                logger.debug(f"Image {image_id}, Class {cat_id} ('{prompted_name}') already processed. Skipping.")
                classes_skipped += 1
                continue

            classes_attempted += 1
            logger.debug(f"Image {image_id} ({file_name}): Querying class {class_idx+1}/{num_classes_total} '{prompted_name}' (ID: {cat_id})")

            single_class_base_prompt = f"Locate every {prompted_name} in the image and output the coordinates in JSON format."
            if base_instructions:
                final_prompt = f"{base_instructions}\n\n{single_class_base_prompt}\n" #Mode not used for single class
            else:
                final_prompt = single_class_base_prompt

            print(f"Prompt for class '{prompted_name}': {final_prompt[:150]}...")
            response, error = inference_with_retry(image_path, final_prompt, logger=logger, rate_limiter=rate_limiter, API_KEY=API_KEY)

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

                        if filtered_boxes_for_class:
                            all_parsed_boxes_for_image.extend(filtered_boxes_for_class)
                            logger.debug(f"Image {image_id}, Class '{prompted_name}': Kept {len(filtered_boxes_for_class)}/{original_label_count} boxes after exact label filtering.")
                else:
                    logger.debug(f"Image {image_id}, Class '{prompted_name}': Inference succeeded but returned no response content.")

            else:
                logger.error(f"Failed inference for class '{prompted_name}' (ID: {cat_id}) on image {image_id} ({file_name}): {error}")
                any_class_inference_failed = True

        logger.info(f"Image {image_id} ({file_name}): Attempted {classes_attempted} classes, Skipped {classes_skipped}, Inference Failures: {any_class_inference_failed}")

        if all_parsed_boxes_for_image:
            logger.info(f"Image {image_id} ({file_name}): Aggregated {len(all_parsed_boxes_for_image)} relevant boxes from newly processed classes.")
            coco_annotations, detected_labels_for_vis = convert_to_coco_format(
                all_parsed_boxes_for_image,
                image_id, original_width, original_height,
                input_width, input_height,
                prompted_name_to_id_map, logger
            )

            if coco_annotations:
                logger.info(f"Image {image_id} ({file_name}): Converted {len(coco_annotations)} boxes to COCO format.")
                visualize_predictions(image_path, detected_labels_for_vis, vis_save_path, logger)
                error_msg_out = "Some classes failed inference but results generated" if any_class_inference_failed else None
                return coco_annotations, error_msg_out, image_id, successfully_processed_classes
            else:
                logger.warning(f"Image {image_id} ({file_name}): Processed classes, got {len(all_parsed_boxes_for_image)} boxes, but no valid COCO annotations derived after conversion/filtering.")
                visualize_predictions(image_path, [], vis_save_path, logger)
                fail_msg = f"No valid COCO annotations derived for {file_name}"
                if any_class_inference_failed: fail_msg += " and some class inferences failed"
                return [], fail_msg, image_id, successfully_processed_classes
        else:
            final_error_msg = f"All attempted class inferences failed for {file_name}" if any_class_inference_failed and classes_attempted > 0 else f"No relevant objects found for attempted classes in {file_name}"
            logger.warning(final_error_msg)
            visualize_predictions(image_path, [], vis_save_path, logger)
            return None, final_error_msg, image_id, successfully_processed_classes

    except Exception as e:
        logger.error(f"Critical error processing image {file_name} (ID: {image_id}): {e}", exc_info=True)
        try:
            if 'frame' in locals():
                 visualize_predictions(image_path, [], vis_save_path, logger)
            else:
                 logger.error("Cannot save visualization on error, frame not loaded.")
        except Exception as vis_e:
            logger.error(f"Additionally failed to save visualization for errored image {file_name}: {vis_e}")
        return None, f"Critical Error processing {file_name}: {e}", image_id, []


def generate_rf100_prompt_maps(categories_dict):
    """
    Generates mappings for RF100 categories (simpler than OdinW).

    Args:
        categories_dict (dict): Mapping from original category ID to original category name.

    Returns:
        tuple: A tuple containing:
            - prompted_name_to_id_map (dict): Maps {category_name.lower() -> original_cat_id}
            - id_to_prompted_name_map (dict): Maps {original_cat_id -> category_name}
    """
    prompted_name_to_id_map = {}
    id_to_prompted_name_map = {}

    for cat_id, original_name in categories_dict.items():
        prompted_name = original_name.strip()

        prompted_name_to_id_map[prompted_name.lower()] = cat_id
        id_to_prompted_name_map[cat_id] = prompted_name

    logger.info(f"Generated RF100 mappings for {len(categories_dict)} categories.")
    logger.debug(f"Prompted Name (lower) -> ID map: {prompted_name_to_id_map}")
    logger.debug(f"ID -> Prompted Name map: {id_to_prompted_name_map}")

    return prompted_name_to_id_map, id_to_prompted_name_map

def process_dataset(dataset_dir, with_instructions=False):
    """Process a single RF100 dataset with per-class status tracking. (Adapted from OdinW)"""
    dataset_name = os.path.basename(dataset_dir)
    logger.info(f"Starting processing for dataset: {dataset_name}")

    suffix = "_instructions" if with_instructions else ""
    results_dir = os.path.join(f"{MODEL_ID}_{BASE_RESULTS_DIR}{suffix}", dataset_name)
    vis_dir = os.path.join(f"{MODEL_ID}_{BASE_VISUALIZE_DIR}{suffix}", dataset_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    test_folder = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_folder):
        raise Exception(f"Test folder not found in {dataset_dir}")

    annotation_file_path = None
    possible_ann_files = ["_annotations.coco.json", "annotations.coco.json"]
    for fname in possible_ann_files:
        potential_path = os.path.join(test_folder, fname)
        if os.path.isfile(potential_path):
             annotation_file_path = potential_path
             logger.info(f"Found annotation file: {annotation_file_path}")
             break
        potential_path_prefix = os.path.join(test_folder, f"{dataset_name}{fname}")
        if os.path.isfile(potential_path_prefix):
            annotation_file_path = potential_path_prefix
            logger.info(f"Found annotation file with prefix: {annotation_file_path}")
            break

    if not annotation_file_path:
        raise Exception(f"COCO annotation file not found in {test_folder} (expected '*_annotations.coco.json' or similar). Skipping dataset.")

    base_instructions = None
    if with_instructions:
        possible_inst_files = ["README.dataset.txt", "instructions.txt", "readme.txt"]
        instruction_file_path = None
        for fname in possible_inst_files:
            potential_path = os.path.join(dataset_dir, fname) # Usually in dataset root
            if os.path.isfile(potential_path):
                instruction_file_path = potential_path
                break
        if instruction_file_path:
            try:
                with open(instruction_file_path, 'r', encoding='utf-8') as f:
                    base_instructions = f.read().strip()
                if base_instructions:
                    logger.info(f"Loaded instructions from {instruction_file_path}")
                else:
                    raise Exception(f"Instruction file found ({instruction_file_path}) but is empty.")
            except Exception as e:
                raise Exception(f"Failed to read instruction file {instruction_file_path}: {e}")
        else:
            raise Exception("Instruction mode enabled, but no instruction file found in {dataset_dir} (checked {possible_inst_files}). Proceeding without instructions.")

    try:
        with open(annotation_file_path, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load or parse annotation file {annotation_file_path}: {e}")

    categories = annotations.get("categories", [])
    if not categories:
        raise Exception(f"No categories found in annotation file: {annotation_file_path}. Skipping dataset.")
    categories_dict = {cat["id"]: cat["name"] for cat in categories}
    logger.info(f"Found {len(categories_dict)} categories for {dataset_name}: {list(categories_dict.values())}")

    images_info = annotations.get("images", [])
    if not images_info:
        raise Exception(f"No images found in annotation file: {annotation_file_path}. Processing may yield empty results.")
    
    prompted_name_to_id_map, id_to_prompted_name_map = generate_rf100_prompt_maps(categories_dict)

    results_file = os.path.join(results_dir, f"qwen_detection_results.json")
    existing_results = load_existing_results(results_file, logger)
    logger.info(f"Found {len(existing_results)} existing annotations in {results_file}.")

    status_filename = STATUS_FILENAME_PATTERN.format(model=MODEL_ID, suffix=suffix.replace("_",""))
    status_file = os.path.join(results_dir, status_filename)
    processing_status = load_processing_status(status_file, logger)
    logger.info(f"Loaded processing status for {len(processing_status)} images from {status_file}.")

    needs_status_update = False
    all_class_ids = set(categories_dict.keys())
    image_ids_in_annotations = set()

    for image_info in images_info:
        if not all(k in image_info for k in ["id", "file_name", "height", "width"]):
             raise Exception(f"Skipping image entry with missing keys in annotations: {image_info}")

        img_id = image_info["id"]
        image_ids_in_annotations.add(img_id)

        if img_id not in processing_status:
            processing_status[img_id] = {cat_id: False for cat_id in all_class_ids}
            needs_status_update = True
            logger.debug(f"Initialized status for new image ID: {img_id}")
        else:
            current_classes_in_status = set(processing_status[img_id].keys())
            missing_classes = all_class_ids - current_classes_in_status
            if missing_classes:
                for cat_id in missing_classes:
                    processing_status[img_id][cat_id] = False
                needs_status_update = True
                logger.debug(f"Initialized status for missing classes {missing_classes} in image ID: {img_id}")
            extra_classes = current_classes_in_status - all_class_ids
            if extra_classes:
                for cat_id in extra_classes:
                    del processing_status[img_id][cat_id]
                needs_status_update = True
                logger.warning(f"Removed status for obsolete classes {extra_classes} in image ID: {img_id}")

    images_to_remove_from_status = set(processing_status.keys()) - image_ids_in_annotations
    if images_to_remove_from_status:
        for img_id in images_to_remove_from_status:
            del processing_status[img_id]
        needs_status_update = True
        logger.warning(f"Removed status entries for {len(images_to_remove_from_status)} images no longer in annotations.")


    if needs_status_update:
         logger.info("Processing status map updated based on current annotations.")
         save_processing_status(processing_status, status_file, logger)

    args_list = []
    images_to_process_count = 0
    fully_processed_count = 0
    skipped_missing_info = 0

    for image_info in images_info:
        if not all(k in image_info for k in ["id", "file_name", "height", "width"]):
             skipped_missing_info += 1
             continue

        image_id = image_info["id"]
        img_status = processing_status.get(image_id)

        if not img_status:
            raise Exception(f"Status map entry missing for image ID {image_id} ({image_info['file_name']}) even after initialization. Skipping.")

        if all(img_status.get(cat_id, False) for cat_id in all_class_ids):
            fully_processed_count += 1
            logger.debug(f"Image {image_id} ({image_info['file_name']}) already fully processed for all classes. Skipping.")
            continue

        args_list.append((
            image_info, image_id, image_info["file_name"],
            image_info["height"], image_info["width"],
            test_folder,
            prompted_name_to_id_map,
            id_to_prompted_name_map,
            results_dir, vis_dir,
            img_status,
            base_instructions
        ))
        images_to_process_count += 1

    if skipped_missing_info > 0:
        raise Exception(f"Skipped {skipped_missing_info} entries from annotation file due to missing keys.")
    logger.info(f"Total valid images in dataset: {len(images_info) - skipped_missing_info}. Images already fully processed: {fully_processed_count}. Images requiring some processing: {images_to_process_count}")

    if not args_list:
        logger.info(f"No images require further processing for dataset {dataset_name}.")
        merged_results_file = os.path.join(results_dir, os.path.basename(results_file))
        if not os.path.exists(merged_results_file) or os.path.getsize(merged_results_file) == 0:
            with open(merged_results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            logger.info(f"Copied existing results ({len(existing_results)} annotations) to merged directory as no new processing was needed: {merged_results_file}")

        else:
            logger.debug(f"Merged results file already exists for {dataset_name}.")

        return results_file, existing_results

    current_results = existing_results.copy()
    processed_images_count = 0
    failed_images_count = 0
    status_updates = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        random.shuffle(args_list)
        futures = [executor.submit(process_image, args) for args in args_list]

        with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="image") as pbar:
            for future in concurrent.futures.as_completed(futures):
                processed_images_count += 1
                try:
                    result_annotations, error_msg, processed_id, successfully_processed_classes = future.result()

                    if result_annotations is not None:
                        if isinstance(result_annotations, list) and result_annotations:
                            start_id = len(current_results)
                            for i, ann in enumerate(result_annotations):
                                ann['id'] = start_id + i
                            current_results.extend(result_annotations)
                            logger.debug(f"Added {len(result_annotations)} new annotations for image {processed_id}.")

                        if error_msg:
                             logger.warning(f"Image ID {processed_id} ({args_list[processed_images_count-1][2]}): Processing completed with message: {error_msg}")

                    elif error_msg:
                        logger.error(f"Failed processing image ID {processed_id} ({args_list[processed_images_count-1][2]}): {error_msg}")
                        failed_images_count += 1
                    else:
                         logger.error(f"Image processing function returned None without an error message for ID {processed_id}. Marking as failed.")
                         failed_images_count += 1

                    if processed_id is not None and successfully_processed_classes:
                        if processed_id not in status_updates:
                            status_updates[processed_id] = {}
                        for cat_id in successfully_processed_classes:
                            status_updates[processed_id][cat_id] = True

                except Exception as e:
                    failed_images_count += 1
                    img_id_for_log = args_list[processed_images_count-1][1] if processed_images_count <= len(args_list) else "UNKNOWN"
                    logger.error(f"Error retrieving result from future for image ID {img_id_for_log}: {e}", exc_info=True)
                finally:
                    pbar.update(1)
                    time_since_last_save = time.time() - pbar.start_t
                    save_interval_images = 50
                    save_interval_time = 600

                    if (processed_images_count % save_interval_images == 0 or time_since_last_save > save_interval_time) and status_updates:
                        logger.info(f"Periodic save triggered after {processed_images_count} images / {time_since_last_save:.0f} seconds.")

                        temp_updated_status_count = 0
                        for img_id, class_updates in status_updates.items():
                            if img_id in processing_status:
                                for cat_id, status in class_updates.items():
                                    if not processing_status[img_id].get(cat_id, False):
                                        processing_status[img_id][cat_id] = status
                                        temp_updated_status_count += 1
                            else:
                                logger.warning(f"Periodic Save: Received status update for image ID {img_id} which is not in the main status map.")

                        if temp_updated_status_count > 0:
                             logger.info(f"Applying {temp_updated_status_count} periodic status updates...")
                             save_processing_status(processing_status, status_file, logger)
                             status_updates.clear()
                             logger.info("Periodic status save complete.")
                             if time_since_last_save > save_interval_time:
                                 pbar.start_t = time.time()

                        try:
                            temp_results_file = results_file + ".tmp"
                            with open(temp_results_file, 'w') as f:
                                json.dump(current_results, f, indent=2)
                            os.replace(temp_results_file, results_file)
                            logger.info(f"Periodic save of results successful to {results_file}")
                        except Exception as e:
                             logger.error(f"Error during periodic saving of results file {results_file}: {e}")


    final_updated_status_count = 0
    if status_updates:
        logger.info(f"Applying final status updates for {len(status_updates)} images...")
        for img_id, class_updates in status_updates.items():
            if img_id in processing_status:
                for cat_id, status in class_updates.items():
                    if not processing_status[img_id].get(cat_id, False):
                        processing_status[img_id][cat_id] = status
                        final_updated_status_count += 1
            else:
                logger.warning(f"Final Save: Received status update for image ID {img_id} which is not in the main status map.")
        logger.info(f"Applied {final_updated_status_count} final individual class status updates.")

    logger.info(f"Finished processing {dataset_name}. Images Attempted: {processed_images_count}, Failed Images: {failed_images_count}")
    logger.info(f"Total annotations for {dataset_name} after processing: {len(current_results)}")

    try:
        with open(results_file, 'w') as f:
            json.dump(current_results, f, indent=2)
        logger.info(f"Saved final updated results for {dataset_name} to {results_file}")
    except Exception as e:
         raise Exception(f"Error saving final results file {results_file}: {e}")

    if final_updated_status_count > 0:
        save_processing_status(processing_status, status_file, logger)
    else:
        logger.info("No final status updates to save.")

    return results_file, current_results


def main():
    """Main function to process all RF100 datasets"""
    parser = argparse.ArgumentParser(description='Run Qwen-VL single-class evaluation on RF100 datasets.')
    parser.add_argument('--with_instructions', action='store_true',
                        help='Use dataset README/instructions to prefix the detection prompt.')
    parser.add_argument('--dataset_dir', type=str, default="rf100_datasets",
                        help='Base directory containing the extracted RF100 dataset folders (e.g., "." if datasets are in current dir, or "rf100_datasets")')
    args = parser.parse_args()

    suffix = "_instructions" if args.with_instructions else ""
    output_dir_root = f"{MODEL_ID}_{BASE_RESULTS_DIR}{suffix}"
    visualize_dir_root = f"{MODEL_ID}_{BASE_VISUALIZE_DIR}{suffix}"

    # Create base output directories
    os.makedirs(output_dir_root, exist_ok=True)
    os.makedirs(visualize_dir_root, exist_ok=True)

    all_items = glob.glob(os.path.join(args.dataset_dir, "*"))
    dataset_dirs = [d for d in all_items
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, "test"))]

    if not dataset_dirs:
        raise Exception(f"No datasets found in '{args.dataset_dir}'. A dataset is expected to be a folder containing a 'test' subfolder.")

    logger.info(f"Found {len(dataset_dirs)} potential dataset directories in '{args.dataset_dir}'")

    for dataset_path in tqdm(dataset_dirs, desc="Overall Dataset Progress", unit="dataset"):
        dataset_name = os.path.basename(dataset_path)
        try:
            logger.info(f"--- Processing dataset: {dataset_name} ---")
            process_dataset(dataset_path, with_instructions=args.with_instructions)
            logger.info(f"--- Finished dataset: {dataset_name} ---")
        except Exception as e:
            logger.error(f"Critical error processing dataset {dataset_name} at path {dataset_path}: {e}", exc_info=True)

    logger.info("RF100 Qwen-VL single-class evaluation script finished.")

if __name__ == "__main__":

    main()