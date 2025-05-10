import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import argparse
import warnings # Import warnings to suppress COCOeval output during loops

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Suppress print statements from COCOeval inside the loop
import sys
from io import StringIO

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def calculate_coco_metrics_modes(gt_json, pred_json):
    """
    Computes COCO metrics in two modes:
    1. Regular: Standard COCO evaluation across the whole dataset.
    2. Present Only: Average of per-image evaluations, where each image is
       evaluated ONLY on the ground truth classes present in that specific image.

    Returns a dictionary:
    {
        'regular': [list of 12 standard COCO metrics],
        'present_only': [list of 12 averaged per-image metrics]
    }
    Returns {'regular': [0.0]*12, 'present_only': [0.0]*12} if predictions are empty or invalid.
    """
    results = {
        'regular': [0.0] * 12,
        'present_only': [0.0] * 12
    }

    # --- Load Predictions ---
    try:
        with open(pred_json, 'r') as f:
            preds_list = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {pred_json} cannot be parsed as JSON, returning zero metrics.")
        return results
    except FileNotFoundError:
         print(f"Error: Prediction file {pred_json} not found, returning zero metrics.")
         return results # Added FileNotFoundError check

    if not preds_list:  # If preds_list == []
        print(f"Info: {pred_json} has no predictions, returning zero metrics.")
        return results

    # --- Load Ground Truth ---
    try:
        coco_gt = COCO(gt_json)
    except Exception as e:
        print(f"Error loading ground truth {gt_json}: {e}. Returning zero metrics.")
        return results


    # === Mode 1: Regular COCO Evaluation ===
    print("--- Calculating Regular COCO Metrics ---")
    try:
        # Use a copy for safety, although loadRes creates its own index
        coco_dt_regular = coco_gt.loadRes(preds_list)

        coco_eval_regular = COCOeval(coco_gt, coco_dt_regular, 'bbox')
        coco_eval_regular.evaluate()
        coco_eval_regular.accumulate()
        print("Regular Evaluation Summary:")
        coco_eval_regular.summarize() # Print summary for regular mode
        results['regular'] = coco_eval_regular.stats.tolist()
    except Exception as e:
        print(f"Error during regular COCO evaluation: {e}. Regular metrics set to zero.")
        # results['regular'] is already zero initialized

    # === Mode 2: Present Classes Only Evaluation ===
    print("--- Calculating Present-Classes-Only Metrics ---")
    img_ids = sorted(coco_gt.getImgIds())
    if not img_ids:
        print("Warning: No image IDs found in ground truth. Present-only metrics set to zero.")
        return results # Cannot proceed without images

    per_image_stats = []
    # Wrap image loop in tqdm for progress
    for img_id in tqdm(img_ids, desc="Present-Only Eval (per image)", leave=False):
        # 1. Get GT annotations and present category IDs for THIS image
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        if not ann_ids:
            continue # Skip images with no GT

        gt_anns = coco_gt.loadAnns(ann_ids)
        present_cat_ids = list(set(ann['category_id'] for ann in gt_anns))

        if not present_cat_ids:
             continue # Skip if no categories found (shouldn't happen if ann_ids exist)

        # 2. Filter predictions for THIS image AND ONLY the present categories
        img_preds = [p for p in preds_list if p['image_id'] == img_id and p['category_id'] in present_cat_ids]

        # --- START: Added Check for Empty img_preds ---
        if not img_preds:
            # If there are no predictions matching the GT categories for this image,
            # then AP/AR for this image evaluation is 0.
            # Append zeros directly and skip COCOeval for this image.
            per_image_stats.append(np.zeros(12))
            # print(f"Debug: Img {img_id}: Found GT for cats {present_cat_ids}, but no matching preds. Appending zeros.")
            continue # Move to the next image
        # --- END: Added Check ---

        # 3. Prepare COCO prediction object (only if img_preds is not empty)
        try:
             # This line is now only reached if img_preds is NOT empty
            coco_dt_img = coco_gt.loadRes(img_preds) # Pass the filtered list

            # 4. Run COCOeval configured for THIS image and present categories
            coco_eval_img = COCOeval(coco_gt, coco_dt_img, 'bbox')
            coco_eval_img.params.imgIds = [img_id]       # Evaluate only this image
            coco_eval_img.params.catIds = present_cat_ids # Evaluate only these categories
            coco_eval_img.params.useCats = 1 # Ensure category filtering is active

            # Suppress print statements during evaluate/accumulate/summarize for loop
            with SuppressPrint():
                coco_eval_img.evaluate()
                coco_eval_img.accumulate()
                coco_eval_img.summarize() # Need to run summarize to compute stats array

            if coco_eval_img.stats is not None:
                per_image_stats.append(coco_eval_img.stats)
            else:
                 print(f"Warning: COCOeval stats were None for image {img_id}. Appending zeros.")
                 per_image_stats.append(np.zeros(12)) # Append zeros if stats are None

        except Exception as e:
            # Catch errors during loadRes or COCOeval for this specific image
            print(f"Error evaluating image {img_id} (after filtering preds): {e}. Appending zeros for this image.")
            per_image_stats.append(np.zeros(12)) # Append zeros if error occurs

    # 5. Average the stats across all processed images
    if per_image_stats:
        # Use nanmean to ignore potential NaNs if any image eval failed in a weird way
        # Convert list of arrays to a 2D numpy array first
        stats_array = np.array(per_image_stats)
        # Replace any potential NaNs/Infs resulting from division by zero in COCOeval (e.g., no GT/DT matches)
        stats_array[np.isnan(stats_array)] = 0.0
        stats_array[np.isinf(stats_array)] = 0.0 # Handle potential infinity as well
        averaged_stats = np.mean(stats_array, axis=0).tolist()
        results['present_only'] = averaged_stats
        print(f"Averaged Present-Only Metrics (over {len(per_image_stats)} images with GT):")
        # Print a summary similar to COCOeval's summarize() output
        stat_names = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                      'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                      'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                      'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                      'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                      'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                      'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                      'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                      'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                      'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                      'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                      'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']
        for name, val in zip(stat_names, averaged_stats):
            print(f' {name} = {val:0.3f}')

    else:
        print("Warning: No images with GT annotations found or processed. Present-only metrics set to zero.")
        # results['present_only'] already zero initialized

    return results


# --- Keep visualize functions as they are ---
def visualize_one_image(gt_json, pred_json, images_folder, dataset_name, out_folder):
    """
    Saves a single visualization image showing ground truth vs. prediction bboxes.
    - GT bboxes in green
    - Predicted bboxes in red
    """
    if not os.path.exists(pred_json):
        print(f"Debug: {pred_json} not found, skipping visualization.")
        return  # If no predictions file found, skip

    # Load JSONs
    try:
      with open(pred_json, 'r') as f:
          predictions = json.load(f)
    except json.JSONDecodeError:
        print(f"Debug: {pred_json} cannot be parsed for viz, skipping.")
        return
    except FileNotFoundError:
        print(f"Debug: {pred_json} not found for viz, skipping.")
        return


    coco_gt = COCO(gt_json)

    # Grab the first image ID from ground truth
    img_ids = coco_gt.getImgIds()
    if not img_ids:
        return  # No images to visualize
    img_id = img_ids[0]
    img_info = coco_gt.loadImgs(img_id)[0]

    # Load the actual image
    img_path = os.path.join(images_folder, img_info['file_name'])
    if not os.path.exists(img_path):
        print(f"Debug: image file {img_path} not found, skipping visualization.")
        return
    image = cv2.imread(img_path)
    if image is None:
        print(f"Debug: failed to read image from {img_path}, skipping visualization.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw ground-truth bounding boxes (in green)
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    gt_anns = coco_gt.loadAnns(ann_ids)
    for ann in gt_anns:
        bbox = ann['bbox']  # [x, y, w, h]
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw predicted bounding boxes (in red)
    pred_boxes_for_image = [p for p in predictions if p.get('image_id') == img_id]
    for p in pred_boxes_for_image:
        bbox = p['bbox']  # [x, y, w, h]
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save visualization
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{dataset_name}.png")
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"{dataset_name}")
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def visualize_all_images(gt_json, pred_json, images_folder, dataset_name, out_folder):
    """
    Saves visualizations showing ground truth vs. prediction bboxes for up to 3 images
    in the dataset.
    - GT bboxes in green
    - Predicted bboxes in red
    """
    if not os.path.exists(pred_json):
        print(f"Debug: {pred_json} not found, skipping visualization.")
        return  # If no predictions file found, skip

    # Load JSONs
    try:
        with open(pred_json, 'r') as f:
            predictions = json.load(f)
    except json.JSONDecodeError:
        print(f"Debug: {pred_json} cannot be parsed as JSON, skipping visualization.")
        return
    except FileNotFoundError:
        print(f"Debug: {pred_json} not found, skipping visualization.")
        return


    try:
        coco_gt = COCO(gt_json)
    except Exception as e:
        print(f"Debug: Failed to load GT {gt_json} for viz: {e}")
        return

    # Get all image IDs from ground truth
    img_ids = coco_gt.getImgIds()
    if not img_ids:
        print(f"Debug: No images found in ground truth for {dataset_name}")
        return  # No images to visualize

    # Create output directory for this dataset
    dataset_out_folder = os.path.join(out_folder, dataset_name)
    os.makedirs(dataset_out_folder, exist_ok=True)

    print(f"Debug: Processing up to 3 images for visualization in {dataset_name}")

    images_saved_count = 0 # Initialize counter for saved images
    max_images_to_save = 3 # Define the maximum number of images to save

    # Process each image
    for img_id in tqdm(img_ids, desc=f"Visualizing {dataset_name}", leave=False):
        # --- Start Check: Break if enough images saved ---
        if images_saved_count >= max_images_to_save:
            print(f"Debug: Reached limit of {max_images_to_save} visualized images for {dataset_name}.")
            break
        # --- End Check ---

        try:
            img_info = coco_gt.loadImgs(img_id)[0]

            # Load the actual image
            img_path = os.path.join(images_folder, img_info['file_name'])
            if not os.path.exists(img_path):
                print(f"Debug: image file {img_path} not found, skipping.")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Debug: failed to read image from {img_path}, skipping.")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Draw ground-truth bounding boxes (in green)
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            gt_anns = coco_gt.loadAnns(ann_ids)
            for ann in gt_anns:
                bbox = ann['bbox']  # [x, y, w, h]
                x, y, w, h = map(int, bbox)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green GT

            # Draw predicted bounding boxes (in red)
            pred_boxes_for_image = [p for p in predictions if p.get('image_id') == img_id]
            for p in pred_boxes_for_image:
                bbox = p['bbox']  # [x, y, w, h]
                score = p.get('score', 1.0) # Get score if available
                cat_id = p.get('category_id', '?') # Get category ID
                x, y, w, h = map(int, bbox)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # Red Predictions
                # Optional: Add text for score/class
                # label = f"Cls {cat_id} {score:.2f}"
                # cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


            # Save visualization
            out_path = os.path.join(dataset_out_folder, f"{img_info['file_name']}") # Use original filename
            plt.figure(figsize=(12, 9)) # Slightly larger figure
            plt.imshow(image)
            plt.title(f"{dataset_name} - {img_info['file_name']}")
            plt.axis('off')
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            images_saved_count += 1 # Increment counter after successful save

        except Exception as e:
            print(f"Error visualizing image id {img_id} in dataset {dataset_name}: {e}")
            continue # Continue with the next image

    print(f"Debug: Saved {images_saved_count} visualizations for {dataset_name} to {dataset_out_folder}")


def main(dataset_root, predictions_root, name_to_label_file, model_name="gemini"):
    """
    Process multiple directories with detection files. Calculates regular and
    present-only metrics. Derives output paths from predictions_root.

    Args:
        dataset_root: Root directory containing dataset folders
        predictions_root: Root directory containing prediction JSON files (e.g., gemini_detection_merged)
        name_to_label_file: JSON file mapping dataset names to cluster labels
        model_name: Name of the model (e.g., "gemini", "qwen") - used for file naming
    """
    # --- Derive Output Paths ---
    # Create a 'results' subdirectory inside the predictions_root
    results_base_dir = os.path.join(predictions_root, "results")
    # Define specific paths for visuals and metrics within the results_base_dir
    visuals_folder = os.path.join(results_base_dir, "visualizations")
    output_metrics_file_base = os.path.join(results_base_dir, "detection_metrics") # Base name, model/mode added later

    # Create output directories if they don't exist
    os.makedirs(results_base_dir, exist_ok=True)
    os.makedirs(visuals_folder, exist_ok=True)
    print(f"Outputting results to: {os.path.abspath(results_base_dir)}") # Show absolute path for clarity
    print(f"Visualizations will be saved in: {os.path.abspath(visuals_folder)}")
    print(f"Metrics file base name: {output_metrics_file_base}")
    # --- End Derive Output Paths ---


    # Read cluster mapping
    try:
        with open(name_to_label_file, 'r') as f:
            name_to_label_old = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file {name_to_label_file} not found. Cannot map clusters.")
        name_to_label_old = {}
    except json.JSONDecodeError:
         print(f"Error: Mapping file {name_to_label_file} is not valid JSON. Cannot map clusters.")
         name_to_label_old = {}


    # Create a mapping from dataset names to labels
    name_to_label = {}
    name_to_label_lst = sorted(list(name_to_label_old.keys())) # Sort old names

    try:
        dataset_names = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]) # Sort dataset names
    except FileNotFoundError:
        print(f"Error: Dataset root {dataset_root} not found. Exiting.")
        return

    # Match dataset names with labels - requires careful checking
    # This assumes a 1:1 correspondence based on sorted lists, which might be fragile.
    # A safer approach would be if dataset names were keys in name_to_label_old directly.
    # Assuming current logic is intended:
    if len(name_to_label_lst) == len(dataset_names):
         for old_name, new_name in zip(name_to_label_lst, dataset_names):
            name_to_label[new_name] = name_to_label_old[old_name]
            print(f"Mapping: '{new_name}' (from dataset dir) -> Cluster '{name_to_label_old[old_name]}' (using mapping key '{old_name}')")
    else:
        print(f"Warning: Mismatch between number of keys in mapping file ({len(name_to_label_lst)}) and dataset folders found ({len(dataset_names)}). Cluster mapping might be incorrect.")
        # Fallback: Try direct mapping if dataset names exist as keys
        for ds_name in dataset_names:
            if ds_name in name_to_label_old:
                 name_to_label[ds_name] = name_to_label_old[ds_name]
            else:
                 print(f"Warning: Dataset '{ds_name}' not found directly in mapping file keys.")
                 name_to_label[ds_name] = "unknown" # Assign to unknown if not found


    # Store metrics for both modes
    # Structure: { dataset_name: { 'regular': [stats], 'present_only': [stats] } }
    all_metrics = {}
    # Structure: { cluster_name: { 'regular': [[stats1], [stats2], ...], 'present_only': [[stats1], [stats2], ...] } }
    cluster_metrics_raw = {}

    # Get all prediction directories in the predictions_root
    try:
        prediction_dirs = sorted([d for d in os.listdir(predictions_root)
                          if os.path.isdir(os.path.join(predictions_root, d))])
    except FileNotFoundError:
        print(f"Error: Predictions root {predictions_root} not found. Exiting.")
        return

    print(f"Found {len(prediction_dirs)} prediction directories in {predictions_root}")

    # Define the results filename based on the model name - Ensure this matches your prediction files
    # Example uses 'qwen_detection_results.json', adjust if needed
    results_filename = "gemini_detection_results.json" #f"qwen_detection_results.json" #"debug_multi_pro_gemini_detection_results.json" #"qwen_single_class_detection_results.json" #"upd_flash_gemini_detection_results.json"
    # results_filename = f"{model_name}_detection_results.json" # Or use model_name if files are named that way

    # Process each prediction directory (assuming it matches dataset name)
    for dataset_name in tqdm(prediction_dirs, desc="Processing datasets"):

        # Find the corresponding dataset folder in dataset_root
        dataset_folder = os.path.join(dataset_root, dataset_name, 'test') # Assuming 'test' subfolder
        if not os.path.exists(dataset_folder):
            print(f"Skipping {dataset_name}, dataset test folder '{dataset_folder}' not found.")
            continue

        # Find ground truth annotation file
        gt_json = os.path.join(dataset_folder, '_annotations.coco.json') # Standard name
        if not os.path.exists(gt_json):
            print(f"Skipping {dataset_name}, ground truth file '{gt_json}' not found.")
            continue

        # Find prediction file using the determined filename
        pred_json = os.path.join(predictions_root, dataset_name, results_filename)
        # No need for exists check here, calculate_coco_metrics_modes handles it


        print(f"\nProcessing dataset '{dataset_name}'...")

        # Calculate both metrics modes
        metrics_dict = calculate_coco_metrics_modes(gt_json, pred_json)
        all_metrics[dataset_name] = metrics_dict

        cluster_name = name_to_label.get(dataset_name, "unknown") # Use .get for safety

        # Initialize cluster dict if first time seeing this cluster
        if cluster_name not in cluster_metrics_raw:
            cluster_metrics_raw[cluster_name] = {'regular': [], 'present_only': []}

        # Append metrics for this dataset to the correct cluster and mode
        cluster_metrics_raw[cluster_name]['regular'].append(metrics_dict['regular'])
        cluster_metrics_raw[cluster_name]['present_only'].append(metrics_dict['present_only'])

        print(f"Finished processing '{dataset_name}'. Mapped to cluster '{cluster_name}'.")


        # Visualize - Run visualization regardless of metric calculation success
        images_folder = dataset_folder # Visualize images from the 'test' folder
        print(f"Running visualization for '{dataset_name}'...")
        visualize_all_images(gt_json, pred_json, images_folder, dataset_name, visuals_folder)


    # --- Aggregate and Average Metrics ---

    # Compute average metrics for each cluster for both modes
    averaged_cluster_metrics = {}
    print("\n--- Averaging Cluster Metrics ---")
    for cluster, mode_metrics in cluster_metrics_raw.items():
        print(f"Cluster: '{cluster}'")
        averaged_cluster_metrics[cluster] = {}
        for mode, metric_list in mode_metrics.items():
            if metric_list:
                # Use nanmean, assuming metric_list contains lists/arrays of numbers
                avg_metrics = np.nanmean(np.array(metric_list), axis=0).tolist()
                averaged_cluster_metrics[cluster][mode] = avg_metrics
                print(f"  Mode: '{mode}', Avg AP (mAP): {avg_metrics[0]:.4f} (from {len(metric_list)} datasets)")
            else:
                averaged_cluster_metrics[cluster][mode] = [0.0] * 12 # Handle empty cluster
                print(f"  Mode: '{mode}', No datasets found.")


    # Calculate overall average metrics across all datasets for both modes
    print("\n--- Averaging Overall Metrics ---")
    overall_average_metrics = {'regular': [0.0] * 12, 'present_only': [0.0] * 12}
    all_regular_metrics = [m['regular'] for m in all_metrics.values() if m] # Get list of regular lists
    all_present_only_metrics = [m['present_only'] for m in all_metrics.values() if m] # Get list of present_only lists

    if all_regular_metrics:
        overall_average_metrics['regular'] = np.nanmean(np.array(all_regular_metrics), axis=0).tolist()
        print(f"Overall Regular Avg AP (mAP): {overall_average_metrics['regular'][0]:.4f} (from {len(all_regular_metrics)} datasets)")
    else:
        print("No regular metrics found to compute overall average.")

    if all_present_only_metrics:
        overall_average_metrics['present_only'] = np.nanmean(np.array(all_present_only_metrics), axis=0).tolist()
        print(f"Overall Present-Only Avg AP (mAP): {overall_average_metrics['present_only'][0]:.4f} (from {len(all_present_only_metrics)} datasets)")
    else:
         print("No present-only metrics found to compute overall average.")


    # --- Save Metrics ---
    output_data = {
        "per_dataset": all_metrics,           # Contains {dataset: {'regular':[], 'present_only':[]}}
        "per_cluster_average": averaged_cluster_metrics, # Contains {cluster: {'regular':[], 'present_only':[]}}
        "overall_average": overall_average_metrics     # Contains {'regular':[], 'present_only':[]}
    }

    # --- START: Use derived output_metrics_file_base ---
    # Add model name to output file using the derived base path
    output_metrics_file_with_model = output_metrics_file_base + f"_{model_name}_modes.json"
    # --- END: Use derived output_metrics_file_base ---
    try:
        with open(output_metrics_file_with_model, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\nMetrics saved to '{output_metrics_file_with_model}'")
    except IOError as e:
         print(f"\nError saving metrics to {output_metrics_file_with_model}: {e}")


    print("Debug: Completed metric computation and visualization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute metrics (regular and present-only) for object detection results')
    parser.add_argument('--dataset_root', type=str, default="rf100_datasets",
                        help='Root directory containing dataset folders (e.g., ./rf100_datasets)')
    parser.add_argument('--predictions_root', type=str, required=True,
                        help='Root directory containing prediction JSON files (e.g., ./gemini_preds). Output paths will be derived from this.')
    parser.add_argument('--name_to_label_file', type=str, default="upd_mappings.json",
                        help='JSON file mapping dataset names (or keys) to cluster labels')
    parser.add_argument('--model_name', type=str, default="gemini",
                        help='Name of the model (e.g., "gemini", "qwen") - used for file naming/output')

    args = parser.parse_args()

    # Add basic validation for paths
    if not os.path.isdir(args.dataset_root):
        print(f"Error: dataset_root '{args.dataset_root}' not found or not a directory.")
        sys.exit(1)
    if not os.path.isdir(args.predictions_root):
        print(f"Error: predictions_root '{args.predictions_root}' not found or not a directory.")
        sys.exit(1)
    # Visuals folder creation is now handled inside main

    # --- START: Updated main call ---
    main(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        name_to_label_file=args.name_to_label_file,
        model_name=args.model_name
    )
    # --- END: Updated main call ---