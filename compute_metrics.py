import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_coco_metrics(gt_json, pred_json):
    """
    Computes COCO metrics for a single dataset.

    Returns a list of 12 metrics. If the predictions file is empty ([]),
    returns a zero list of length 12.
    """
    with open(pred_json, 'r') as f:
        try:
            preds = json.load(f)
        except json.JSONDecodeError:
            print(f"Debug: {pred_json} cannot be parsed as JSON, returning zero metrics.")
            return [0.0] * 12

    if not preds:  # If preds == []
        print(f"Debug: {pred_json} has no predictions, returning zero metrics.")
        return [0.0] * 12

    # Load ground truth and predictions
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    # Evaluate via COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats.tolist()  # return all coco metrics


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
    with open(pred_json, 'r') as f:
        try:
            predictions = json.load(f)
        except json.JSONDecodeError:
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
    Saves visualizations showing ground truth vs. prediction bboxes for ALL images
    in the dataset if the dataset name matches the target.
    - GT bboxes in green
    - Predicted bboxes in red
    """
    # Only process all images for our target dataset
    if "aerial-airport-7ap9o-fsod-ddgc-4qt0q" not in dataset_name:
        # For other datasets, just use the original function
        visualize_one_image(gt_json, pred_json, images_folder, dataset_name, out_folder)
        return
        
    if not os.path.exists(pred_json):
        print(f"Debug: {pred_json} not found, skipping visualization.")
        return  # If no predictions file found, skip
    
    # Load JSONs
    with open(pred_json, 'r') as f:
        try:
            predictions = json.load(f)
        except json.JSONDecodeError:
            print(f"Debug: {pred_json} cannot be parsed as JSON, skipping visualization.")
            return
    
    coco_gt = COCO(gt_json)
    
    # Get all image IDs from ground truth
    img_ids = coco_gt.getImgIds()
    if not img_ids:
        print(f"Debug: No images found in ground truth for {dataset_name}")
        return  # No images to visualize
    
    # Create output directory for this dataset
    dataset_out_folder = os.path.join(out_folder, dataset_name)
    os.makedirs(dataset_out_folder, exist_ok=True)
    
    print(f"Debug: Processing ALL {len(img_ids)} images for {dataset_name}")
    
    # Process each image
    for img_id in img_ids:
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
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw predicted bounding boxes (in red)
        pred_boxes_for_image = [p for p in predictions if p.get('image_id') == img_id]
        for p in pred_boxes_for_image:
            bbox = p['bbox']  # [x, y, w, h]
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Save visualization
        out_path = os.path.join(dataset_out_folder, f"{img_info['file_name']}")
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f"{dataset_name} - {img_info['file_name']}")
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    print(f"Debug: Saved {len(img_ids)} visualizations for {dataset_name} to {dataset_out_folder}")


def main(dataset_root, predictions_root, name_to_label_file, output_metrics_file, visuals_folder, model_name="gemini"):
    """
    Process multiple directories with detection files.
    
    Args:
        dataset_root: Root directory containing dataset folders
        predictions_root: Root directory containing prediction JSON files (e.g., gemini_detection_merged)
        name_to_label_file: JSON file mapping dataset names to cluster labels
        output_metrics_file: Output file to save metrics
        visuals_folder: Folder to save visualizations
        model_name: Name of the model (e.g., "gemini", "qwen") - used for file naming
    """
    # Read cluster mapping
    with open(name_to_label_file, 'r') as f:
        name_to_label_old = json.load(f)

    # Create a mapping from dataset names to labels
    name_to_label = {}
    name_to_label_lst = list(name_to_label_old.keys())
    name_to_label_lst.sort()
    dataset_names = os.listdir(dataset_root)
    dataset_names.sort()
    
    # Match dataset names with labels
    for old_name, new_name in zip(name_to_label_lst, dataset_names):
        name_to_label[new_name] = name_to_label_old[old_name]

    all_metrics = {}      # Maps dataset_name -> metrics list of length 12
    cluster_metrics = {}  # Maps cluster_name -> list of metrics lists

    # Get all prediction directories in the predictions_root
    prediction_dirs = [d for d in os.listdir(predictions_root) 
                      if os.path.isdir(os.path.join(predictions_root, d))]
    
    print(f"Found {len(prediction_dirs)} prediction directories")
    
    # Define the results filename based on the model name
    results_filename = f"just_instructions_{model_name}_detection_results.json"
    
    # Process each prediction directory
    for pred_dir in tqdm(prediction_dirs, desc="Processing datasets"):
        dataset_name = pred_dir  # Directory name is the dataset name
        
        # Find the corresponding dataset folder in dataset_root
        dataset_folder = os.path.join(dataset_root, dataset_name, 'test')
        if not os.path.exists(dataset_folder):
            print(f"Skipping {dataset_name}, dataset folder not found.")
            continue
            
        # Find ground truth annotation file
        gt_json = os.path.join(dataset_folder, '_annotations.coco.json')
        if not os.path.exists(gt_json):
            print(f"Skipping {dataset_name}, ground truth file not found.")
            continue
            
        # Find prediction file using the model-specific filename
        pred_json = os.path.join(predictions_root, dataset_name, results_filename)
        if not os.path.exists(pred_json):
            print(f"Skipping {dataset_name}, prediction file {results_filename} not found.")
            continue

        print(f"Calculating metrics for '{dataset_name}'...")
        try:
            metrics = calculate_coco_metrics(gt_json, pred_json)
            all_metrics[dataset_name] = metrics
            
            # Check if dataset is in the mapping
            if dataset_name in name_to_label:
                print(f"Debug: Metrics for '{dataset_name}' -> {metrics}")
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in mapping file.")
                # Assign to an "unknown" cluster
                name_to_label[dataset_name] = "unknown"
        except Exception as e:
            # If for any reason there's an error, store zero metrics
            print(f"Error processing '{dataset_name}': {e}")
            all_metrics[dataset_name] = [0.0] * 12

        # Visualize on all images for the dataset
        images_folder = dataset_folder
        visualize_all_images(gt_json, pred_json, images_folder, dataset_name, visuals_folder)

    # Aggregate metrics under their respective clusters
    for dataset_name, metrics in all_metrics.items():
        cluster_name = name_to_label.get(dataset_name, "unknown")
        print(f"Debug: dataset_name='{dataset_name}' mapped to cluster='{cluster_name}'")
        cluster_metrics.setdefault(cluster_name, []).append(metrics)

    # Compute average metrics for each cluster
    averaged_cluster_metrics = {}
    for cluster, metric_list in cluster_metrics.items():
        print(f"\nDebug: Computing average for cluster='{cluster}'\n  Number of datasets in this cluster = {len(metric_list)}\n  Raw metrics list = {metric_list}\n")
        averaged_cluster_metrics[cluster] = np.mean(metric_list, axis=0).tolist()

    # Calculate overall average metrics across all datasets
    overall_average_metrics = np.mean(list(all_metrics.values()), axis=0).tolist()

    # Save metrics to output file
    output_data = {
        "per_dataset": all_metrics,
        "per_cluster": averaged_cluster_metrics,
        "overall_average": overall_average_metrics
    }
    
    # Add model name to output file
    output_metrics_file_with_model = output_metrics_file.replace(".json", f"_{model_name}.json")
    with open(output_metrics_file_with_model, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nMetrics saved to '{output_metrics_file_with_model}'")
    print("Debug: Completed metric computation.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute metrics for object detection results')
    parser.add_argument('--dataset_root', type=str, default="rf100_datasets",
                        help='Root directory containing dataset folders')
    parser.add_argument('--predictions_root', type=str, required=True,
                        help='Root directory containing prediction JSON files')
    parser.add_argument('--name_to_label_file', type=str, default="upd_mappings.json",
                        help='JSON file mapping dataset names to cluster labels')
    parser.add_argument('--output_metrics_file', type=str, default="metrics_results_instructions.json",
                        help='Output file to save metrics')
    parser.add_argument('--visuals_folder', type=str, required=True,
                        help='Folder to save visualizations')
    parser.add_argument('--model_name', type=str, default="gemini",
                        help='Name of the model (e.g., "gemini", "qwen") - used for file naming')
    
    args = parser.parse_args()
    
    main(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        name_to_label_file=args.name_to_label_file,
        output_metrics_file=args.output_metrics_file,
        visuals_folder=args.visuals_folder,
        model_name=args.model_name
    )