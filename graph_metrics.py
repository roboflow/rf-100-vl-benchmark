import json
import numpy as np
import matplotlib.pyplot as plt

def load_first_metric_map(filepath):
    """
    Loads the 'per_cluster' and 'overall_average' from a COCO metrics JSON file,
    returning:
      - cluster_map: dict of {cluster_name: first_metric_value}
      - overall_value: the first_metric from overall_average
    If any cluster is missing or overall_average is missing, those entries are 0.0 by default.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    per_cluster = data.get("per_cluster", {})
    overall_list = data.get("overall_average", [])

    # Gather cluster -> first metric
    cluster_map = {}
    for cluster_name, metrics_array in per_cluster.items():
        if metrics_array:
            cluster_map[cluster_name] = metrics_array[0]
        else:
            cluster_map[cluster_name] = 0.0

    # Overall is the first index if it exists
    if overall_list:
        overall_value = overall_list[0]
    else:
        overall_value = 0.0

    return cluster_map, overall_value

def plot_groundingdino_map(file_path, output_image="groundingdino_map.png", y_limit=0.3):
    """
    Reads a GroundingDINO COCO metric JSON file, extracts the first metric for each cluster & overall,
    and produces a bar chart with:
      - All clusters in alphabetical order on the left
      - A dotted line
      - Overall on the far right
    """
    # Load cluster -> first_metric and overall from the file
    cluster_map, overall_value = load_first_metric_map(file_path)
    
    # Sort cluster names so they're consistently ordered on the x-axis
    sorted_clusters = sorted(cluster_map.keys())
    
    # Prepare y-values for cluster bars
    y_values = [cluster_map[cl] for cl in sorted_clusters]
    
    # We'll place each cluster at indices [0..N-1], then skip one index, put 'Overall' at index N+1
    n_clusters = len(sorted_clusters)
    cluster_x = np.arange(n_clusters)      # e.g. [0, 1, 2, ..., N-1]
    overall_x = n_clusters + 1             # place Overall to the right with a gap
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot cluster bars
    bars = plt.bar(cluster_x, y_values, width=0.6)
    
    # Get the color from the cluster bars for consistency
    bar_color = bars[0].get_facecolor()
    
    # Plot Overall bar with matching color
    plt.bar(overall_x, overall_value, width=0.6, color=bar_color)
    
    # Create tick labels
    # We'll place the cluster labels at the cluster_x positions
    # and "Overall" at overall_x
    x_labels = list(sorted_clusters) + ["", "Overall"]  # add a blank for the gap
    x_positions = list(cluster_x) + [n_clusters, overall_x]
    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    
    # Dotted line after the last cluster
    if n_clusters > 0:
        # The vertical line is halfway between the last cluster index and overall_x
        mid_line = (cluster_x[-1] + overall_x) / 2
        plt.axvline(x=mid_line - 0.5, linestyle="--", color="gray")
    
    # Y-axis limit
    plt.ylim(0, y_limit)
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Add value label for overall bar
    plt.text(overall_x, overall_value + 0.01,
            f'{overall_value:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.title("GroundingDINO COCO mAP 50:95 per Cluster + Overall")
    plt.xlabel("Cluster")
    plt.ylabel("mAP")
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Saved GroundingDINO chart to '{output_image}'")

def main():
    """
    Generate a bar chart for GroundingDINO metrics showing mAP for each cluster, 
    plus overall on the far right.
    """
    plot_groundingdino_map(
        file_path="instr_gemini.json",
        output_image="inst_seg_map_gemini.png",
        y_limit=0.3  # Adjust this value based on your data range
    )

if __name__ == "__main__":
    main()