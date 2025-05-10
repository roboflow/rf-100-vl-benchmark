import os
import json
import pandas as pd
from collections import defaultdict
import argparse
import sys # Added for sys.exit on critical errors

# Default configuration values (can be overridden by command-line args)
DEFAULT_METRIC_KEY = "regular"  # Use 'regular' or 'present_only' results
DEFAULT_METRIC_INDEX = 0       # Index of the specific metric (0 is usually mAP @ IoU=0.50:0.95)
HTML_OUTPUT_FILENAME = "multi_debug_table.html"

# Define the preferred nested structure ending
PREFERRED_NESTED_ENDING = os.path.join("results", "results")

def find_results_json(mode_dir_path):
    """
    Searches within a mode directory for a JSON results file.
    Priority 1: Looks for a path ending in 'results/results/*.json'.
    Priority 2: Looks for a path ending in 'results/*.json'.
    Returns the path to the first .json file found according to priority.
    """
    print(f"   Searching for JSON in: '{mode_dir_path}'...")
    target_json_path = None
    found_preferred = False
    found_fallback = False

    # --- Pass 1: Check for the preferred 'results/results' structure ---
    print(f"    -> Priority 1: Searching for '{PREFERRED_NESTED_ENDING}' structure...")
    for root, dirs, files in os.walk(mode_dir_path):
        # Check if the current directory 'root' ends with the preferred structure
        # Use os.path.normpath to handle different OS separators (/ vs \)
        if os.path.normpath(root).endswith(os.path.normpath(PREFERRED_NESTED_ENDING)):
            print(f"      -> Found preferred folder structure: {root}")
            json_files_found = [f for f in files if f.lower().endswith(".json")]
            if json_files_found:
                target_json_path = os.path.join(root, json_files_found[0])
                print(f"      -> Found JSON file (Priority 1): {target_json_path}")
                found_preferred = True
                break # Found the best option, stop searching
            else:
                print(f"      -> WARNING: Found preferred folder structure '{root}' but no .json file within.")
                # Don't break yet, maybe the fallback structure exists elsewhere

    # --- Pass 2: If preferred not found, check for fallback 'results' structure ---
    if not found_preferred:
        print(f"    -> Priority 2: Searching for any 'results' folder structure...")
        for root, dirs, files in os.walk(mode_dir_path):
            # Check if the *current directory's name* is 'results'
            # Using os.path.basename is generally reliable here
            if os.path.basename(root) == "results":
                 # Avoid re-checking the preferred path if it was already found empty
                 is_preferred_path = os.path.normpath(root).endswith(os.path.normpath(PREFERRED_NESTED_ENDING))
                 if is_preferred_path:
                      continue # Already checked and found empty in Pass 1

                 print(f"      -> Found fallback 'results' directory: {root}")
                 json_files_found = [f for f in files if f.lower().endswith(".json")]
                 if json_files_found:
                     target_json_path = os.path.join(root, json_files_found[0])
                     print(f"      -> Found JSON file (Priority 2): {target_json_path}")
                     found_fallback = True
                     break # Found a fallback option, stop searching
                 else:
                     print(f"      -> WARNING: Found fallback 'results' folder '{root}' but no .json file within.")
                     # Continue search in case another 'results' folder exists deeper

    # --- Final Result ---
    if target_json_path:
        return target_json_path
    else:
        print(f"   -> WARNING: Could not find any suitable .json file in expected 'results' locations within '{mode_dir_path}'.")
        return None

def process_results_file(file_path, mode_display_name, metric_key, metric_index, results_data, all_datasets):
    """
    Reads a JSON results file and populates the results_data dictionary.
    """
    try:
        print(f" -> Reading {mode_display_name} results from: {file_path}")
        with open(file_path, 'r') as f:
            mode_data = json.load(f)

        # Process per-dataset results for this mode
        per_dataset_results = mode_data.get("per_dataset", {})
        found_data_for_mode = False
        for dataset_name, metrics in per_dataset_results.items():
            all_datasets.add(dataset_name) # Track all unique dataset names
            metric_list = metrics.get(metric_key)
            if metric_list and len(metric_list) > metric_index:
                score = metric_list[metric_index]
                results_data[dataset_name][mode_display_name] = score
                found_data_for_mode = True
            else:
                results_data[dataset_name][mode_display_name] = None
                print(f"    -> WARNING: Metric '{metric_key}' (index {metric_index}) not found or invalid for dataset '{dataset_name}' in {mode_display_name}.")

        # Process overall average results for this mode
        overall_average_results = mode_data.get("overall_average", {})
        overall_metric_list = overall_average_results.get(metric_key)
        if overall_metric_list and len(overall_metric_list) > metric_index:
            overall_score = overall_metric_list[metric_index]
            results_data["Overall Average"][mode_display_name] = overall_score
            found_data_for_mode = True # Mark as found even if only overall is present
        else:
            results_data["Overall Average"][mode_display_name] = None
            print(f"    -> WARNING: Overall average metric '{metric_key}' (index {metric_index}) not found or invalid for {mode_display_name}.")

        if not found_data_for_mode:
             print(f"   -> WARNING: No data extracted for mode '{mode_display_name}' using metric '{metric_key}' index {metric_index} from {file_path}.")


    except FileNotFoundError:
        print(f"   ERROR: File not found: {file_path}. Skipping mode '{mode_display_name}'.")
        # Ensure keys exist even if skipped, initialize with None
        results_data["Overall Average"][mode_display_name] = None
        for dataset_name in list(all_datasets): # Ensure dataset entries exist for this mode
             results_data[dataset_name][mode_display_name] = None
    except json.JSONDecodeError:
        print(f"   ERROR: Could not decode JSON from {file_path}. Skipping mode '{mode_display_name}'.")
    except Exception as e:
        print(f"   ERROR: An unexpected error occurred processing {file_path}: {e}")

def main(args):
    """
    Main function to discover modes, process results, and generate table.
    """
    base_dir = args.base_dir
    metric_key = args.metric_key
    metric_index = args.metric_index

    if not os.path.isdir(base_dir):
        print(f"ERROR: Base directory not found or is not a directory: {base_dir}")
        sys.exit(1) # Exit if base directory is invalid

    print(f"Starting results processing from base directory: {base_dir}")
    print(f"Using Metric Key: '{metric_key}', Metric Index: {metric_index}")
    print("-" * 50)

    # --- Discover Modes and Find Results Files ---
    discovered_modes = {} # mode_name: path_to_json
    try:
        for item_name in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item_name)
            if os.path.isdir(item_path):
                mode_name = item_name # Use directory name as mode identifier
                print(f"Found potential mode directory: {mode_name}")
                json_path = find_results_json(item_path)
                if json_path:
                    discovered_modes[mode_name] = json_path
                else:
                     # Handle cases where a mode dir exists but no valid JSON path is found
                    print(f" -> No suitable JSON results file found for mode '{mode_name}'. Skipping.")
                    # Optionally, we could still add the mode with None values later,
                    # but let's only include modes where we found a file.
    except OSError as e:
        print(f"ERROR: Could not read base directory {base_dir}: {e}")
        sys.exit(1)

    if not discovered_modes:
        print("\nERROR: No mode directories with valid result files found. Exiting.")
        sys.exit(1)

    print("-" * 50)
    print("Processing discovered evaluation results...")

    # --- Process Each Found Results File ---
    # data[dataset_name][mode_display_name] = score
    results_data = defaultdict(dict)
    all_datasets = set()
    processed_modes = sorted(discovered_modes.keys()) # Sort modes alphabetically

    for mode_name in processed_modes:
        file_path = discovered_modes[mode_name]
        process_results_file(file_path, mode_name, metric_key, metric_index, results_data, all_datasets)

    # --- Prepare and Display DataFrame ---
    if not results_data:
        print("\nNo results data successfully extracted to create a table.")
    else:
        # Ensure consistent column order (datasets first, then overall)
        sorted_datasets = sorted(list(all_datasets))
        columns_ordered = sorted_datasets + ["Overall Average"]

        # Create DataFrame (Modes as index, Datasets + Overall as columns)
        df_data = {}
        for mode_name in processed_modes: # Use the sorted list of modes we actually processed
            row_data = {}
            for col_name in columns_ordered:
                # Get score, default to None if dataset/overall key or mode key missing
                row_data[col_name] = results_data.get(col_name, {}).get(mode_name, None)
            df_data[mode_name] = row_data

        df = pd.DataFrame.from_dict(df_data, orient='index', columns=columns_ordered)

        # Fill potential missing values (e.g., if a metric was missing) with 0.0
        # Consider if NaN or another placeholder is more appropriate depending on needs
        df = df.fillna(0.0)

        # --- Apply Bolding ---
        def highlight_max(s, props='font-weight: bold'):
            """Highlights the max value(s) in a Series (column)"""
            # Ensure we handle non-numeric gracefully if fillna wasn't used or failed
            try:
                 max_val = s.max()
                 return [props if v == max_val else '' for v in s]
            except TypeError:
                 # Handle columns that might not be numeric after all
                 return [''] * len(s)


        # Apply styling - highlight max in each column
        styled_df = df.style.apply(highlight_max, axis=0) # axis=0 applies column-wise

        # Format numbers to 3 decimal places (or as desired)
        styled_df = styled_df.format("{:.3f}")

        # --- Print Table ---
        print("\n" + "="*80)
        print(f"Comparison of mAP Scores (Metric: {metric_key}, Index: {metric_index})")
        print("="*80)
        # Use to_string() for console alignment
        print(styled_df.to_string())
        print("="*80)

        # --- Optional: Save to HTML (preserves formatting) ---
        try:
            output_html_path = os.path.join(os.getcwd(), HTML_OUTPUT_FILENAME) # Save in current dir
            styled_df.to_html(output_html_path, escape=False)
            print(f"\nFormatted table saved to: {output_html_path}")
        except Exception as e:
            print(f"\nError saving table to HTML: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process detection metrics from multiple mode directories and generate a comparison table.")

    parser.add_argument("--base-dir",
                        required=True,
                        help="The base directory containing the mode-specific subdirectories.")
    parser.add_argument("--metric-key",
                        default=DEFAULT_METRIC_KEY,
                        help=f"The primary key in the JSON for the metrics list (e.g., 'regular', 'present_only'). Default: '{DEFAULT_METRIC_KEY}'")
    parser.add_argument("--metric-index",
                        type=int,
                        default=DEFAULT_METRIC_INDEX,
                        help=f"The index within the metric list to extract the score (e.g., 0 for mAP@0.50:0.95). Default: {DEFAULT_METRIC_INDEX}")

    args = parser.parse_args()
    main(args)