import requests
import os
import zipfile
import pandas as pd
import time
import json
from urllib.parse import urlparse
from pathlib import Path
from functools import wraps
import requests
import shutil

def retry_with_backoff(max_retries=5, initial_delay=0.1, exception_types=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    tries += 1
                    if tries == max_retries:
                        raise e
                    sleep_time = initial_delay + (tries ** 2) * initial_delay
                    time.sleep(sleep_time)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=30, initial_delay=0.1, exception_types=(KeyError,))
def get_rf_download_link_from_url(dataset_url):
    dataset_url = dataset_url.replace("/dataset/", "/")
    url = os.path.join(dataset_url, "coco")
    url = url.replace("https://universe", "https://api")
    url = url.replace("https://app", "https://api")
    print("Getting download link from", url)
    response = requests.get(url, params={"api_key": os.getenv("ROBOFLOW_API_KEY")})
    response.raise_for_status()
    print(response.json())
    link = response.json()["export"]["link"]
    print(link)
    return link

def fix_coco_annotations(dataset_dir):
    """
    Fix COCO annotations by removing the dataset name from categories
    and remapping all category IDs to start from 0.
    """
    # Find all annotation files
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('_annotations.coco.json'):
                json_path = os.path.join(root, file)
                print(f"Fixing annotations in: {json_path}")
                
                try:
                    # Load the JSON file
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if the categories list has the dataset name as class 0
                    if len(data['categories']) > 0 and data['categories'][0]['supercategory'] == 'none':
                        # Create a mapping from old category IDs to new ones (shifting by -1)
                        category_map = {}
                        new_categories = []
                        
                        # Remove the first category (dataset name) and remap others
                        for i, category in enumerate(data['categories'][1:]):
                            old_id = category['id']
                            new_id = i  # Start from 0
                            category_map[old_id] = new_id
                            
                            # Update the category with the new ID
                            category['id'] = new_id
                            # Keep supercategory as 'none' for simplicity
                            category['supercategory'] = 'none'
                            new_categories.append(category)
                        
                        # Update categories in the JSON
                        data['categories'] = new_categories
                        
                        # Update all annotations with the new category IDs
                        for annotation in data['annotations']:
                            old_cat_id = annotation['category_id']
                            if old_cat_id in category_map:
                                annotation['category_id'] = category_map[old_cat_id]
                        
                        # Save the modified JSON file
                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        
                        print(f"Successfully fixed annotations in {json_path}")
                    else:
                        print(f"No remapping needed for {json_path}")
                
                except Exception as e:
                    print(f"Error fixing annotations in {json_path}: {e}")

import tempfile
from urllib.parse import urlparse
import os
import shutil
import requests
import zipfile


def extract_dataset_name(dataset_url):
    """Extract and return a unique dataset name from the Roboflow URL."""
    parsed_url = urlparse(dataset_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if "dataset" in path_parts:
        dataset_index = path_parts.index("dataset")
        dataset_name = path_parts[dataset_index - 1]
    else:
        dataset_name = path_parts[-2]
    return dataset_name #.replace('-', '_').replace(' ', '_')


def finalize_dataset(dataset_dir):
    """Handle final dataset processing steps after moving data to final directory."""
    fix_coco_annotations(dataset_dir)


def download_and_extract_dataset(dataset_url, base_output_dir="rf100_datasets"):
    """Download, extract dataset, fix annotations, and store in a unique folder."""

    dataset_name = extract_dataset_name(dataset_url)
    print(f"Dataset name extracted from URL: {dataset_name}")

    os.makedirs(base_output_dir, exist_ok=True)

    download_link = get_rf_download_link_from_url(dataset_url)
    if not download_link:
        print(f"Failed to get download link for {dataset_name}")
        return False

    print(f"Downloading {dataset_name}...")
    try:
        response = requests.get(download_link)
        response.raise_for_status()

        zip_path = os.path.join(base_output_dir, f"{dataset_name}.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"Extracting {dataset_name} to temporary folder...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            extracted_contents = os.listdir(tmp_dir)
            if len(extracted_contents) == 1 and os.path.isdir(os.path.join(tmp_dir, extracted_contents[0])):
                extracted_folder = os.path.join(tmp_dir, extracted_contents[0])
            else:
                extracted_folder = tmp_dir

            final_dataset_dir = os.path.join(base_output_dir, dataset_name)

            if os.path.exists(final_dataset_dir):
                shutil.rmtree(final_dataset_dir)

            shutil.move(extracted_folder, final_dataset_dir)

        finalize_dataset(final_dataset_dir)

        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed zip file: {zip_path}.")

        print(f"Successfully downloaded, extracted, and fixed {dataset_name}")
        return True

    except Exception as e:
        print(f"Error downloading or extracting {dataset_name}: {e}")
        return False

def main():
    # Path to the CSV file
    csv_file = "urls.csv"
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        if "Link (Link to version)" not in df.columns:
            print(f"CSV file does not contain the expected column: 'Link (Link to version)'")
            return
        
        # Get the list of dataset URLs
        dataset_urls = df["Link (Link to version)"].tolist()
        
        # Track failed URLs for retry
        failed_urls = []
        
        # Download and extract each dataset
        print(f"Found {len(dataset_urls)} datasets to download")
        
        for i, url in enumerate(dataset_urls):
            print(f"\nProcessing dataset {i+1}/{len(dataset_urls)}")
            success = download_and_extract_dataset(url)
            
            # If download failed, add to retry list
            if not success:
                failed_urls.append(url)
            
            # Add a small delay between downloads to avoid potential rate limiting 
            if i < len(dataset_urls) - 1:
                time.sleep(2)
        
        # Retry failed downloads
        if failed_urls:
            print(f"\n\nRetrying {len(failed_urls)} failed downloads...")
            
            # Save failed URLs to a file for reference
            with open("failed_urls.csv", "w") as f:
                f.write("Link (Link to version)\n")
                for url in failed_urls:
                    f.write(f"{url}\n")
            
            retry_failed_urls = []
            for i, url in enumerate(failed_urls):
                print(f"\nRetry attempt {i+1}/{len(failed_urls)}")
                success = download_and_extract_dataset(url)
                
                # If still failed after retry, add to final failed list
                if not success:
                    retry_failed_urls.append(url)
                
                # Add a longer delay between retries
                if i < len(failed_urls) - 1:
                    time.sleep(5)
            
            # Save final failed URLs to a file
            if retry_failed_urls:
                with open("final_failed_urls.csv", "w") as f:
                    f.write("Link (Link to version)\n")
                    for url in retry_failed_urls:
                        f.write(f"{url}\n")
                print(f"\nAfter retries, {len(retry_failed_urls)} downloads still failed. See final_failed_urls.csv")
            else:
                print("\nAll retries successful!")
        
        print("\nAll datasets processed!")
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")

if __name__ == "__main__":
    main()