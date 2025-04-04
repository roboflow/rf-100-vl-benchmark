import argparse
import os
import requests
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_names", default="all", type=str)  # "all" or names joined by comma
argparser.add_argument("--dataset_path", default="DATASET/odinw", type=str)
args = argparser.parse_args()

# Use the resolve URL format
root = "https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35"

all_datasets = [
    "AerialMaritimeDrone",
    "Aquarium", 
    "CottontailRabbits",
    "EgoHands",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "Raccoon", 
    "ShellfishOpenImages",
    "VehiclesOpenImages",
    "pistols",
    "pothole",
    "thermalDogsAndPeople"
]

datasets_to_download = []
if args.dataset_names == "all":
    datasets_to_download = all_datasets
else:
    datasets_to_download = args.dataset_names.split(",")

# Create dataset path if it doesn't exist
os.makedirs(args.dataset_path, exist_ok=True)

# Set up headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

for dataset in datasets_to_download:
    if dataset in all_datasets:
        zip_url = f"{root}/{dataset}.zip"
        zip_path = f"{args.dataset_path}/{dataset}.zip"
        
        print(f"Downloading dataset: {dataset}")
        
        try:
            # Download the file with requests
            response = requests.get(zip_url, headers=headers, stream=True)
            
            if response.status_code == 200:
                # Save the file
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                os.system(f"unzip {zip_path} -d {args.dataset_path}")
                
                # Remove the zip file
                os.system(f"rm {zip_path}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
            else:
                print(f"ERROR: Failed to download: {zip_url} (Status code: {response.status_code})")
        except Exception as e:
            print(f"ERROR: Exception when downloading {dataset}: {str(e)}")
    else:
        print(f"Dataset not found: {dataset}")