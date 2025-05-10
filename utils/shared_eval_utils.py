import threading
import time
import random

ODINW13_DATASETS = [
    "AerialMaritimeDrone", "Aquarium", "CottontailRabbits", "EgoHands",
    "NorthAmericaMushrooms", "Packages", "PascalVOC", "pistols", "pothole",
    "Raccoon", "ShellfishOpenImages", "thermalDogsAndPeople",
    "VehiclesOpenImages"
]

ODINW_CONFIG = {
    "AerialMaritimeDrone": {
        "data_root_suffix": "large/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    },
    "Aquarium": {
        "data_root_suffix": "Aquarium Combined.v2-raw-1024.coco/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    },
    "CottontailRabbits": {
        "data_root_suffix": "",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": {
            'Cottontail-Rabbit': {'name': 'rabbit'}
        }
    },
    "EgoHands": {
        "data_root_suffix": "generic/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": {
            'hand': {'suffix': ' of a person'}
        }
    },
    "NorthAmericaMushrooms": {
        "data_root_suffix": "North American Mushrooms.v1-416x416.coco/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": {
            'CoW': {'name': 'flat mushroom'},
            'chanterelle': {'name': 'yellow mushroom'}
        }
    },
    "Packages": {
        "data_root_suffix": "Raw/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": {
            'package': {'prefix': 'there is a ', 'suffix': ' on the porch'}
        }
    },
    "PascalVOC": {
        "data_root_suffix": "",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    },
    "pistols": {
        "data_root_suffix": "export/",
        # Note the different annotation file name for pistols
        "ann_file_pattern": "val_annotations_without_background.json",
        "img_prefix": "", # Images are in the same directory as the annotation file
        "caption_prompt": None
    },
    "pothole": {
        "data_root_suffix": "",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": {
            'pothole': {'prefix': 'there are some ', 'name': 'holes', 'suffix': ' on the road'}
        }
    },
    "Raccoon": {
        "data_root_suffix": "Raccoon.v2-raw.coco/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    },
    "ShellfishOpenImages": {
        "data_root_suffix": "raw/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    },
    "thermalDogsAndPeople": {
        "data_root_suffix": "",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    },
    "VehiclesOpenImages": {
        "data_root_suffix": "416x416/",
        "ann_file_pattern": "valid/annotations_without_background.json",
        "img_prefix": "valid/",
        "caption_prompt": None
    }
}

class RateLimiter:
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
        
    def __call__(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                print(f"Sleeping for {sleep_time} seconds")
                if sleep_time > 0:
                    jitter = random.uniform(0, 0.1 * sleep_time)
                    print(f"Jittering for {jitter} seconds")
                    time.sleep(sleep_time + jitter)
                    now = time.time()
                self.calls = self.calls[1:]
                
            self.calls.append(now)

def is_rate_limit_error(error_msg):
    """Check if an error message indicates a rate limit issue"""
    rate_limit_indicators = [
        "rate limit", "ratelimit", "too many requests", 
        "429", "throttl", "quota exceeded", "limit exceeded", "overloaded", "quota", "limit",
        "current limit", "current quota", "current usage", "unavailable", "later", "try again"
    ]
    error_lower = str(error_msg).lower()
    return any(indicator in error_lower for indicator in rate_limit_indicators)

def generate_odinw_prompt_maps(categories_dict, caption_prompt_config):
    """
    Generates mappings between original category IDs, original names,
    and the potentially modified names used in prompts based on ODinW caption rules.

    Args:
        categories_dict (dict): Mapping from original category ID to original category name.
        caption_prompt_config (dict or None): Configuration for custom prompts per category.

    Returns:
        tuple: A tuple containing:
            - prompted_name_to_id_map (dict): Maps {prompted_name.lower() -> original_cat_id}
            - id_to_prompted_name_map (dict): Maps {original_cat_id -> prompted_name}
    """
    prompted_name_to_id_map = {}
    id_to_prompted_name_map = {}

    if caption_prompt_config is None:
        caption_prompt_config = {}

    calculated_prompted_names = {}

    for cat_id, original_name in categories_dict.items():
        prompt_details = caption_prompt_config.get(original_name, {})
        prefix = prompt_details.get('prefix', '')
        name = prompt_details.get('name', original_name)
        suffix = prompt_details.get('suffix', '')

        prompted_name = f"{prefix}{name}{suffix}".strip()
        calculated_prompted_names[cat_id] = prompted_name

        prompted_name_to_id_map[prompted_name.lower()] = cat_id
        id_to_prompted_name_map[cat_id] = prompted_name

    return prompted_name_to_id_map, id_to_prompted_name_map