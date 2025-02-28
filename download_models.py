#!/usr/bin/env python3
"""
Download script for waste classification models.
This script downloads the model files from HuggingFace if they don't exist locally.
"""

import os
import sys
import requests
from tqdm import tqdm

# Model URLs from HuggingFace
MODEL_URLS = {
    "efficientnetv2s_waste_classifier.keras": "https://huggingface.co/ashuwhy/efficientnetv2s/resolve/main/efficientnetv2s_waste_classifier.keras",
    "efficientnetv2s_waste_classifier_fine_tuned.keras": "https://huggingface.co/ashuwhy/efficientnetv2s/resolve/main/efficientnetv2s_waste_classifier_fine_tuned.keras",
    "efficientnetv2s_waste_classifier_final.keras": "https://huggingface.co/ashuwhy/efficientnetv2s/resolve/main/efficientnetv2s_waste_classifier_final.keras"
}

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    print(f"Downloading {os.path.basename(destination)} ({total_size / (1024*1024):.2f} MB)")
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    
    print(f"Downloaded {os.path.basename(destination)} to {destination}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check and download each model
    for model_name, model_url in MODEL_URLS.items():
        model_path = os.path.join(script_dir, model_name)
        
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists at {model_path}")
        else:
            print(f"Model {model_name} not found. Downloading...")
            try:
                download_file(model_url, model_path)
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
    
    print("\nDownload complete. Available models:")
    for model_name in MODEL_URLS.keys():
        model_path = os.path.join(script_dir, model_name)
        if os.path.exists(model_path):
            print(f"✅ {model_name} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")
        else:
            print(f"❌ {model_name} (not available)")

if __name__ == "__main__":
    main() 