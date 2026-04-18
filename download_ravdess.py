#!/usr/bin/env python3
"""
Download and setup RAVDESS dataset for tasks 1.6, 1.7, 1.8
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
import argparse


def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    print(f"\n✅ Downloaded {filename}")


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Extracted to {extract_to}")


def setup_ravdess_dataset():
    """Setup RAVDESS dataset directory structure"""
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    ravdess_dir = datasets_dir / "RAVDESS"
    
    # Check if already exists
    if ravdess_dir.exists() and any(ravdess_dir.glob("Actor_*")):
        print(f"✅ RAVDESS dataset already exists at {ravdess_dir}")
        actor_folders = list(ravdess_dir.glob("Actor_*"))
        print(f"Found {len(actor_folders)} Actor folders")
        return str(ravdess_dir)
    
    print("📥 RAVDESS dataset not found locally.")
    print("🔗 Please download the RAVDESS dataset manually from:")
    print("   https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
    print("   OR")
    print("   https://zenodo.org/record/1188976")
    print()
    print("📁 After downloading, extract it to: datasets/RAVDESS/")
    print("   The structure should be:")
    print("   datasets/RAVDESS/")
    print("   ├── Actor_01/")
    print("   ├── Actor_02/")
    print("   ├── ...")
    print("   └── Actor_24/")
    print()
    
    # Create placeholder structure for now
    ravdess_dir.mkdir(exist_ok=True)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Download and setup RAVDESS dataset')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if dataset exists, do not download')
    
    args = parser.parse_args()
    
    print("🎵 RAVDESS Dataset Setup")
    print("=" * 50)
    
    dataset_path = setup_ravdess_dataset()
    
    if dataset_path:
        print(f"\n✅ Dataset ready at: {dataset_path}")
        return dataset_path
    else:
        print("\n⚠️  Dataset not found. Please download manually.")
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🚀 Ready to run training with:")
        print(f"python run_all_tasks.py --data {result}")
    else:
        sys.exit(1)