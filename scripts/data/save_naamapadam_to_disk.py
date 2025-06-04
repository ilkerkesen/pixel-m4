#!/usr/bin/env python3
"""
Simple script to download the Naamapadam dataset from Hugging Face and save all language configs to disk.

Usage:
  python save_naamapadam.py --output-dir /path/to/save_dir
"""
import os
import argparse
from datasets import load_dataset, get_dataset_config_names

def main():
    parser = argparse.ArgumentParser(
        description="Save ai4bharat/naamapadam data to disk for the selected languages."
    )
    parser.add_argument(
        "--save-dir", required=True,
        help="Directory where prepared datasets will be saved."
    )
    args = parser.parse_args()

    repo_id = "ai4bharat/naamapadam"
    os.makedirs(args.save_dir, exist_ok=True)

    # Retrieve all available configuration names
    config_names = ["hi", "bn", "ta", "te"]
    print(f"Will download configs: {config_names}")

    # Download each config and save
    for cfg in config_names:
        print(f"Downloading config '{cfg}'...")
        ds = load_dataset(repo_id, cfg)  # positional argument for config name
        output_path = os.path.join(args.save_dir, f"{cfg}")
        ds.save_to_disk(output_path)
        print(f"Saved config '{cfg}' to {output_path}\n")

if __name__ == "__main__":
    main()
