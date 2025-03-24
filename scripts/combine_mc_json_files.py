"""
This script combines multiple Minecraft conversation JSON files into one larger dataset.
Each input file should be in the format created by convert_mc_txt_to_json.py:

[
  {
    "instruction": "...",
    "input": "...",
    "output": "..."
  },
  ...
]

The script will combine all entries from all input files into a single output JSON file
with the same format. It can process individual JSON files or directories containing JSON files.
"""

import os
import json
import argparse
import glob

def combine_json_files(input_paths, output_file):
    """
    Combine multiple JSON files into one larger dataset.
    
    Args:
        input_paths (list): List of paths to input JSON files or directories
        output_file (str): Path to the output combined JSON file
    """
    combined_data = []
    all_json_files = []
    
    # Process each input path (file or directory)
    for path in input_paths:
        if os.path.isdir(path):
            # If path is a directory, find all JSON files in it
            json_files = glob.glob(os.path.join(path, "*.json"))
            all_json_files.extend(json_files)
            print(f"Found {len(json_files)} JSON files in directory: {path}")
        elif os.path.isfile(path) and path.endswith('.json'):
            # If path is a JSON file
            all_json_files.append(path)
        else:
            print(f"Warning: {path} is not a JSON file or directory. Skipping.")
    
    # Process each JSON file
    for file_path in all_json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                    print(f"Added {len(data)} entries from {file_path}")
                else:
                    print(f"Warning: {file_path} does not contain a JSON array. Skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write combined data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined dataset created with {len(combined_data)} total entries")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine multiple Minecraft JSON files into one dataset')
    parser.add_argument('--input_paths', type=str, nargs='+', required=True,
                        help='List of input JSON files or directories containing JSON files')
    parser.add_argument('--output_file', type=str, default="data/custom/combined_mc_data.json",
                        help='Path to output combined JSON file (default: data/custom/combined_mc_data.json)')
    
    args = parser.parse_args()
    
    combine_json_files(args.input_paths, args.output_file)
