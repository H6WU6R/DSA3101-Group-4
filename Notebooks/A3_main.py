"""
Main entry point for customer segmentation analysis pipeline.

This script orchestrates the complete workflow by:
1. Processing raw customer behavioral data
2. Creating customer segment profiles
3. Generating visualizations for segment analysis

Usage:
    python main.py --input [csv_file_path] --output [output_directory]
"""

import os
import argparse
import time
from pathlib import Path

# Import modules from the project
from data_processing import behavioral_data_processing
from segment_profiler import get_segment_profile_data
from segment_visualizer import visualize_segment_comparisons


def main():
    """
    Main function to run the complete customer segmentation analysis pipeline.
    
    This function:
    1. Parses command line arguments
    2. Processes the input customer data
    3. Generates segment profiles
    4. Creates visualizations
    5. Saves all results to the specified output directory (by default, in the same directory as this script)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run customer segmentation analysis pipeline.')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to the input CSV file containing customer data')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory where to save all results (default: in the same directory as this script)')
    
    args = parser.parse_args()
    
    # Get the input file path - use default path if not provided
    input_path = args.input
    if input_path is None:
        # Try to locate the default data file in common locations
        default_paths = [
            '/Users/cindy/Desktop/DSA3101-Project-3/Data/A1-segmented_df.csv',
            './Data/A1-segmented_df.csv',
            '../Data/A1-segmented_df.csv',
            './A1-segmented_df.csv'
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                input_path = path
                print(f"Using default data file: {input_path}")
                break
        
        if input_path is None:
            print("\nERROR: No input file specified and could not find default data file.")
            print("Please provide the path to your input CSV file using the --input argument:")
            print("Example: python main.py --input path/to/your/data.csv\n")
            return
    
    # Verify that input file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please check the file path and try again.")
        return
    
    try:
        # Determine the output directory
        # If not specified, use the directory of this script
        if args.output is None:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'segmentation_results')
        else:
            output_dir = args.output
        
        # Create output directory structure
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profiles_dir = output_dir / 'profiles'
        profiles_dir.mkdir(exist_ok=True)
        
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Set the profile JSON path
        profile_json_path = profiles_dir / 'profiles.json'
        
        # Start timing the process
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"CUSTOMER SEGMENTATION ANALYSIS PIPELINE")
        print(f"{'='*60}")
        print(f"Results will be saved to: {output_dir}")
        
        # Step 1: Process the data
        print("\n[1/3] Processing customer behavioral data...")
        processed_df = behavioral_data_processing(input_path)
        print(f"     ✓ Processed {len(processed_df)} customer records")
        
        # Step 2: Generate segment profiles
        print("\n[2/3] Generating customer segment profiles...")
        profile_data = get_segment_profile_data(processed_df)
        
        # Save profiles to JSON
        # Convert NumPy types to Python native types to make them JSON serializable
        import json
        import numpy as np
        
        def convert_numpy_types(obj):
            """
            Recursively convert NumPy types to native Python types for JSON serialization.
            """
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Convert all NumPy types in the profile data
        serializable_profile_data = convert_numpy_types(profile_data)
        
        # Make sure all keys at the top level are strings (JSON requirement)
        string_keyed_data = {str(k): v for k, v in serializable_profile_data.items()}
        
        with open(profile_json_path, 'w') as f:
            json.dump(string_keyed_data, f, indent=4)
        
        num_segments = len(profile_data)
        print(f"     ✓ Created profiles for {num_segments} customer segments")
        print(f"     ✓ Saved profiles to {profile_json_path}")
        
        # Step 3: Generate visualizations
        print("\n[3/3] Creating segment visualizations...")
        vis_paths = visualize_segment_comparisons(profile_json_path, str(vis_dir))
        
        print(f"     ✓ Generated {len(vis_paths)} visualization charts")
        
        # Report completion
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        print("\nVisualization files created:")
        for path in vis_paths:
            print(f"  - {os.path.basename(path)}")
        print("\nTo view the complete analysis, open the visualizations in the output directory.")
    
    except Exception as e:
        print(f"\nERROR: An exception occurred during processing:")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your input data and try again.")


if __name__ == "__main__":
    main()