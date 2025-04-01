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
    5. Saves all results to the specified output directory
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run customer segmentation analysis pipeline.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input CSV file containing customer data')
    parser.add_argument('--output', type=str, default='segmentation_results',
                        help='Directory where to save all results')
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_dir = Path(args.output)
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
    
    # Step 1: Process the data
    print("\n[1/3] Processing customer behavioral data...")
    processed_df = behavioral_data_processing(args.input)
    print(f"     ✓ Processed {len(processed_df)} customer records")
    
    # Step 2: Generate segment profiles
    print("\n[2/3] Generating customer segment profiles...")
    profile_data = get_segment_profile_data(processed_df)
    
    # Save profiles to JSON
    import json
    with open(profile_json_path, 'w') as f:
        json.dump(profile_data, f, indent=4)
    
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


if __name__ == "__main__":
    main()