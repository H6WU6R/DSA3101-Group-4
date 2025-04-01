
import os
import sys
import argparse
import json
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to sys.path to ensure imports work correctly
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Try different import strategies to handle potential module import issues
try:
    # First try standard imports
    from data_processing import behavioral_data_processing
    from segment_profiler import get_segment_profile_data
    from segment_visualizer import visualize_segment_comparisons
except ImportError:
    # If that fails, try importing from the current directory
    try:
        import data_processing
        import segment_profiler
        import segment_visualizer
        
        behavioral_data_processing = data_processing.behavioral_data_processing
        get_segment_profile_data = segment_profiler.get_segment_profile_data
        visualize_segment_comparisons = segment_visualizer.visualize_segment_comparisons
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
        print("Make sure all required Python packages are installed:")
        print("pip install pandas numpy scikit-learn matplotlib")
        sys.exit(1)


def main():
    """
    Main function to run the complete customer segmentation analysis pipeline.
    
    This function:
    1. Processes the input data
    2. Generates segment profiles
    3. Creates visualizations for the segments
    4. Saves all outputs to the script directory
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate customer segment profiles and visualizations.')
    parser.add_argument('--input', type=str, 
                        default='../src/A1_Customer_Segmentation/A1-segmented_df.csv',
                        help='Path to the input CSV file containing customer data')
    
    args = parser.parse_args()
    
    # Set output directory to the script directory
    output_dir = SCRIPT_DIR
    
    print(f"Starting customer segmentation analysis pipeline...")
    print(f"Input data: {args.input}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Process the data
    print("\n=== Step 1: Processing Data ===")
    try:
        processed_df = behavioral_data_processing(args.input)
        print(f"Successfully processed data with shape: {processed_df.shape}")
        print(f"Columns in processed data: {', '.join(processed_df.columns)}")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print(f"Detailed error: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 2: Generate segment profiles
    print("\n=== Step 2: Generating Segment Profiles ===")
    try:
        profile_data = get_segment_profile_data(processed_df)
        segments = list(profile_data.keys())
        print(f"Successfully generated profiles for {len(segments)} segments: {segments}")
        
        # Save profiles to JSON in the script directory
        profile_json_path = os.path.join(output_dir, 'profiles.json')
        
        # Convert any non-serializable objects to string or simple types
        serializable_data = {}
        for segment, data in profile_data.items():
            serializable_data[str(segment)] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    serializable_data[str(segment)][key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            serializable_data[str(segment)][key][k] = {str(kk): str(vv) for kk, vv in v.items()}
                        else:
                            serializable_data[str(segment)][key][k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                else:
                    serializable_data[str(segment)][key] = str(value) if not isinstance(value, (int, float, str, bool, type(None))) else value
        
        with open(profile_json_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        print(f"Saved profile data to {profile_json_path}")
    except Exception as e:
        print(f"Error generating segment profiles: {str(e)}")
        print(f"Detailed error: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 3: Generate visualizations
    print("\n=== Step 3: Generating Visualizations ===")
    try:
        vis_output_dir = os.path.join(output_dir, 'visualizations')
        # Create visualizations directory if it doesn't exist
        Path(vis_output_dir).mkdir(parents=True, exist_ok=True)
        visualization_paths = visualize_segment_comparisons(profile_json_path, vis_output_dir)
        print(f"Successfully generated {len(visualization_paths)} visualizations:")
        for path in visualization_paths:
            print(f"  - {path}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        print(f"Detailed error: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("\n=== Analysis Complete ===")
    print(f"All results have been saved to {output_dir}")
    print("Summary:")
    print(f"  - Processed {processed_df.shape[0]} customer records")
    print(f"  - Generated profiles for {len(segments)} customer segments")
    print(f"  - Created {len(visualization_paths) if 'visualization_paths' in locals() else 0} visualization charts")
    
    # Optional: Display key insights from the segments
    print("\n=== Key Segment Insights ===")
    for segment in segments:
        profile = profile_data[segment]
        segment_size = profile['size']['count']
        segment_pct = profile['size']['percentage']
        conversion_rate = profile['value_metrics']['conversion_rate']
        
        # Get best channel and campaign
        best_channel = profile['channel_preferences']['best_channel']
        best_channel_conv = profile['channel_preferences']['best_channel_conversion']
        best_campaign = profile['campaign_preferences']['best_campaign']
        best_campaign_conv = profile['campaign_preferences']['best_campaign_conversion']
        
        print(f"Segment {segment} ({segment_pct} of customers):")
        print(f"  - Conversion Rate: {conversion_rate}")
        print(f"  - Best Channel: {best_channel} ({best_channel_conv})")
        print(f"  - Best Campaign: {best_campaign} ({best_campaign_conv})")
        
        # If there are top features, display them
        if profile['top_features'] and len(profile['top_features']) > 0:
            print("  - Key Distinctive Features:")
            for feature in profile['top_features'][:3]:  # Top 3 features only
                if isinstance(feature, dict):
                    feature_name = feature.get('feature', 'Unknown')
                    direction = feature.get('direction', 'different')
                    deviation = feature.get('deviation', '0')
                    
                    if isinstance(deviation, str):
                        try:
                            deviation = float(deviation.rstrip('%'))
                        except:
                            deviation = 0
                    
                    print(f"    * {feature_name}: {abs(deviation):.1f}% {direction} than average")
        print("")


if __name__ == "__main__":
    main()
