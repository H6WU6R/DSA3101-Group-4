import re
import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.patches import Patch
import matplotlib.cm as cm

###############################################################################
# Global path to your profiles.json
###############################################################################
PROFILE_JSON = '/Users/cindy/Desktop/DSA3101-Project-3/profiles.json'

###############################################################################
# Utility functions
###############################################################################

def extract_numeric(value):
    """
    Extract numeric value from a string (e.g., "12.5%" -> 12.5).
    If parsing fails or value is not numeric, returns 0.0.
    """
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        match = re.search(r'([-+]?\d*\.\d+|\d+)', value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
    return 0.0


def normalize_value(value, max_val=10.0):
    """
    Normalize a numeric value by dividing by `max_val`, clamped to [0, 1].
    If max_val <= 0, returns 0 to avoid dividing by zero.
    """
    if max_val <= 0:
        return 0.0
    ratio = value / max_val
    return max(0.0, min(ratio, 1.0))


###############################################################################
# Multi-segment comparison plots
###############################################################################

def create_features_comparison_chart(segments, profile_data, output_path):
    """
    Create separate charts for each segment showing its top features.
    """
    # Set up the figure with subplots for each segment
    fig = plt.figure(figsize=(18, 14))
    
    # Calculate rows and columns for the grid
    n_segments = len(segments)
    n_cols = min(3, n_segments)
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    # Create a colormap for positive and negative values
    positive_color = '#4CAF50'  # Green
    negative_color = '#E57373'  # Red
    
    # Create a subplot for each segment
    for idx, segment in enumerate(segments):
        segment_data = profile_data.get(segment, {})
        
        # Parse top_features from string if needed
        top_feat = segment_data.get('top_features')
        if isinstance(top_feat, str):
            try:
                top_feat = ast.literal_eval(top_feat)
            except (SyntaxError, ValueError):
                top_feat = []
        elif not isinstance(top_feat, list):
            top_feat = []
        
        # Build feature and deviation lists
        features = []
        deviations = []
        directions = []
        
        for item in top_feat:
            feature = item.get('feature', 'Unnamed')
            deviation = extract_numeric(item.get('deviation', 0))
            direction = item.get('direction', '')
            
            features.append(feature)
            deviations.append(deviation)
            directions.append(direction)
        
        # Create subplot
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        # Set title
        ax.set_title(f"Segment {segment} - Top Features", fontsize=14, fontweight='bold')
        
        # Plot horizontal bars if we have data
        if features and deviations:
            # Sort by absolute deviation for better visualization
            sorted_indices = sorted(range(len(deviations)), key=lambda i: abs(deviations[i]), reverse=True)
            features = [features[i] for i in sorted_indices]
            deviations = [deviations[i] for i in sorted_indices]
            directions = [directions[i] for i in sorted_indices]
            
            # Create horizontal bars
            y_pos = np.arange(len(features))
            bars = ax.barh(
                y_pos, 
                deviations, 
                color=[positive_color if d > 0 else negative_color for d in deviations],
                alpha=0.8
            )
            
            # Add feature labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            
            # Add deviation values and direction
            for i, (deviation, direction) in enumerate(zip(deviations, directions)):
                label_pos = deviation + (0.5 if deviation >= 0 else -0.5)
                ha = 'left' if deviation >= 0 else 'right'
                
                ax.text(
                    label_pos, i,
                    f"{deviation:.1f}% ({direction})",
                    va='center',
                    ha=ha,
                    fontsize=9,
                    fontweight='bold'
                )
            
            # Add a vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add legend
            legend_elements = [
                Patch(facecolor=positive_color, label='Higher than average', alpha=0.8),
                Patch(facecolor=negative_color, label='Lower than average', alpha=0.8)
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, "No feature data available", ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created features comparison chart at: {output_path}")


def create_radar_comparison_chart(segments, profile_data, output_path):
    """
    Create a radar chart comparing engagement metrics across all segments.
    """
    # Define the metrics to display
    radar_metrics = [
        'Website Visits',
        'Pages/Visit',
        'Time on Site',
        'Email CTR',
        'Engagement Depth',
        'Social Propensity'
    ]
    radar_to_key = {
        'Website Visits': 'website_visits',
        'Pages/Visit': 'pages_per_visit',
        'Time on Site': 'time_on_site',
        'Email CTR': 'email_ctr',
        'Engagement Depth': 'engagement_depth',
        'Social Propensity': 'social_propensity'
    }
    
    # Set up the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Create a colormap for segments
    colors = cm.tab10(np.linspace(0, 1, len(segments)))
    segment_colors = {seg: color for seg, color in zip(segments, colors)}
    
    # Calculate angles for the radar chart
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
    # Close the polygon by repeating the first angle
    angles = np.append(angles, angles[0])
    
    # Determine max values for normalization
    max_values = {}
    for metric in radar_metrics:
        key = radar_to_key.get(metric)
        values = [extract_numeric(profile_data.get(seg, {}).get('engagement_patterns', {}).get(key, 0))
                 for seg in segments]
        max_values[metric] = max(values) * 1.1  # Add 10% for scaling
    
    # Plot each segment
    for i, segment in enumerate(segments):
        segment_data = profile_data.get(segment, {})
        engagement = segment_data.get('engagement_patterns', {})
        
        # Gather values for this segment
        values = []
        for metric in radar_metrics:
            key = radar_to_key.get(metric)
            val = extract_numeric(engagement.get(key, 0))
            # Normalize by the maximum across all segments
            norm_val = normalize_value(val, max_val=max_values[metric])
            values.append(norm_val)
        
        # Close the polygon by repeating the first value
        values = np.append(values, values[0])
        
        # Plot the segment's polygon
        ax.plot(angles, values, 'o-', linewidth=2, color=segment_colors[segment], 
                label=f'Segment {segment}', alpha=0.8)
        ax.fill(angles, values, color=segment_colors[segment], alpha=0.1)
    
    # Customize the chart
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, radar_metrics, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Add title and legend
    plt.title('Engagement Metrics Comparison Across Segments', fontsize=18, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add data range information
    ranges_text = "Metrics ranges across segments:\n"
    for metric in radar_metrics:
        key = radar_to_key.get(metric)
        values = [extract_numeric(profile_data.get(seg, {}).get('engagement_patterns', {}).get(key, 0))
                 for seg in segments]
        min_val = min(values)
        max_val = max(values)
        ranges_text += f"{metric}: {min_val:.1f} - {max_val:.1f}\n"
    
    plt.figtext(1.0, 0.5, ranges_text, fontsize=9, ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created radar comparison chart at: {output_path}")


def create_value_metrics_comparison(segments, profile_data, output_path):
    """
    Create a table comparing value metrics across all segments.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Define the metrics to compare
    metrics = ['Conversion Rate (%)', 'Previous Purchases', 'Loyalty Points']
    
    # Collect data for each segment
    data = []
    cell_colors = []
    
    # Header row
    header = ['Metric'] + [f'Segment {s}' for s in segments]
    data.append(header)
    cell_colors.append(['#f0f0f0'] + ['#f0f0f0'] * len(segments))
    
    # Create a colormap for value ranges
    cmap_green = plt.cm.Greens
    cmap_red = plt.cm.Reds_r  # Reversed Reds colormap
    
    # Add rows for each metric
    for metric_idx, metric in enumerate(metrics):
        row = [metric]
        row_colors = ['#f0f0f0']  # Header column
        
        values = []
        for segment in segments:
            vm = profile_data.get(segment, {}).get('value_metrics', {})
            
            if metric == 'Conversion Rate (%)':
                val = extract_numeric(vm.get('conversion_rate', '0%'))
                formatted_val = f"{val:.1f}%"
                # Color based on conversion rate (higher is better)
                color_val = min(1.0, val / 100)
                color = cmap_green(color_val)
            elif metric == 'Previous Purchases':
                val = extract_numeric(vm.get('previous_purchases', 0))
                formatted_val = f"{val:.1f}"
                # Color based on purchases (higher is better)
                color_val = min(1.0, val / 5)  # Assuming 5 is max
                color = cmap_green(color_val)
            else:  # Loyalty Points
                val = extract_numeric(vm.get('loyalty_points', 0))
                formatted_val = f"{int(val)}"
                # Color based on loyalty points (higher is better)
                color_val = min(1.0, val / 3000)  # Assuming 3000 is max
                color = cmap_green(color_val)
            
            values.append(val)
            row.append(formatted_val)
            row_colors.append(color)
        
        data.append(row)
        cell_colors.append(row_colors)
    
    # Create the table
    table = ax.table(
        cellText=data,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Set column widths
    for (i, j), cell in table.get_celld().items():
        if j == 0:  # Metric name column
            cell.set_width(0.2)
        else:
            cell.set_width(0.8 / len(segments))
            
        # Add borders
        cell.set_edgecolor('white')
        
        # Make header row bold
        if i == 0 or j == 0:
            cell.get_text().set_fontweight('bold')
    
    # Add title
    ax.set_title('Value Metrics Comparison Across Segments', fontsize=18, pad=20)
    
    # Add color legend
    plt.figtext(0.5, 0.05, 
               "Color intensity indicates value (darker green = higher value)",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created value metrics comparison at: {output_path}")


def create_preferences_comparison(segments, profile_data, output_path):
    """
    Create separate visualizations for each segment's channel and campaign preferences.
    Uses a single set of labels at the bottom of the plot instead of individual legends.
    Ensures all segment labels are positioned correctly.
    """
    # Set up the figure with subplots for each segment
    fig = plt.figure(figsize=(18, 14))
    
    # Calculate rows and columns for the grid
    n_segments = len(segments)
    n_cols = min(3, n_segments)
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    # Create color maps for consistent colors across all segments
    # For channels
    channel_types = set()
    # For campaigns
    campaign_types = set()
    
    # First, collect all unique channel and campaign types
    for segment in segments:
        segment_data = profile_data.get(segment, {})
        
        # Channel distribution
        ch_dist = segment_data.get('channel_preferences', {}).get('distribution', {})
        for channel in ch_dist.keys():
            channel_types.add(channel)
            
        # Campaign distribution
        camp_dist = segment_data.get('campaign_preferences', {}).get('distribution', {})
        for campaign in camp_dist.keys():
            campaign_types.add(campaign)
    
    # Convert to sorted lists for consistent coloring
    channel_types = sorted(list(channel_types))
    campaign_types = sorted(list(campaign_types))
    
    # Create color dictionaries
    ch_colors = dict(zip(channel_types, cm.tab10(np.linspace(0, 1, len(channel_types)))))
    camp_colors = dict(zip(campaign_types, cm.Paired(np.linspace(0, 1, len(campaign_types)))))
    
    # Create a subplot for each segment
    for idx, segment in enumerate(segments):
        segment_data = profile_data.get(segment, {})
        
        # Channel preference data
        ch_pref = segment_data.get('channel_preferences', {})
        best_channel = ch_pref.get('best_channel', 'Unknown')
        best_channel_conv = extract_numeric(ch_pref.get('best_channel_conversion', '0%'))
        channel_dist = ch_pref.get('distribution', {})
        
        # Campaign preference data
        camp_pref = segment_data.get('campaign_preferences', {})
        best_campaign = camp_pref.get('best_campaign', 'Unknown')
        best_campaign_conv = extract_numeric(camp_pref.get('best_campaign_conversion', '0%'))
        campaign_dist = camp_pref.get('distribution', {})
        
        # Calculate the position for this segment's subplots
        row = idx // n_cols
        col = idx % n_cols
        
        # Create two axes for channel and campaign
        ax1 = plt.subplot(n_rows, n_cols * 2, idx * 2 + 1)
        ax2 = plt.subplot(n_rows, n_cols * 2, idx * 2 + 2)
        
        # Add segment title at the top of each column
        # Only add segment labels in the top row
        if row == 0:
            plt.figtext((0.5 + col) / n_cols, 0.95, 
                        f"Segment {segment}", 
                        ha='center', va='center', 
                        fontsize=16, fontweight='bold')
        
        # 1. Channel Distribution Pie
        if channel_dist:
            labels = []
            sizes = []
            colors = []
            
            # Process channel data
            for channel_type in channel_types:
                if channel_type in channel_dist:
                    labels.append(channel_type)
                    sizes.append(extract_numeric(channel_dist[channel_type]))
                    colors.append(ch_colors[channel_type])
            
            # Create the pie chart
            ax1.pie(
                sizes, 
                labels=None,
                colors=colors, 
                autopct='%1.1f%%',
                startangle=90, 
                wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
            )
            ax1.set_title(f"Channel Distribution\nBest: {best_channel} ({best_channel_conv:.1f}%)", fontsize=12)
            
            # Add segment label for 2nd and 3rd row
            if row > 0:
                # Add segment labels correctly positioned above each group of two charts
                plt.figtext((0.5 + col) / n_cols, 
                           0.95 - 0.45 * row,  # Adjust this value to position correctly for each row
                           f"Segment {segment}", 
                           ha='center', va='center',
                           fontsize=16, fontweight='bold')
            
        else:
            ax1.text(0.5, 0.5, "No channel data available", ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Campaign Distribution Pie
        if campaign_dist:
            labels = []
            sizes = []
            colors = []
            
            # Process campaign data
            for campaign_type in campaign_types:
                if campaign_type in campaign_dist:
                    labels.append(campaign_type)
                    sizes.append(extract_numeric(campaign_dist[campaign_type]))
                    colors.append(camp_colors[campaign_type])
            
            # Create the pie chart
            ax2.pie(
                sizes, 
                labels=None,
                colors=colors, 
                autopct='%1.1f%%',
                startangle=90, 
                wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
            )
            ax2.set_title(f"Campaign Distribution\nBest: {best_campaign} ({best_campaign_conv:.1f}%)", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "No campaign data available", ha='center', va='center', transform=ax2.transAxes)
    
    # Add a shared legend at the bottom of the plot for channels
    channel_legend_elements = [Patch(facecolor=ch_colors[ch], label=ch) for ch in channel_types]
    fig.legend(
        handles=channel_legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.25, 0.02),
        ncol=len(channel_types),
        fontsize=10,
        title="Channel Types",
        title_fontsize=12
    )
    
    # Add a shared legend at the bottom of the plot for campaigns
    campaign_legend_elements = [Patch(facecolor=camp_colors[camp], label=camp) for camp in campaign_types]
    fig.legend(
        handles=campaign_legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.75, 0.02),
        ncol=len(campaign_types),
        fontsize=10,
        title="Campaign Types",
        title_fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])  # Make room for the bottom legends and top titles
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created preferences comparison chart at: {output_path}")

def create_segment_size_distribution(segments, profile_data, output_path):
    """
    Create a pie chart showing the size distribution of segments.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect segment sizes
    sizes = []
    percentages = []
    segment_labels = []
    
    for segment in segments:
        segment_data = profile_data.get(segment, {})
        size_info = segment_data.get('size', {})
        
        count = extract_numeric(size_info.get('count', 0))
        percentage = extract_numeric(size_info.get('percentage', '0%'))
        
        sizes.append(count)
        percentages.append(percentage)
        segment_labels.append(f"Segment {segment} ({int(count)} customers)")
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Make it a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_patch(centre_circle)
    
    # Customize the chart
    ax.set_title('Segment Size Distribution', fontsize=20, pad=20)
    ax.legend(wedges, segment_labels, loc='center', bbox_to_anchor=(0.5, -0.1), fontsize=12)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created segment size distribution chart at: {output_path}")


###############################################################################
# Main function
###############################################################################

def visualize_segment_comparisons(profile_json_path=PROFILE_JSON, output_dir='segment_comparisons'):
    """
    Generate comparison visualizations:
    1. Segment size distribution (pie chart)
    2. Features comparison (individual charts for each segment)
    3. Radar chart engagement comparison (all segments in one chart)
    4. Value metrics comparison table
    5. Channel and campaign preferences (individual charts for each segment)
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the JSON profiles
    with open(profile_json_path, 'r') as f:
        profile_data = json.load(f)
    
    # Get all segment IDs
    segments = list(profile_data.keys())
    
    # 1. Segment Size Distribution
    size_path = os.path.join(output_dir, 'segment_size_distribution.png')
    create_segment_size_distribution(segments, profile_data, size_path)
    
    # 2. Features Comparison Chart (separate for each segment)
    features_path = os.path.join(output_dir, 'features_comparison.png')
    create_features_comparison_chart(segments, profile_data, features_path)
    
    # 3. Radar Chart Comparison (all segments in one chart)
    radar_path = os.path.join(output_dir, 'engagement_radar_comparison.png')
    create_radar_comparison_chart(segments, profile_data, radar_path)
    
    # 4. Value Metrics Comparison
    metrics_path = os.path.join(output_dir, 'value_metrics_comparison.png')
    create_value_metrics_comparison(segments, profile_data, metrics_path)
    
    # 5. Preferences Comparison (separate for each segment)
    preferences_path = os.path.join(output_dir, 'preferences_comparison.png')
    create_preferences_comparison(segments, profile_data, preferences_path)
    
    print(f"\nAll segment comparison visualizations have been generated in '{output_dir}'.")
    return [size_path, features_path, radar_path, metrics_path, preferences_path]


def main():
    """
    Entry point for the script. Uses the hardcoded PROFILE_JSON path.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate segment comparison visualizations.')
    parser.add_argument('--profile-json', type=str, default=PROFILE_JSON,
                        help=f'Path to the JSON file with segment profiles (default: {PROFILE_JSON})')
    parser.add_argument('--output-dir', type=str, default='segment_comparisons',
                        help='Directory where to save the visualizations')
    
    args = parser.parse_args()
    
    visualize_segment_comparisons(args.profile_json, args.output_dir)


if __name__ == "__main__":
    main()