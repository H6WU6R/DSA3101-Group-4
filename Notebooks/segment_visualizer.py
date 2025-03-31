import re
import ast
import os
import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.patches import Patch

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
# Visualization subcomponents
###############################################################################

def create_header_section(fig, gs, segment, profile_data, main_color):
    """
    Draws the header text for a segment, including the segment name/ID
    and size (count + percentage).
    """
    ax_header = fig.add_subplot(gs)
    ax_header.axis('off')

    # Title
    ax_header.text(
        0.5, 0.7,
        f"Banking Segment {segment} Profile",
        fontsize=28, weight='bold', ha='center', color=main_color,
        transform=ax_header.transAxes
    )

    # Safer dictionary lookups
    size_info = profile_data.get('size', {})
    cust_count = size_info.get('count', 'Unknown')
    cust_pct = size_info.get('percentage', 'N/A')

    ax_header.text(
        0.5, 0.4,
        f"Size: {cust_count} customers ({cust_pct})",
        fontsize=18, ha='center', transform=ax_header.transAxes
    )
    return ax_header


def create_features_deviation_chart(fig, gs, features, deviations, positive_color, negative_color):
    """
    Plot horizontal bars showing how each feature deviates from
    the average (positive or negative).
    """
    ax_features = fig.add_subplot(gs)
    ax_features.axis('off')
    ax_features.text(
        0.5, 1.0,
        "Top Features Driving Conversion",
        fontsize=20, weight='bold', ha='center', transform=ax_features.transAxes
    )
    ax_features.text(
        0.5, 0.93,
        "Comparison to Average",
        fontsize=16, ha='center', transform=ax_features.transAxes
    )

    legend_elements = [
        Patch(facecolor=positive_color, label='Higher than average', alpha=0.8),
        Patch(facecolor=negative_color, label='Lower than average', alpha=0.8)
    ]

    if features and len(features) > 0:
        y_pos = np.arange(len(features))
        bar_height = 0.5
        bars = ax_features.barh(
            y_pos, deviations, height=bar_height,
            color=[positive_color if d > 0 else negative_color for d in deviations],
            alpha=0.8
        )

        # Manually add feature labels and numeric offsets
        xmin, xmax = ax_features.get_xlim()
        offset = 0.05 * (xmax - xmin)

        for i, (feature, deviation) in enumerate(zip(features, deviations)):
            ax_features.text(
                xmin - offset, i + bar_height / 2,
                feature, va='center', ha='right', fontsize=12, fontweight='bold'
            )
            # Place label on the side of the bar
            label_pos = deviation + (1 if deviation >= 0 else -1)
            ax_features.text(
                label_pos, i + bar_height / 2,
                f"{deviation:.1f}% vs. avg",
                va='center',
                ha='left' if deviation >= 0 else 'right',
                fontsize=11,
                fontweight='bold'
            )

        # Dynamic x-limits to show some space
        max_dev = max(abs(d) for d in deviations) * 1.2
        ax_features.set_xlim(-max_dev, max_dev)
        ax_features.set_ylim(-0.5, len(features) - 0.5)
        ax_features.set_yticks([])
        ax_features.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)

        ax_features.legend(
            handles=legend_elements, loc='lower center', fontsize=10,
            bbox_to_anchor=(0.5, -0.2), frameon=True, framealpha=0.9, edgecolor='lightgray'
        )
        ax_features.text(
            0.5, -0.3,
            "Shows how top conversion-driving features compare to average",
            fontsize=9, style='italic', ha='center', transform=ax_features.transAxes
        )
    else:
        ax_features.text(
            0.5, 0.5,
            "Insufficient data for feature importance",
            ha='center', fontsize=12, style='italic', transform=ax_features.transAxes
        )
        ax_features.legend(
            handles=legend_elements, loc='lower center', fontsize=10,
            bbox_to_anchor=(0.5, 0.2), frameon=True, framealpha=0.9, edgecolor='lightgray'
        )
    return ax_features


def create_radar_chart(fig, gs, profile_data, main_color, radar_max_val=10.0):
    """
    Creates a radar chart for a segment's engagement patterns.
    Normalizes each metric to [0,1] by dividing by `radar_max_val`.
    """
    ax_engagement = fig.add_subplot(gs, polar=True)
    engagement = profile_data.get('engagement_patterns', {})

    # Define the metrics to display on the radar, along with the keys in the data
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

    # Gather raw numeric values
    radar_values = []
    for metric in radar_metrics:
        key = radar_to_key.get(metric)
        val = extract_numeric(engagement.get(key, 0))
        radar_values.append(val)

    # Normalize each value
    normalized_values = [normalize_value(val, max_val=radar_max_val) for val in radar_values]

    # Close the polygon by repeating the first item
    radar_metrics = np.append(radar_metrics, radar_metrics[0])
    normalized_values = np.append(normalized_values, normalized_values[0])

    angles = np.linspace(0, 2*np.pi, len(radar_metrics) - 1, endpoint=False)
    angles = np.append(angles, angles[0])

    ax_engagement.plot(angles, normalized_values, 'o-', linewidth=2, color=main_color)
    ax_engagement.fill(angles, normalized_values, alpha=0.25, color=main_color)
    ax_engagement.set_thetagrids(angles * 180 / np.pi, radar_metrics, fontsize=9)
    ax_engagement.set_rlim(0, 1.05)
    ax_engagement.grid(True, alpha=0.3)

    # Optional radial lines for reference
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax_engagement.text(
            np.pi / 8, level, f"{level:.1f}",
            transform=ax_engagement.transData,
            ha='left', va='center', fontsize=8
        )

    ax_engagement.set_title("Digital Engagement Profile", fontsize=18, weight='bold', y=1.1)
    return ax_engagement


def create_value_metrics_table(fig, gs, profile_data):
    """
    Displays a small table of value metrics (conversion rate, purchases, loyalty points).
    """
    ax_value = fig.add_subplot(gs)
    ax_value.axis('off')

    vm = profile_data.get('value_metrics', {})
    conversion_rate = extract_numeric(vm.get('conversion_rate', '0%'))
    previous_purchases = extract_numeric(vm.get('previous_purchases', 0))
    loyalty_points = extract_numeric(vm.get('loyalty_points', 0))

    ax_value.text(
        0.5, 1.25,
        "Value Metrics", fontsize=18, weight='bold', ha='center',
        transform=ax_value.transAxes
    )

    value_data = [
        ['Metric', 'Value'],
        ['Conversion Rate', f"{conversion_rate:.1f}%"],
        ['Previous Purchases', f"{previous_purchases:.1f}"],
        ['Loyalty Points', f"{int(loyalty_points)}"]
    ]
    table = ax_value.table(
        cellText=value_data, loc='center', cellLoc='center',
        colWidths=[0.5, 0.3], bbox=[0.15, 0.05, 0.7, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5)

    # Header row styling
    for j in range(2):
        cell = table[(0, j)]
        cell.set_facecolor('#EBF1F6')
        cell.set_text_props(weight='bold')

    return ax_value


def create_preferences_section(fig, gs, profile_data, main_color):
    """
    Create a section showing channel and campaign preferences,
    with big “Channel Preferences” and “Campaign Preferences” titles.
    """
    ax_pref = fig.add_subplot(gs)
    ax_pref.axis('off')

    # Big Titles
    ax_pref.text(
        0.25, 0.95,
        "Channel Preferences", fontsize=18, weight='bold', ha='center',
        transform=ax_pref.transAxes
    )
    ax_pref.text(
        0.75, 0.95,
        "Campaign Preferences", fontsize=18, weight='bold', ha='center',
        transform=ax_pref.transAxes
    )

    # Channel preference info
    ch_pref = profile_data.get('channel_preferences', {})
    best_channel = ch_pref.get('best_channel', 'Unknown')
    best_channel_rate = extract_numeric(ch_pref.get('best_channel_conversion', '0%')) / 100.0
    channel_dist = {
        k: extract_numeric(v) for k, v in (ch_pref.get('distribution', {}) or {}).items()
    }

    # Campaign preference info
    camp_pref = profile_data.get('campaign_preferences', {})
    best_campaign = camp_pref.get('best_campaign', 'Unknown')
    best_campaign_rate = extract_numeric(camp_pref.get('best_campaign_conversion', '0%')) / 100.0
    campaign_dist = {
        k: extract_numeric(v) for k, v in (camp_pref.get('distribution', {}) or {}).items()
    }

    # Best channel details
    ax_pref.text(
        0.05, 0.85,
        "Best Channel:", fontsize=14, weight='bold', transform=ax_pref.transAxes
    )
    ax_pref.text(
        0.35, 0.85,
        best_channel, fontsize=14, color=main_color, weight='bold',
        transform=ax_pref.transAxes
    )
    ax_pref.text(
        0.35, 0.78,
        f"Conversion: {best_channel_rate*100:.1f}%", fontsize=12,
        transform=ax_pref.transAxes
    )

    # Best campaign details
    ax_pref.text(
        0.55, 0.85,
        "Best Campaign Type:", fontsize=14, weight='bold', transform=ax_pref.transAxes
    )
    ax_pref.text(
        0.85, 0.85,
        best_campaign, fontsize=14, color=main_color, weight='bold',
        transform=ax_pref.transAxes
    )
    ax_pref.text(
        0.85, 0.78,
        f"Conversion: {best_campaign_rate*100:.1f}%", fontsize=12,
        transform=ax_pref.transAxes
    )

    # Channel distribution mini-chart (top 3 bars)
    if channel_dist:
        sorted_channels = sorted(channel_dist.items(), key=lambda x: x[1], reverse=True)
        channels = [c[0] for c in sorted_channels]
        percentages = [c[1] for c in sorted_channels]

        left_space = 0.05
        chart_width = 0.4
        for i, (chan, pct) in enumerate(zip(channels, percentages)):
            if i < 3:
                bar_y = 0.65 - (i * 0.15)
                bar_length = pct / 100.0 * chart_width
                ax_pref.barh(
                    bar_y, bar_length, height=0.08, left=left_space,
                    color=main_color, alpha=max(0.2, 0.7 - i * 0.15)
                )
                ax_pref.text(
                    left_space - 0.01, bar_y + 0.04, f"{chan}",
                    ha='right', va='center', fontsize=10, transform=ax_pref.transAxes
                )
                ax_pref.text(
                    left_space + bar_length + 0.01, bar_y + 0.04, f"{pct:.1f}%",
                    ha='left', va='center', fontsize=10, transform=ax_pref.transAxes
                )

    # Campaign distribution mini-chart (top 3 bars)
    if campaign_dist:
        sorted_campaigns = sorted(campaign_dist.items(), key=lambda x: x[1], reverse=True)
        campaigns = [c[0] for c in sorted_campaigns]
        percentages = [c[1] for c in sorted_campaigns]

        right_space = 0.95
        chart_width = 0.4
        for i, (camp, pct) in enumerate(zip(campaigns, percentages)):
            if i < 3:
                bar_y = 0.65 - (i * 0.15)
                bar_length = pct / 100.0 * chart_width
                ax_pref.barh(
                    bar_y, bar_length, height=0.08, left=right_space - bar_length,
                    color=main_color, alpha=max(0.2, 0.7 - i * 0.15)
                )
                ax_pref.text(
                    right_space + 0.01, bar_y + 0.04, f"{camp}",
                    ha='left', va='center', fontsize=10, transform=ax_pref.transAxes
                )
                ax_pref.text(
                    right_space - bar_length - 0.01, bar_y + 0.04, f"{pct:.1f}%",
                    ha='right', va='center', fontsize=10, transform=ax_pref.transAxes
                )

    # Divider line
    ax_pref.axvline(x=0.5, ymin=0, ymax=0.9, color='lightgray', linewidth=1, alpha=0.7)
    return ax_pref


###############################################################################
# Main visualization functions
###############################################################################

def create_segment_profile_visualization(segment_id, profile_data, output_path):
    """
    Create a detailed visualization for a single segment with a figure size of (28, 24) inches.
    """
    colors = {
        'main': '#1E5C97',
        'secondary': '#78A2CC',
        'positive': '#4CAF50',
        'negative': '#E57373',
    }

    # Retrieve this segment's data
    segment_data = profile_data.get(segment_id, {})

    # Increase figure size for the individual segment visualization
    fig = plt.figure(figsize=(28, 24), constrained_layout=True)
    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[0.5, 1.8, 0.4, 1.6],
        width_ratios=[1, 1],
        hspace=0.5,
        wspace=0.3
    )
    fig.patch.set_edgecolor(colors['main'])
    fig.patch.set_linewidth(1)

    # 1. Header
    create_header_section(fig, gs[0, :], segment_id, segment_data, colors['main'])

    # 2. Parse top_features from string if needed
    #    e.g. "top_features": "[{'feature': 'X', 'deviation': 10}, ...]"
    top_feat = segment_data.get('top_features')
    if isinstance(top_feat, str):
        try:
            top_feat = ast.literal_eval(top_feat)
        except (SyntaxError, ValueError):
            top_feat = []
    elif not isinstance(top_feat, list):
        top_feat = []

    # Build feature and deviation lists
    features, deviations = [], []
    for item in top_feat:
        f = item.get('feature', 'Unnamed Feature')
        d = extract_numeric(item.get('deviation', 0))
        features.append(f)
        deviations.append(d)

    # 3. Features chart (left) and radar chart (right)
    create_features_deviation_chart(
        fig, gs[1, 0],
        features, deviations,
        colors['positive'], colors['negative']
    )
    create_radar_chart(
        fig, gs[1, 1],
        segment_data, colors['main'], radar_max_val=10.0
    )

    # 4. (Note: We skip any big middle row titles, as requested.)

    # 5. Bottom row: Value metrics (left), Preferences section (right)
    create_value_metrics_table(fig, gs[3, 0], segment_data)
    create_preferences_section(fig, gs[3, 1], segment_data, colors['main'])

    # Save & close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created visualization for Segment {segment_id} at: {output_path}")


def create_segment_comparison_dashboard(segments, profile_data, output_path):
    """
    Create a dashboard comparing key metrics across all segments.
    """
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Segment Comparison Dashboard', fontsize=22, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Conversion Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    conversion_values = [
        extract_numeric(profile_data.get(seg, {}).get('value_metrics', {}).get('conversion_rate', '0%'))
        for seg in segments
    ]
    bars = ax1.bar(segments, conversion_values, color='#3498db')
    ax1.set_title('Conversion Rate by Segment', fontsize=14)
    ax1.set_xlabel('Segment ID')
    ax1.set_ylabel('Conversion Rate (%)')
    if conversion_values:
        ax1.set_ylim(0, max(conversion_values) * 1.2)

    for bar, value in zip(bars, conversion_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height + 1,
            f'{value:.1f}%', ha='center', va='bottom', fontsize=10
        )

    # 2. Segment Size Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sizes = [
        extract_numeric(profile_data.get(seg, {}).get('size', {}).get('count', 0))
        for seg in segments
    ]
    wedges, texts, autotexts = ax2.pie(
        sizes, autopct='%1.1f%%', startangle=90, shadow=False
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_patch(centre_circle)
    ax2.axis('equal')
    ax2.set_title('Segment Size Distribution', fontsize=14)

    labels = [
        f'Segment {seg} ({int(sz)} customers)' for seg, sz in zip(segments, sizes)
    ]
    ax2.legend(wedges, labels, loc='center', bbox_to_anchor=(0.5, 0), fontsize=10)

    # 3. Best Channel by Segment
    ax3 = fig.add_subplot(gs[1, 0])
    channels = [
        profile_data.get(seg, {}).get('channel_preferences', {}).get('best_channel', 'Unknown')
        for seg in segments
    ]
    channel_conv_rates = [
        extract_numeric(
            profile_data.get(seg, {}).get('channel_preferences', {}).get('best_channel_conversion', '0%')
        )
        for seg in segments
    ]
    y_pos = np.arange(len(segments))
    ax3.barh(y_pos, channel_conv_rates, color='#2ecc71')
    ax3.set_yticks([])  # We’ll annotate manually

    for i, (seg, channel) in enumerate(zip(segments, channels)):
        ax3.text(-5, i, f"Segment {seg}", ha='right', va='center', fontsize=10, fontweight='bold')
        ax3.text(1, i, channel, va='center', fontsize=10)

    for i, rate in enumerate(channel_conv_rates):
        ax3.text(rate + 1, i, f"{rate:.1f}%", va='center', fontsize=9)

    ax3.set_xlabel('Conversion Rate (%)')
    ax3.set_title('Best Channel by Segment', fontsize=14)

    # 4. Digital Engagement Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ['website_visits', 'pages_per_visit', 'time_on_site']
    x = np.arange(len(segments))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [
            extract_numeric(
                profile_data.get(seg, {}).get('engagement_patterns', {}).get(metric, 0)
            )
            for seg in segments
        ]
        ax4.bar(
            x + (i - 1)*width, values, width,
            label=metric.replace('_', ' ').title()
        )

    ax4.set_title('Digital Engagement Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Segment {s}' for s in segments])
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created segment comparison dashboard at: {output_path}")


def visualize_all_segments(df=None, profile_json_path=None, output_dir='segment_visualizations'):
    """
    Load segment profile data (from a JSON or possibly a dataframe) and
    generate:
      1) A detailed profile visualization for each segment.
      2) A single comparison dashboard covering all segments.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if profile_json_path:
        with open(profile_json_path, 'r') as f:
            profile_data = json.load(f)
    else:
        raise NotImplementedError("Direct data processing is not implemented in this example.")

    segments = list(profile_data.keys())
    visualization_paths = []

    # Generate a visualization for each segment
    for segment_id in segments:
        output_path = os.path.join(output_dir, f'segment_{segment_id}_profile.png')
        create_segment_profile_visualization(segment_id, profile_data, output_path)
        visualization_paths.append(output_path)

    # Generate a comparison dashboard for all segments
    comparison_path = os.path.join(output_dir, 'segment_comparison_dashboard.png')
    create_segment_comparison_dashboard(segments, profile_data, comparison_path)
    visualization_paths.append(comparison_path)

    return visualization_paths


def main():
    """
    Command-line entry point.
    Example usage:
      python test.py --profile-json /path/to/profiles.json --output-dir out
    """
    PROFILE_JSON = '/Users/cindy/Desktop/DSA3101-Project-3/profiles.json'
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(PROFILE_JSON)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    parser = argparse.ArgumentParser(description='Generate visualizations for customer segments.')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to the input CSV file')
    parser.add_argument('--profile-json', type=str, default=PROFILE_JSON,
                        help='Path to a JSON file containing pre-generated profile data')
    parser.add_argument('--output-dir', type=str, default='segment_visualizations',
                        help='Directory where to save the visualizations')
    args = parser.parse_args()

    if args.input is None and args.profile_json is None:
        parser.error("Either --input or --profile-json must be provided")

    if args.profile_json is not None:
        print(f"Generating visualizations from profile data in {args.profile_json}...")
        visualization_paths = visualize_all_segments(
            profile_json_path=args.profile_json,
            output_dir=args.output_dir
        )
    else:
        # If you want to process raw data to produce the JSON on the fly:
        # from data_processing import behavioral_data_processing
        # print(f"Processing data from {args.input}...")
        # processed_df = behavioral_data_processing(args.input)
        # ...
        raise NotImplementedError("Data processing step is not implemented.")

    print(f"Created {len(visualization_paths)} visualizations in {args.output_dir}")


if __name__ == "__main__":
    main()
