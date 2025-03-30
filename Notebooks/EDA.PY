import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.table import Table

# 1. Initial Data Exploration and Segment Profiling
# -------------------------------------------------

# Load the dataset
df = pd.read_csv("../Data/segments0324.csv")

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nData types:")
print(df.dtypes)

print("\nCheck for missing values:")
print(df.isnull().sum())

# Display first few rows
print("\nSample data:")
df.head()

# Basic segment profile
segment_sizes = df['Cluster_Label'].value_counts().sort_index()
print("\nSegment sizes:")
print(segment_sizes)

# Visualize segment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster_Label', data=df)
plt.title('Distribution of Customers Across Segments')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.grid(axis='y')
plt.tight_layout()
# plt.show()


# Define key behavioral metrics to analyze
digital_metrics = ['WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks']
transaction_metrics = ['PreviousPurchases', 'LoyaltyPoints']
campaign_metrics = ['ClickThroughRate', 'ConversionRate', 'AdSpend']
all_metrics = digital_metrics + transaction_metrics + campaign_metrics

# Calculate segment profiles with key metrics
segment_profiles = {}
segments = sorted(df['Cluster_Label'].unique())

for segment in segments:
    # Filter data for this segment
    segment_data = df[df['Cluster_Label'] == segment]
    segment_size = len(segment_data)
    
    # Initialize profile object
    segment_profiles[segment] = {
        'size': segment_size,
        'size_percentage': segment_size / len(df) * 100,
        'digital_engagement': {},
        'transaction_history': {},
        'campaign_performance': {},
        'channel_preferences': {},
        'campaign_type_preferences': {}
    }
    
    # Digital engagement metrics
    for metric in digital_metrics:
        segment_profiles[segment]['digital_engagement'][metric] = segment_data[metric].mean()
    
    # Transaction history
    for metric in transaction_metrics:
        segment_profiles[segment]['transaction_history'][metric] = segment_data[metric].mean()
    
    # Campaign performance
    for metric in campaign_metrics:
        segment_profiles[segment]['campaign_performance'][metric] = segment_data[metric].mean()
    
    # Conversion rate
    segment_profiles[segment]['campaign_performance']['conversion_count'] = segment_data['Conversion'].sum()
    segment_profiles[segment]['campaign_performance']['conversion_rate'] = segment_data['Conversion'].mean()
    
    # Channel preferences
    channel_counts = segment_data['CampaignChannel'].value_counts(normalize=True) * 100
    channel_conversion = {}
    
    for channel in segment_data['CampaignChannel'].unique():
        channel_data = segment_data[segment_data['CampaignChannel'] == channel]
        channel_conversion[channel] = channel_data['Conversion'].mean()
    
    segment_profiles[segment]['channel_preferences'] = {
        'distribution': channel_counts.to_dict(),
        'conversion_rate': channel_conversion
    }
    
    # Campaign type preferences
    campaign_counts = segment_data['CampaignType'].value_counts(normalize=True) * 100
    campaign_conversion = {}
    
    for campaign_type in segment_data['CampaignType'].unique():
        campaign_data = segment_data[segment_data['CampaignType'] == campaign_type]
        campaign_conversion[campaign_type] = campaign_data['Conversion'].mean()
    
    segment_profiles[segment]['campaign_type_preferences'] = {
        'distribution': campaign_counts.to_dict(),
        'conversion_rate': campaign_conversion
    }

# Calculate additional behavioral metrics
for segment in segments:
    profile = segment_profiles[segment]
    
    # Email engagement effectiveness
    if profile['digital_engagement']['EmailOpens'] > 0:
        profile['digital_engagement']['email_ctr'] = (
            profile['digital_engagement']['EmailClicks'] / 
            profile['digital_engagement']['EmailOpens']
        )
    else:
        profile['digital_engagement']['email_ctr'] = 0
    
    # Website engagement depth
    profile['digital_engagement']['engagement_depth'] = (
        profile['digital_engagement']['PagesPerVisit'] * 
        profile['digital_engagement']['TimeOnSite']
    )
    
    # Social sharing propensity
    if profile['digital_engagement']['WebsiteVisits'] > 0:
        profile['digital_engagement']['social_propensity'] = (
            profile['digital_engagement']['SocialShares'] / 
            profile['digital_engagement']['WebsiteVisits']
        )
    else:
        profile['digital_engagement']['social_propensity'] = 0

# Show an overview of segment profiles
segment_conversion_rates = df.groupby('Cluster_Label')['Conversion'].mean() * 100

segment_overview = pd.DataFrame({
    'Segment': segments,
    'Size': [segment_profiles[s]['size'] for s in segments],
    'Size %': [f"{segment_profiles[s]['size_percentage']:.1f}%" for s in segments],
    'Conversion Rate': [f"{segment_conversion_rates[s]:.1f}%" for s in segments],
    'Website Visits': [f"{segment_profiles[s]['digital_engagement']['WebsiteVisits']:.1f}" for s in segments],
    'Email CTR': [f"{segment_profiles[s]['digital_engagement']['email_ctr']*100:.1f}%" for s in segments],
    'Engagement Depth': [f"{segment_profiles[s]['digital_engagement']['engagement_depth']:.1f}" for s in segments],
    'Previous Purchases': [f"{segment_profiles[s]['transaction_history']['PreviousPurchases']:.1f}" for s in segments]
})

print("\nSegment Overview:")
print(segment_overview)


# 2. Behavioral Pattern Identification
# -----------------------------------

# Create function to visualize segment differences for a specific metric
def visualize_segment_differences(df, metric, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster_Label', y=metric, data=df)
    plt.title(title)
    plt.xlabel('Segment')
    plt.ylabel(metric)
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.show()

# Analyze key behavioral metrics
for metric in ['WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'EmailOpens', 'EmailClicks']:
    visualize_segment_differences(df, metric, f'{metric} by Customer Segment')

# Calculate channel effectiveness by segment
channel_effectiveness = pd.DataFrame(columns=['Segment', 'Channel', 'Conversion_Rate'])

for segment in segments:
    segment_data = df[df['Cluster_Label'] == segment]
    
    for channel in segment_data['CampaignChannel'].unique():
        channel_data = segment_data[segment_data['CampaignChannel'] == channel]
        conversion_rate = channel_data['Conversion'].mean()
        
        channel_effectiveness = pd.concat([
            channel_effectiveness, 
            pd.DataFrame({
                'Segment': [segment],
                'Channel': [channel],
                'Conversion_Rate': [conversion_rate]
            })
        ], ignore_index=True)

# Visualize channel effectiveness
plt.figure(figsize=(12, 8))
channel_pivot = channel_effectiveness.pivot(index='Channel', columns='Segment', values='Conversion_Rate')
sns.heatmap(channel_pivot, annot=True, fmt='.1%', cmap='YlGnBu')
plt.title('Channel Effectiveness (Conversion Rate) by Segment')
plt.tight_layout()
# plt.show()

# Calculate campaign type effectiveness by segment
campaign_effectiveness = pd.DataFrame(columns=['Segment', 'Campaign_Type', 'Conversion_Rate'])

for segment in segments:
    segment_data = df[df['Cluster_Label'] == segment]
    
    for campaign_type in segment_data['CampaignType'].unique():
        campaign_data = segment_data[segment_data['CampaignType'] == campaign_type]
        conversion_rate = campaign_data['Conversion'].mean()
        
        campaign_effectiveness = pd.concat([
            campaign_effectiveness, 
            pd.DataFrame({
                'Segment': [segment],
                'Campaign_Type': [campaign_type],
                'Conversion_Rate': [conversion_rate]
            })
        ], ignore_index=True)

# Visualize campaign type effectiveness
plt.figure(figsize=(12, 8))
campaign_pivot = campaign_effectiveness.pivot(index='Campaign_Type', columns='Segment', values='Conversion_Rate')
sns.heatmap(campaign_pivot, annot=True, fmt='.1%', cmap='YlGnBu')
plt.title('Campaign Type Effectiveness (Conversion Rate) by Segment')
plt.tight_layout()
# plt.show()


# 3. Segment Differentiation Using Feature Importance for Conversion
# -----------------------------------------------------------------

# 3. Segment Differentiation Using Feature Importance for Conversion
# -----------------------------------------------------------------

# Prepare data for feature importance analysis
X = df[all_metrics].copy()

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

def segment_conversion_feature_importance(segment):
    """
    Calculate feature importance for conversion prediction within a specific segment.
    Handles segments with 100% conversion rates.
    
    Parameters:
    -----------
    segment : int
        The segment ID to analyze
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with features and their importance scores
    """
    # Filter data for just this segment
    segment_data = df[df['Cluster_Label'] == segment]
    
    # Get features and target for this segment
    X_segment = X_scaled.iloc[segment_data.index]
    y_segment = segment_data['Conversion']
    
    # Check if we have a 100% conversion rate
    if y_segment.mean() == 1.0:
        print(f"Segment {segment} has 100% conversion rate. Using alternative feature analysis.")
        
        # Method 1: Compare to segments with lower conversion
        # Find most distinctive features compared to other segments
        all_other_segments = df[df['Cluster_Label'] != segment]
        
        # Calculate mean differences for each feature
        feature_diff = {}
        for feature in X.columns:
            segment_mean = segment_data[feature].mean()
            others_mean = all_other_segments[feature].mean()
            
            # Calculate percent difference
            if others_mean != 0:
                pct_diff = (segment_mean - others_mean) / others_mean * 100
                feature_diff[feature] = abs(pct_diff)  # Use absolute difference for ranking
        
        # Sort features by their distinctive value
        sorted_features = sorted(feature_diff.items(), key=lambda x: x[1], reverse=True)
        
        # Create DataFrame in same format as feature importance
        feature_imp = pd.DataFrame({
            'Feature': [f[0] for f in sorted_features],
            'Importance': [f[1]/100 for f in sorted_features]  # Normalize to similar scale as feature importance
        })
        
        return feature_imp
    
    # For segments with regular conversion patterns, use the normal approach
    # Train a random forest classifier to predict conversion within this segment
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Check if we have enough data and both classes represented
    if len(y_segment) < 10 or y_segment.nunique() < 2:
        return pd.DataFrame({
            'Feature': ['Insufficient data'],
            'Importance': [0]
        })
    
    rf.fit(X_segment, y_segment)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create a dataframe with feature importances
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_imp




# Calculate conversion feature importance for each segment
segment_conversion_features = {}
for segment in segments:
    segment_conversion_features[segment] = segment_conversion_feature_importance(segment)
    
    print(f"\nTop 5 features driving conversion in Segment {segment}:")
    print(segment_conversion_features[segment].head(5))


# Analyze how each segment differs from the average for key metrics
# Create a DataFrame with only numeric metrics
numeric_metrics = []
for metric in all_metrics:
    if df[metric].dtype.kind in 'ifc':  # integer, float, or complex
        numeric_metrics.append(metric)

segment_deviations = pd.DataFrame(index=segments, columns=numeric_metrics)

for metric in numeric_metrics:
    overall_avg = df[metric].mean()
    
    for segment in segments:
        segment_avg = df[df['Cluster_Label'] == segment][metric].mean()
        # Avoid division by zero
        if overall_avg != 0:
            percent_diff = (segment_avg - overall_avg) / overall_avg * 100
        else:
            percent_diff = 0 if segment_avg == 0 else 100  # handle division by zero
        segment_deviations.loc[segment, metric] = percent_diff

# Convert all values to float to ensure they're numeric
segment_deviations = segment_deviations.astype(float)

# Visualize segment deviations from average
plt.figure(figsize=(15, 10))
sns.heatmap(segment_deviations, annot=True, fmt='.1f', cmap='RdBu_r', center=0)
plt.title('Percentage Deviation from Average for Key Metrics by Segment')
plt.tight_layout()
# plt.show()


# Combine feature importance with segment deviations to create important traits
important_traits = {}

for segment in segments:
    # Get the top 5 important features for this segment
    segment_features = segment_conversion_features[segment]
    
    # Initialize traits list
    traits = []
    
    # Check if we have valid feature importance data
    if 'Insufficient data' not in segment_features['Feature'].values:
        # Get top 5 important features
        top_features = segment_features.head(5)
        
        for _, row in top_features.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            # Check if feature is in our numeric metrics
            if feature in segment_deviations.columns:
                deviation = segment_deviations.loc[segment, feature]
                direction = 'higher' if deviation > 0 else 'lower'
                
                traits.append({
                    'feature': feature,
                    'importance': importance,
                    'deviation': deviation,
                    'direction': direction,
                    'description': f"{abs(deviation):.1f}% {direction} than average"
                })
    
    # Store traits for this segment
    important_traits[segment] = traits

# Determine the best channel and campaign type for each segment
best_channels = {}
best_campaign_types = {}

for segment in segments:
    # Find best channel
    channel_conv = segment_profiles[segment]['channel_preferences']['conversion_rate']
    best_channel = max(channel_conv.items(), key=lambda x: x[1])
    best_channels[segment] = best_channel
    
    # Find best campaign type
    campaign_conv = segment_profiles[segment]['campaign_type_preferences']['conversion_rate']
    best_campaign = max(campaign_conv.items(), key=lambda x: x[1])
    best_campaign_types[segment] = best_campaign

# Function to visualize distinctive traits for a segment
def visualize_distinctive_traits(segment):
    """
    Create a horizontal bar chart of distinctive traits for a segment based on top features.
    
    Parameters:
    -----------
    segment : int
        Segment ID to visualize
    """
    # Get traits for this segment
    traits = important_traits[segment]
    
    # Check if we have traits to visualize
    if not traits:
        print(f"No distinctive traits available for Segment {segment}")
        return
    
    # Extract data for visualization
    features = [t['feature'] for t in traits]
    deviations = [t['deviation'] for t in traits]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bars with appropriate colors
    colors = ['#4CAF50' if dev > 0 else '#E57373' for dev in deviations]
    y_pos = range(len(features))
    
    # Create horizontal bars
    bars = plt.barh(y_pos, deviations, color=colors)
    
    # Add percentage labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_pos = width if width >= 0 else width - 2
        plt.text(label_pos, bar.get_y() + bar.get_height()/2, 
                f"{deviations[i]:.1f}%", 
                ha='left' if width >= 0 else 'right', 
                va='center',
                fontsize=10,
                fontweight='bold')
    
    # Set labels and title
    plt.yticks(y_pos, features)
    plt.xlabel('% Difference from Average')
    plt.title(f'Distinctive Traits (% vs. Average) - Segment {segment}')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Higher than average'),
        Patch(facecolor='#E57373', label='Lower than average')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()  # Return the figure for saving if needed

# Example: Visualize distinctive traits for segment 0
visualize_distinctive_traits(1)

# Function to create behavioral profile cards (without recommendations section)
def create_behavioral_profile_cards(df, segment_profiles, segment_conversion_features, important_traits, 
                                   best_channels, best_campaign_types, output_dir="profile_cards"):
    """
    Create comprehensive behavioral profile cards for each banking segment
    
    Parameters:
    -----------
    df : DataFrame
        The main dataset with customer information
    segment_profiles : dict
        Dictionary with segment profiles from previous analysis
    segment_conversion_features : dict
        Dictionary with feature importance for conversion by segment
    important_traits : dict
        Dictionary with important traits for each segment
    best_channels : dict
        Dictionary with best channel for each segment
    best_campaign_types : dict
        Dictionary with best campaign type for each segment
    output_dir : str, optional
        Directory to save the profile cards
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate overall metrics for comparison
    segments = sorted(df['Cluster_Label'].unique())
    segment_conversion_rates = df.groupby('Cluster_Label')['Conversion'].mean() * 100
    
    # Create behavioral profile cards for each segment
    profile_cards = {}
    
    for segment in segments:
        # Create a figure for this segment's profile card
        fig = plt.figure(figsize=(12, 12))  # Reduced height since we removed recommendations
        gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1.5, 1.5, 1.5])  # Reduced to 4 rows
        
        # Get segment data
        profile = segment_profiles[segment]
        segment_size = profile['size']
        segment_size_pct = profile['size_percentage']
        segment_conversion_rate = segment_conversion_rates[segment]
        
        # Define a color scheme for banking
        main_color = '#1E5C97'  # Banking blue
        secondary_color = '#78A2CC'  # Lighter blue
        
        #----- HEADER SECTION -----#
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        ax_header.text(0.5, 0.7, f"Banking Segment {segment} Profile", 
                      fontsize=22, weight='bold', ha='center', color=main_color)
        ax_header.text(0.5, 0.3, 
                      f"Size: {segment_size} customers ({segment_size_pct:.1f}%)", 
                      fontsize=14, ha='center')
        
        #----- DISTINCTIVE TRAITS SECTION -----#
        ax_traits = fig.add_subplot(gs[1, 0])
        ax_traits.axis('off')
        ax_traits.text(0, 1, "Distinctive Traits (% vs. Average)", fontsize=16, weight='bold')
        
        # Get traits for this segment
        traits = important_traits[segment]
        
        if traits:
            # Extract data for visualization
            features = [t['feature'] for t in traits]
            deviations = [t['deviation'] for t in traits]
            
            # Calculate positions for bars
            y_pos = np.arange(len(features))
            y_height = 0.7  # Bar height
            
            # Create bars with appropriate colors
            for i, (feature, deviation) in enumerate(zip(features, deviations)):
                color = '#4CAF50' if deviation > 0 else '#E57373'  # Green for positive, red for negative
                ax_traits.barh(y_pos[i], deviation, height=y_height, color=color, alpha=0.7)
                
                # Add feature labels
                ax_traits.text(-0.5, y_pos[i], f"{feature}", 
                            va='center', ha='right', fontsize=10, fontweight='bold')
                
                # Add percentage labels
                ax_traits.text(deviation + (2 if deviation > 0 else -2), y_pos[i],
                            f"{deviation:.1f}%", va='center', ha='left' if deviation > 0 else 'right',
                            fontsize=10, fontweight='bold')
            
            # Add vertical line at 0
            ax_traits.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            
            # Set limits and hide y-ticks (since we have text labels)
            max_dev = max([abs(d) for d in deviations]) * 1.2
            ax_traits.set_xlim(-max_dev, max_dev)
            ax_traits.set_yticks([])
        else:
            ax_traits.text(0.5, 0.5, "No distinctive traits available", 
                         ha='center', fontsize=10, style='italic')
        
        #----- CONVERSION FEATURES SECTION -----#
        ax_features = fig.add_subplot(gs[1, 1])
        ax_features.axis('off')
        ax_features.text(0, 1, "Top Features Driving Conversion", fontsize=16, weight='bold')
        
        # Get top 5 features for this segment
        if 'Insufficient data' not in segment_conversion_features[segment]['Feature'].values:
            top_features = segment_conversion_features[segment].head(5)
            
            # Add top features with importance scores
            for i, (_, row) in enumerate(top_features.iterrows()):
                feature = row['Feature']
                importance = row['Importance']
                
                ax_features.text(0, 0.9 - i*0.15, f"{i+1}. {feature}", fontsize=12, weight='bold')
                ax_features.text(0.7, 0.9 - i*0.15, f"Importance: {importance:.3f}", fontsize=10)
        else:
            ax_features.text(0.5, 0.5, "Insufficient data for feature importance", 
                           ha='center', fontsize=10, style='italic')
        
        #----- ENGAGEMENT PATTERNS SECTION -----#
        ax_engagement = fig.add_subplot(gs[2, 0])
        ax_engagement.axis('off')
        ax_engagement.text(0, 1, "Digital Engagement Patterns", fontsize=16, weight='bold')
        
        engagement = profile['digital_engagement']
        
        # Format engagement metrics
        engagement_metrics = [
            ("Website Visits", f"{engagement['WebsiteVisits']:.1f}"),
            ("Pages per Visit", f"{engagement['PagesPerVisit']:.1f}"),
            ("Time on Site", f"{engagement['TimeOnSite']:.1f} min"),
            ("Email CTR", f"{engagement['email_ctr']*100:.1f}%"),
            ("Engagement Depth", f"{engagement['engagement_depth']:.1f}"),
            ("Social Propensity", f"{engagement['social_propensity']:.2f}")
        ]
        
        # Add engagement metrics
        for i, (metric, value) in enumerate(engagement_metrics):
            ax_engagement.text(0, 0.85 - i*0.14, metric, fontsize=12, weight='bold')
            ax_engagement.text(0.5, 0.85 - i*0.14, value, fontsize=12)
        
        #----- VALUE METRICS SECTION -----#
        ax_value = fig.add_subplot(gs[2, 1])
        ax_value.axis('off')
        ax_value.text(0, 1, "Banking Value Metrics", fontsize=16, weight='bold')
        
        # Format value metrics
        value_metrics = [
            ("Conversion Rate", f"{segment_conversion_rate:.1f}%"),
            ("Previous Purchases", f"{profile['transaction_history']['PreviousPurchases']:.1f}"),
            ("Loyalty Points", f"{profile['transaction_history']['LoyaltyPoints']:.0f}"),
            ("Click-Through Rate", f"{profile['campaign_performance']['ClickThroughRate']*100:.1f}%"),
            ("Ad Spend", f"${profile['campaign_performance']['AdSpend']:.2f}")
        ]
        
        # Add value metrics
        for i, (metric, value) in enumerate(value_metrics):
            ax_value.text(0, 0.85 - i*0.14, metric, fontsize=12, weight='bold')
            ax_value.text(0.5, 0.85 - i*0.14, value, fontsize=12)
        
        #----- CHANNEL PREFERENCES SECTION -----#
        ax_channels = fig.add_subplot(gs[3, 0])
        ax_channels.axis('off')
        ax_channels.text(0, 1, "Channel Preferences", fontsize=16, weight='bold')
        
        # Best channel info
        best_channel, best_channel_conv = best_channels[segment]
        ax_channels.text(0, 0.85, "Best Channel:", fontsize=12, weight='bold')
        ax_channels.text(0.4, 0.85, f"{best_channel}", fontsize=12, color=main_color)
        ax_channels.text(0.4, 0.78, f"Conversion: {best_channel_conv*100:.1f}%", fontsize=10)
        
        # Show distribution of all channels
        channel_dist = profile['channel_preferences']['distribution']
        sorted_channels = sorted(channel_dist.items(), key=lambda x: x[1], reverse=True)
        
        ax_channels.text(0, 0.65, "Channel Distribution:", fontsize=12, weight='bold')
        
        for i, (channel, percentage) in enumerate(sorted_channels):
            if i < 4:  # Show top 4 channels
                ax_channels.text(0, 0.55 - i*0.1, f"{channel}", fontsize=11)
                ax_channels.text(0.6, 0.55 - i*0.1, f"{percentage:.1f}%", fontsize=10)
        
        #----- CAMPAIGN PREFERENCES SECTION -----#
        ax_campaigns = fig.add_subplot(gs[3, 1])
        ax_campaigns.axis('off')
        ax_campaigns.text(0, 1, "Campaign Preferences", fontsize=16, weight='bold')
        
        # Best campaign info
        best_campaign, best_campaign_conv = best_campaign_types[segment]
        ax_campaigns.text(0, 0.85, "Best Campaign Type:", fontsize=12, weight='bold')
        ax_campaigns.text(0.5, 0.85, f"{best_campaign}", fontsize=12, color=main_color)
        ax_campaigns.text(0.5, 0.78, f"Conversion: {best_campaign_conv*100:.1f}%", fontsize=10)
        
        # Show distribution of all campaign types
        campaign_dist = profile['campaign_type_preferences']['distribution']
        sorted_campaigns = sorted(campaign_dist.items(), key=lambda x: x[1], reverse=True)
        
        ax_campaigns.text(0, 0.65, "Campaign Distribution:", fontsize=12, weight='bold')
        
        for i, (campaign, percentage) in enumerate(sorted_campaigns):
            if i < 4:  # Show top 4 campaign types
                ax_campaigns.text(0, 0.55 - i*0.1, f"{campaign}", fontsize=11)
                ax_campaigns.text(0.7, 0.55 - i*0.1, f"{percentage:.1f}%", fontsize=10)
        
        # Add border around the entire card
        for spine in ['top', 'bottom', 'left', 'right']:
            if hasattr(fig, 'spines') and hasattr(fig.spines, spine):
                fig.spines[spine].set_visible(True)
                fig.spines[spine].set_color(main_color)
                fig.spines[spine].set_linewidth(2)
        
        # Save the profile card
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        fig.savefig(f"{output_dir}/segment_{segment}_profile_card.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Store profile card data in dictionary for potential further use
        profile_cards[segment] = {
            'segment_id': segment,
            'size': {
                'count': segment_size,
                'percentage': f"{segment_size_pct:.1f}%"
            },
            # Add distinctive traits as requested
            'distinctive_traits': [
                {
                    'feature': trait['feature'],
                    'deviation': trait['deviation'],
                    'description': trait['description']
                }
                for trait in traits
            ],
            'engagement_patterns': {
                'website_visits': f"{profile['digital_engagement']['WebsiteVisits']:.1f}",
                'pages_per_visit': f"{profile['digital_engagement']['PagesPerVisit']:.1f}",
                'time_on_site': f"{profile['digital_engagement']['TimeOnSite']:.1f} min",
                'email_ctr': f"{profile['digital_engagement']['email_ctr']*100:.1f}%",
                'engagement_depth': f"{profile['digital_engagement']['engagement_depth']:.1f}",
                'social_propensity': f"{profile['digital_engagement']['social_propensity']:.2f}"
            },
            'value_metrics': {
                'conversion_rate': f"{segment_conversion_rate:.1f}%",
                'previous_purchases': f"{profile['transaction_history']['PreviousPurchases']:.1f}",
                'loyalty_points': f"{profile['transaction_history']['LoyaltyPoints']:.0f}"
            },
            'channel_preferences': {
                'best_channel': best_channel,
                'best_channel_conversion': f"{best_channel_conv*100:.1f}%",
                'distribution': channel_dist
            },
            'campaign_preferences': {
                'best_campaign': best_campaign,
                'best_campaign_conversion': f"{best_campaign_conv*100:.1f}%",
                'distribution': campaign_dist
            }
        }
    
    print(f"Created profile cards for {len(segments)} segments in '{output_dir}' directory")
    return profile_cards

def display_segment_profile_card(segment, output_dir="banking_profile_cards"):
    """Display a profile card for a specific segment"""
    from IPython.display import Image, display
    
    file_path = f"{output_dir}/segment_{segment}_profile_card.png"
    
    if os.path.exists(file_path):
        print(f"Profile Card for Segment {segment}:")
        display(Image(file_path))
        
        # Print key insights for this segment
        if segment in profile_cards:
            card = profile_cards[segment]
            
            print("\nKey Conversion Drivers:")
            for i, driver in enumerate(card['key_conversion_drivers'], 1):
                feature = driver['feature']
                deviation = driver['deviation']
                direction = driver['direction']
                
                print(f"  {i}. {feature}: {abs(deviation):.1f}% {direction} than average")
            
            print(f"\nBest Channel: {card['channel_preferences']['best_channel']} " +
                 f"(Conversion: {card['channel_preferences']['best_channel_conversion']})")
            
            print(f"Best Campaign: {card['campaign_preferences']['best_campaign']} " +
                 f"(Conversion: {card['campaign_preferences']['best_campaign_conversion']})")
    else:
        print(f"Profile card for Segment {segment} not found at {file_path}")

# Display the profile card for segment 0 (or any other segment you want to check)
display_segment_profile_card(0)