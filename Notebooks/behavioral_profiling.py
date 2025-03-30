
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import importlib


def behavioral_data_processing(df):
    """
    Process customer behavioral data by applying segmentation and engineering additional
    behavioral metrics.
    
    This function:
    1. Takes a DataFrame of customer data
    2. Imports and runs segment.py to add cluster labels
    3. Adds engineered features for deeper behavioral analysis
    
    Args:
        df (pd.DataFrame): Original DataFrame containing customer data
        
    Returns:
        pd.DataFrame: Processed DataFrame with Cluster_Label and engineered features
    """
    try:
        # Import the segmentation module
        import segment
        import importlib
        
        # Reload in case of recent edits
        importlib.reload(segment)
        
        # Apply segmentation to get Cluster_Label
        # Call main() with no arguments since that's how it's defined
        segmented_df = segment.main()
        
        # Validate that segmentation was successful
        if 'Cluster_Label' not in segmented_df.columns:
            raise ValueError("Segmentation did not produce a 'Cluster_Label' column")
        
        return segmented_df
        
    except ImportError:
        raise ImportError("Could not import 'segment.py'. Make sure it exists in your path.")
    except Exception as e:
        raise RuntimeError(f"Error applying segmentation: {str(e)}")
    
    # Data cleaning
    # 1. Remove unnecessary columns if they exist
    columns_to_drop = ['AdvertisingPlatform', 'AdvertisingTool']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 2. Convert categorical columns to category dtype for efficiency
    categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 3. Feature engineering: Create additional behavioral metrics
    
    # Email Click-Through Rate (CTR)
    if 'EmailClicks' in df.columns and 'EmailOpens' in df.columns:
        df['email_ctr'] = df.apply(
            lambda row: row['EmailClicks'] / row['EmailOpens'] if row['EmailOpens'] > 0 else 0, 
            axis=1
        )
    
    # Website Engagement Depth
    if 'PagesPerVisit' in df.columns and 'TimeOnSite' in df.columns:
        df['engagement_depth'] = df['PagesPerVisit'] * df['TimeOnSite']
    
    # Social Sharing Propensity
    if 'SocialShares' in df.columns and 'WebsiteVisits' in df.columns:
        df['social_propensity'] = df.apply(
            lambda row: row['SocialShares'] / row['WebsiteVisits'] if row['WebsiteVisits'] > 0 else 0,
            axis=1
        )
    
    return df


class SegmentProfiler:
    """
    A class for analyzing and visualizing customer segment profiles based on behavioral data.
    
    This class provides comprehensive methods for:
    - Calculating feature importance for conversion prediction
    - Creating detailed segment profiles 
    - Identifying optimal marketing channels and campaign types
    - Visualizing segment characteristics through profile cards
    
    Attributes:
        df (pd.DataFrame): DataFrame containing customer data with segment labels
        segments (List[int]): List of unique segment IDs
        all_metrics (List[str]): List of all metric column names used in analysis
        digital_metrics (List[str]): List of digital engagement metric column names
        transaction_metrics (List[str]): List of transaction history metric column names
        campaign_metrics (List[str]): List of campaign performance metric column names
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the SegmentProfiler with customer data.
        
        Args:
            df (pd.DataFrame): The dataset containing customer data with a 'Cluster_Label' column
                               identifying which segment each customer belongs to.
        
        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        # Validate input data has required columns
        required_columns = ['Cluster_Label', 'Conversion', 'CampaignChannel', 'CampaignType']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
        
        self.df = df.copy()
        self.segments = sorted(df['Cluster_Label'].unique())
        
        # Define metric categories
        self.digital_metrics = ['WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 
                                'SocialShares', 'EmailOpens', 'EmailClicks']
        self.transaction_metrics = ['PreviousPurchases', 'LoyaltyPoints']
        self.campaign_metrics = ['ClickThroughRate', 'ConversionRate', 'AdSpend']
        self.all_metrics = self.digital_metrics + self.transaction_metrics + self.campaign_metrics
        
        # Validate that all metrics exist in the DataFrame
        missing_metrics = [metric for metric in self.all_metrics if metric not in df.columns]
        if missing_metrics:
            warnings.warn(f"Some metrics are missing from DataFrame: {', '.join(missing_metrics)}")
            # Remove missing metrics from lists
            self.all_metrics = [m for m in self.all_metrics if m not in missing_metrics]
            self.digital_metrics = [m for m in self.digital_metrics if m not in missing_metrics]
            self.transaction_metrics = [m for m in self.transaction_metrics if m not in missing_metrics]
            self.campaign_metrics = [m for m in self.campaign_metrics if m not in missing_metrics]
    
    def analyze_conversion_feature_importance(self, segment: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate feature importance for conversion prediction within a specific segment.
        
        This method uses Random Forest feature importance as the primary approach.
        If there's insufficient data or lack of class variability, it falls back to
        a segment comparison method.
        
        Args:
            segment (int): The segment ID to analyze
            
        Returns:
            Tuple containing:
                pd.DataFrame: DataFrame with features and their importance scores
                np.ndarray: Raw importance scores
        
        Example:
            >>> profiler = SegmentProfiler(customer_data)
            >>> feature_imp_df, importances = profiler.analyze_conversion_feature_importance(1)
            >>> print(feature_imp_df.head())
               Feature  Importance
            0  WebsiteVisits    0.245
            1  TimeOnSite       0.189
            2  EmailClicks      0.143
        """
        # Prepare data for feature importance analysis
        X = self.df[self.all_metrics].copy()

        # Scale features for better model performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Filter data for just this segment
        segment_data = self.df[self.df['Cluster_Label'] == segment]
        
        # Get features and target for this segment
        X_segment = X_scaled.iloc[segment_data.index]
        y_segment = segment_data['Conversion']
        
        # Try primary approach: Random Forest feature importance
        if len(y_segment) >= 10 and y_segment.nunique() >= 2:
            try:
                # Use Random Forest classifier for feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_segment, y_segment)
                
                # Get feature importances
                importances = rf.feature_importances_
                
                # Create a dataframe with feature importances
                feature_imp = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                return feature_imp, importances
                
            except Exception:
                # If model fails, fall back to comparison method
                pass
        
        # Fallback approach: Compare with other segments
        # This handles: 1) small segments, 2) no class variability, 3) 100% conversion rate
        
        # Compare to other segments
        all_other_segments = self.df[self.df['Cluster_Label'] != segment]
        
        # Calculate mean differences for each feature
        feature_diff = {}
        for feature in X.columns:
            segment_mean = segment_data[feature].mean()
            others_mean = all_other_segments[feature].mean()
            
            # Calculate percent difference (avoiding division by zero)
            if others_mean != 0:
                pct_diff = (segment_mean - others_mean) / others_mean * 100
                feature_diff[feature] = abs(pct_diff)  # Use absolute difference for ranking
            else:
                feature_diff[feature] = 0 if segment_mean == 0 else 100
        
        # Sort features by their distinctive value
        sorted_features = sorted(feature_diff.items(), key=lambda x: x[1], reverse=True)
        
        # Create DataFrame in same format as feature importance
        normalized_values = [f[1]/100 for f in sorted_features]  # Normalize to similar scale
        feature_imp = pd.DataFrame({
            'Feature': [f[0] for f in sorted_features],
            'Importance': normalized_values
        })
        
        # Convert importance values to numpy array for consistency with RF method
        importances = np.array(normalized_values)
        
        return feature_imp, importances

    def calculate_segment_profiles(self) -> Dict[int, Dict[str, Any]]:
        """
        Calculate detailed profiles for each customer segment.
        
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary containing comprehensive segment profiles
                                      with engagement metrics, transaction history,
                                      campaign performance, and preferences
        
        Example:
            >>> profiler = SegmentProfiler(customer_data)
            >>> profiles = profiler.calculate_segment_profiles()
            >>> email_ctr = profiles[1]['digital_engagement']['email_ctr']
            >>> print(f"Segment 1 Email CTR: {email_ctr:.2%}")
        """
        # Calculate segment profiles with key metrics
        segment_profiles = {}

        for segment in self.segments:
            # Filter data for this segment
            segment_data = self.df[self.df['Cluster_Label'] == segment]
            segment_size = len(segment_data)
            
            # Initialize profile object with default empty structures
            segment_profiles[segment] = {
                'size': segment_size,
                'size_percentage': segment_size / len(self.df) * 100,
                'digital_engagement': {},
                'transaction_history': {},
                'campaign_performance': {},
                'channel_preferences': {
                    'distribution': {},
                    'conversion_rate': {}
                },
                'campaign_type_preferences': {
                    'distribution': {},
                    'conversion_rate': {}
                }
            }
            
            # Digital engagement metrics
            for metric in self.digital_metrics:
                if metric in self.df.columns:
                    segment_profiles[segment]['digital_engagement'][metric] = segment_data[metric].mean()
            
            # Transaction history
            for metric in self.transaction_metrics:
                if metric in self.df.columns:
                    segment_profiles[segment]['transaction_history'][metric] = segment_data[metric].mean()
            
            # Campaign performance
            for metric in self.campaign_metrics:
                if metric in self.df.columns:
                    segment_profiles[segment]['campaign_performance'][metric] = segment_data[metric].mean()
            
            # Conversion rate
            segment_profiles[segment]['campaign_performance']['conversion_count'] = segment_data['Conversion'].sum()
            segment_profiles[segment]['campaign_performance']['conversion_rate'] = segment_data['Conversion'].mean()
            
            # Channel preferences
            if 'CampaignChannel' in segment_data.columns:
                channel_counts = segment_data['CampaignChannel'].value_counts(normalize=True) * 100
                
                for channel in segment_data['CampaignChannel'].unique():
                    channel_data = segment_data[segment_data['CampaignChannel'] == channel]
                    segment_profiles[segment]['channel_preferences']['conversion_rate'][channel] = channel_data['Conversion'].mean()
                
                segment_profiles[segment]['channel_preferences']['distribution'] = channel_counts.to_dict()
            
            # Campaign type preferences
            if 'CampaignType' in segment_data.columns:
                campaign_counts = segment_data['CampaignType'].value_counts(normalize=True) * 100
                
                for campaign_type in segment_data['CampaignType'].unique():
                    campaign_data = segment_data[segment_data['CampaignType'] == campaign_type]
                    segment_profiles[segment]['campaign_type_preferences']['conversion_rate'][campaign_type] = campaign_data['Conversion'].mean()
                
                segment_profiles[segment]['campaign_type_preferences']['distribution'] = campaign_counts.to_dict()

        # Calculate additional behavioral metrics
        for segment in self.segments:
            profile = segment_profiles[segment]
            
            # Email engagement effectiveness (CTR)
            if 'EmailOpens' in profile['digital_engagement'] and profile['digital_engagement'].get('EmailOpens', 0) > 0:
                profile['digital_engagement']['email_ctr'] = (
                    profile['digital_engagement'].get('EmailClicks', 0) / 
                    profile['digital_engagement']['EmailOpens']
                )
            else:
                profile['digital_engagement']['email_ctr'] = 0
            
            # Website engagement depth
            if 'PagesPerVisit' in profile['digital_engagement'] and 'TimeOnSite' in profile['digital_engagement']:
                profile['digital_engagement']['engagement_depth'] = (
                    profile['digital_engagement']['PagesPerVisit'] * 
                    profile['digital_engagement']['TimeOnSite']
                )
            else:
                profile['digital_engagement']['engagement_depth'] = 0
            
            # Social sharing propensity
            if 'WebsiteVisits' in profile['digital_engagement'] and profile['digital_engagement'].get('WebsiteVisits', 0) > 0:
                profile['digital_engagement']['social_propensity'] = (
                    profile['digital_engagement'].get('SocialShares', 0) / 
                    profile['digital_engagement']['WebsiteVisits']
                )
            else:
                profile['digital_engagement']['social_propensity'] = 0
        
        return segment_profiles

    def find_optimal_channels_and_campaigns(self, segment_profiles: Dict[int, Dict[str, Any]]) -> Tuple[Dict[int, Tuple[str, float]], Dict[int, Tuple[str, float]]]:
        """
        Determine the best channel and campaign type for each segment based on conversion rates.
        
        Args:
            segment_profiles (Dict[int, Dict[str, Any]]): Dictionary containing segment profiles
                
        Returns:
            Tuple containing:
                Dict[int, Tuple[str, float]]: Mapping of segment ID to (best_channel, conversion_rate)
                Dict[int, Tuple[str, float]]: Mapping of segment ID to (best_campaign, conversion_rate)
        
        Example:
            >>> profiler = SegmentProfiler(customer_data)
            >>> profiles = profiler.calculate_segment_profiles()
            >>> best_channels, best_campaigns = profiler.find_optimal_channels_and_campaigns(profiles)
            >>> print(f"Best channel for segment 1: {best_channels[1][0]}")
        """
        best_channels = {}
        best_campaign_types = {}
        
        for segment in self.segments:
            # Find best channel (with error handling for missing data)
            try:
                channel_conv = segment_profiles[segment]['channel_preferences']['conversion_rate']
                if channel_conv:  # Check if the dictionary is not empty
                    best_channel = max(channel_conv.items(), key=lambda x: x[1])
                    best_channels[segment] = best_channel
                else:
                    best_channels[segment] = ("No data", 0.0)
            except (KeyError, ValueError):
                best_channels[segment] = ("Error", 0.0)
            
            # Find best campaign type
            try:
                campaign_conv = segment_profiles[segment]['campaign_type_preferences']['conversion_rate']
                if campaign_conv:  # Check if the dictionary is not empty
                    best_campaign = max(campaign_conv.items(), key=lambda x: x[1])
                    best_campaign_types[segment] = best_campaign
                else:
                    best_campaign_types[segment] = ("No data", 0.0)
            except (KeyError, ValueError):
                best_campaign_types[segment] = ("Error", 0.0)
        
        return best_channels, best_campaign_types

    def calculate_segment_deviations(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculate how each segment deviates from the overall average for key metrics.
        
        Returns:
            Tuple containing:
                pd.DataFrame: DataFrame with percentage deviations from average for each segment and metric
                List[str]: List of numeric metrics that were analyzed
        
        Example:
            >>> profiler = SegmentProfiler(customer_data)
            >>> deviations, metrics = profiler.calculate_segment_deviations()
            >>> print(f"Segment 1 has {deviations.loc[1, 'WebsiteVisits']:.1f}% more website visits than average")
        """
        # Identify numeric metrics from all_metrics
        numeric_metrics = []
        for metric in self.all_metrics:
            if metric in self.df.columns and self.df[metric].dtype.kind in 'ifc':  # integer, float, or complex
                numeric_metrics.append(metric)
        
        segment_deviations = pd.DataFrame(index=self.segments, columns=numeric_metrics)
        
        for metric in numeric_metrics:
            overall_avg = self.df[metric].mean()
            
            for segment in self.segments:
                segment_avg = self.df[self.df['Cluster_Label'] == segment][metric].mean()
                # Avoid division by zero
                if overall_avg != 0:
                    percent_diff = (segment_avg - overall_avg) / overall_avg * 100
                else:
                    percent_diff = 0 if segment_avg == 0 else 100  # handle division by zero
                segment_deviations.loc[segment, metric] = percent_diff
        
        # Convert all values to float to ensure they're numeric
        segment_deviations = segment_deviations.astype(float)
        
        return segment_deviations, numeric_metrics

    def identify_key_segment_traits(self, segment_conversion_features: Dict[int, pd.DataFrame], 
                                    segment_deviations: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
        """
        Identify the most important defining traits for each segment by combining
        feature importance with segment deviations.
        
        Args:
            segment_conversion_features (Dict[int, pd.DataFrame]): Feature importance for each segment
            segment_deviations (pd.DataFrame): Percentage deviations from average
            
        Returns:
            Dict[int, List[Dict[str, Any]]]: Dictionary with important traits for each segment
        
        Example:
            >>> profiler = SegmentProfiler(customer_data)
            >>> deviations, _ = profiler.calculate_segment_deviations()
            >>> feature_importance = {1: feature_imp_df1, 2: feature_imp_df2}
            >>> traits = profiler.identify_key_segment_traits(feature_importance, deviations)
            >>> print(f"Key trait for segment 1: {traits[1][0]['feature']} ({traits[1][0]['description']})")
        """
        important_traits = {}
        
        for segment in self.segments:
            # Skip if segment is not in the features dictionary
            if segment not in segment_conversion_features:
                important_traits[segment] = []
                continue
            
            # Get features for this segment
            segment_features = segment_conversion_features[segment]
            
            # Initialize traits list
            traits = []
            
            # Check if we have valid feature importance data
            if len(segment_features) > 0 and 'Insufficient data' not in segment_features['Feature'].values:
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
        
        return important_traits


class SegmentVisualizer:
    """
    A class for creating visual representations of customer segment profiles.
    
    This class provides methods for generating visualizations including radar charts,
    bar charts, and comprehensive profile cards that highlight key segment traits.
    
    Attributes:
        main_color (str): Primary color used in visualizations
        secondary_color (str): Secondary color used in visualizations
        positive_color (str): Color used for positive values
        negative_color (str): Color used for negative values
    """
    
    def __init__(self, color_scheme: Optional[Dict[str, str]] = None):
        """
        Initialize the SegmentVisualizer with an optional custom color scheme.
        
        Args:
            color_scheme (Optional[Dict[str, str]]): Custom color scheme with keys:
                'main', 'secondary', 'positive', 'negative'
        """
        # Define default color scheme
        default_colors = {
            'main': '#1E5C97',       # Banking blue
            'secondary': '#78A2CC',  # Lighter blue
            'positive': '#4CAF50',   # Green for positive values
            'negative': '#E57373',   # Red for negative values
        }
        
        # Use provided colors or defaults
        colors = color_scheme or default_colors
        
        self.main_color = colors.get('main', default_colors['main'])
        self.secondary_color = colors.get('secondary', default_colors['secondary'])
        self.positive_color = colors.get('positive', default_colors['positive'])
        self.negative_color = colors.get('negative', default_colors['negative'])
    
    def create_profile_card(self, 
                          segment: int, 
                          profile_data: Dict[str, Any],
                          features: List[str],
                          deviations: List[float],
                          conversion_rate: float,
                          best_channel: Tuple[str, float],
                          best_campaign: Tuple[str, float],
                          output_path: str):
        """
        Create a comprehensive visual profile card for a customer segment.
        
        Args:
            segment (int): Segment ID
            profile_data (Dict[str, Any]): Profile data for this segment
            features (List[str]): List of top features for this segment
            deviations (List[float]): List of deviations from average for top features
            conversion_rate (float): Conversion rate for this segment
            best_channel (Tuple[str, float]): Best channel and its conversion rate
            best_campaign (Tuple[str, float]): Best campaign type and its conversion rate
            output_path (str): Path where the visualization should be saved
        
        Returns:
            None: The profile card is saved to the specified path
        """
        # Create figure
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.6, 1.8, 1.5], width_ratios=[1, 1])
        
        # Add a border to the entire figure
        fig.patch.set_edgecolor(self.main_color)
        fig.patch.set_linewidth(1)
        
        #----- HEADER SECTION -----#
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        ax_header.text(0.5, 0.6, f"Banking Segment {segment} Profile", 
                      fontsize=22, weight='bold', ha='center', color=self.main_color)
        ax_header.text(0.5, 0.2, 
                      f"Size: {profile_data['size']['count']} customers ({profile_data['size']['percentage']})", 
                      fontsize=14, ha='center')
        
        #----- TOP FEATURES WITH DEVIATION COMPARISON -----#
        ax_features = fig.add_subplot(gs[1, 0])
        ax_features.axis('off')
        ax_features.text(0.5, 1.05, "Top Features Driving Conversion", 
                       fontsize=16, weight='bold', ha='center')
        ax_features.text(0.5, 0.97, "Comparison to Average", 
                       fontsize=14, ha='center')
        
        # Check if we have valid features
        if features and len(features) > 0:
            # Position for bars
            y_pos = np.arange(len(features))
            bar_height = 0.5
            
            # Create horizontal bars for deviations
            for i, (feature, deviation) in enumerate(zip(features, deviations)):
                # Color based on whether higher or lower than average
                color = self.positive_color if deviation > 0 else self.negative_color
                
                # Create the horizontal bar
                ax_features.barh(y_pos[i], deviation, height=bar_height, color=color, alpha=0.8)
                
                # Feature label on y-axis
                ax_features.text(-5, y_pos[i], feature, va='center', ha='right', 
                               fontsize=12, fontweight='bold')
                
                # Add percentage labels
                label_pos = deviation + (1 if deviation >= 0 else -1)
                ax_features.text(label_pos, y_pos[i], f"{deviation:.1f}% vs. avg", 
                               va='center', ha='left' if deviation >= 0 else 'right',
                               fontsize=11, fontweight='bold')
                
            # Set axis limits
            max_dev = max([abs(d) for d in deviations]) * 1.2
            ax_features.set_xlim(-max_dev, max_dev)
            
            # Add extra space for labels
            ax_features.set_ylim(-0.5, len(features)-0.5)
            
            # Remove y-ticks
            ax_features.set_yticks([])
            
            # Add a vertical line at 0
            ax_features.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add legend
            legend_elements = [
                Patch(facecolor=self.positive_color, label='Higher than average', alpha=0.8),
                Patch(facecolor=self.negative_color, label='Lower than average', alpha=0.8)
            ]
            ax_features.legend(handles=legend_elements, loc='lower center', fontsize=10, 
                             frameon=True, framealpha=0.9, edgecolor='lightgray')
            
            # Add explanatory note
            ax_features.text(0.5, -0.15, 
                           "Shows how top conversion-driving features compare to average", 
                           fontsize=9, style='italic', ha='center', transform=ax_features.transAxes)
        else:
            ax_features.text(0.5, 0.5, "Insufficient data for feature importance", 
                           ha='center', fontsize=12, style='italic')
        
        #----- DIGITAL ENGAGEMENT PROFILE (RADAR CHART) -----#
        ax_engagement = fig.add_subplot(gs[1, 1], polar=True)
        
        # Get engagement metrics for radar chart
        engagement = profile_data.get('digital_engagement', {})
        
        # Define the metrics to display in the radar chart
        radar_metrics = [
            'Website Visits', 
            'Pages/Visit', 
            'Time on Site', 
            'Email CTR', 
            'Engagement Depth', 
            'Social Propensity'
        ]
        
        # Get values for these metrics (with safety checks)
        radar_values = [
            engagement.get('WebsiteVisits', 0),
            engagement.get('PagesPerVisit', 0),
            engagement.get('TimeOnSite', 0),
            engagement.get('email_ctr', 0),
            engagement.get('engagement_depth', 0),
            engagement.get('social_propensity', 0)
        ]
        
        # Normalize the values for display
        max_values = {
            'WebsiteVisits': 10,  # Example max values, adjust based on your data
            'PagesPerVisit': 5,
            'TimeOnSite': 10,
            'email_ctr': 0.5,
            'engagement_depth': 50,
            'social_propensity': 2
        }
        
        # Ensure we don't divide by zero
        for key, value in max_values.items():
            if value == 0:
                max_values[key] = 1
        
        # Create mapping from radar metrics to internal keys
        radar_to_key = {
            'Website Visits': 'WebsiteVisits',
            'Pages/Visit': 'PagesPerVisit',
            'Time on Site': 'TimeOnSite',
            'Email CTR': 'email_ctr',
            'Engagement Depth': 'engagement_depth',
            'Social Propensity': 'social_propensity'
        }
        
        # Normalize the values
        normalized_values = []
        for i, metric in enumerate(radar_metrics):
            key = radar_to_key.get(metric)
            if key in max_values:
                max_val = max_values[key]
                val = radar_values[i]
                normalized_values.append(min(val / max_val, 1) if max_val > 0 else 0)
            else:
                normalized_values.append(0)
        
        # Add the first value again to close the polygon
        radar_metrics = np.append(radar_metrics, radar_metrics[0])
        normalized_values = np.append(normalized_values, normalized_values[0])
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(radar_metrics)-1, endpoint=False)
        angles = np.append(angles, angles[0])  # Close the loop
        
        # Draw the radar chart
        ax_engagement.plot(angles, normalized_values, 'o-', linewidth=2, color=self.main_color)
        ax_engagement.fill(angles, normalized_values, alpha=0.25, color=self.main_color)
        
        # Set the labels
        ax_engagement.set_thetagrids(angles * 180/np.pi, radar_metrics, fontsize=10)
        
        # Draw concentric circles
        ax_engagement.set_ylim(0, 1)
        ax_engagement.grid(True, alpha=0.3)
        
        # Add a title
        plt.figtext(0.75, 0.72, "Digital Engagement Profile", fontsize=16, weight='bold', ha='center')
        
        #----- VALUE METRICS SECTION (TABLE) -----#
        ax_value = fig.add_subplot(gs[2, 0])
        ax_value.axis('off')
        
        # Add a title
        ax_value.text(0.5, 1.1, "Value Metrics", fontsize=16, weight='bold', ha='center')
        
        # Create a table
        value_data = [
            ['Metric', 'Value'],
            ['Conversion Rate', f"{conversion_rate:.1f}%"],
            ['Previous Purchases', f"{profile_data.get('transaction_history', {}).get('PreviousPurchases', 0):.1f}"],
            ['Loyalty Points', f"{profile_data.get('transaction_history', {}).get('LoyaltyPoints', 0):.0f}"]
        ]
        
        # Create the table
        table = ax_value.table(
            cellText=value_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.5, 0.3],
            bbox=[0.15, 0.15, 0.7, 0.7]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Style header row
        for j in range(2):
            table[(0, j)].set_facecolor('#EBF1F6')
            table[(0, j)].set_text_props(weight='bold')
        
        #----- CHANNEL & CAMPAIGN PREFERENCES SECTION -----#
        ax_pref = fig.add_subplot(gs[2, 1])
        ax_pref.axis('off')
        
        # Left side - Channel Preferences
        ax_pref.text(0.25, 1.1, "Channel Preferences", fontsize=16, weight='bold', ha='center')
        
        # Best channel info
        ax_pref.text(0.05, 0.85, "Best Channel:", fontsize=12, weight='bold')
        ax_pref.text(0.35, 0.85, f"{best_channel[0]}", fontsize=12, color=self.main_color, weight='bold')
        ax_pref.text(0.35, 0.78, f"Conversion: {best_channel[1]*100:.1f}%", fontsize=10)
        
        # Right side - Campaign Preferences
        ax_pref.text(0.75, 1.1, "Campaign Preferences", fontsize=16, weight='bold', ha='center')
        
        # Best campaign info
        ax_pref.text(0.55, 0.85, "Best Campaign Type:", fontsize=12, weight='bold')
        ax_pref.text(0.85, 0.85, f"{best_campaign[0]}", fontsize=12, color=self.main_color, weight='bold')
        ax_pref.text(0.85, 0.78, f"Conversion: {best_campaign[1]*100:.1f}%", fontsize=10)
        
        # Add a divider line
        ax_pref.axvline(x=0.5, ymin=0, ymax=0.9, color='lightgray', linewidth=1, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        # Save the profile card
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def get_segment_profile_data(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Extract comprehensive profile data for all segments in the dataset without creating visualizations.
    
    Args:
        df (pd.DataFrame): DataFrame containing customer data with Cluster_Label column
        
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary with profile data for each segment
    """
    # Initialize profiler
    profiler = SegmentProfiler(df)
    
    # Get segments and conversion rates
    segments = sorted(df['Cluster_Label'].unique())
    segment_conversion_rates = df.groupby('Cluster_Label')['Conversion'].mean() * 100
    
    # Calculate segment profiles
    segment_profiles = profiler.calculate_segment_profiles()
    
    # Calculate conversion feature importance for each segment
    segment_conversion_features = {}
    for segment in segments:
        feature_imp, _ = profiler.analyze_conversion_feature_importance(segment)
        segment_conversion_features[segment] = feature_imp
    
    # Calculate segment deviations from average
    segment_deviations, numeric_metrics = profiler.calculate_segment_deviations()
    
    # Determine the best channel and campaign type for each segment
    best_channels, best_campaign_types = profiler.find_optimal_channels_and_campaigns(segment_profiles)
    
    # Identify important traits for each segment
    important_traits = profiler.identify_key_segment_traits(segment_conversion_features, segment_deviations)
    
    # Create profile data for each segment
    profile_cards = {}
    
    for segment in segments:
        # Get segment data
        profile = segment_profiles[segment]
        segment_size = profile['size']
        segment_size_pct = profile['size_percentage']
        segment_conversion_rate = segment_conversion_rates[segment]
        
        # Get top 5 features for this segment
        if 'Insufficient data' not in segment_conversion_features[segment]['Feature'].values:
            # Get top 5 features by importance
            top_features = segment_conversion_features[segment].head(5)
            features = top_features['Feature'].values
            
            # Get deviations for these features
            deviations = []
            for feature in features:
                if feature in segment_deviations.columns:
                    deviations.append(segment_deviations.loc[segment, feature])
                else:
                    deviations.append(0)  # Default if not found
            
            # Sort by importance (using the original DataFrame order)
            features = list(features)
            deviations = list(deviations)
        else:
            features = []
            deviations = []
        
        # Get best channel and campaign type
        best_channel = best_channels[segment]
        best_campaign = best_campaign_types[segment]
        
        # Create profile card data structure
        profile_cards[segment] = {
            'segment_id': segment,
            'size': {
                'count': segment_size,
                'percentage': f"{segment_size_pct:.1f}%"
            },
            'top_features': [
                {
                    'feature': feat,
                    'deviation': dev,
                    'direction': 'higher' if dev > 0 else 'lower'
                }
                for feat, dev in zip(features, deviations)
            ] if features else [],
            'engagement_patterns': {
                'website_visits': f"{profile.get('digital_engagement', {}).get('WebsiteVisits', 0):.1f}",
                'pages_per_visit': f"{profile.get('digital_engagement', {}).get('PagesPerVisit', 0):.1f}",
                'time_on_site': f"{profile.get('digital_engagement', {}).get('TimeOnSite', 0):.1f} min",
                'email_ctr': f"{profile.get('digital_engagement', {}).get('email_ctr', 0)*100:.1f}%",
                'engagement_depth': f"{profile.get('digital_engagement', {}).get('engagement_depth', 0):.1f}",
                'social_propensity': f"{profile.get('digital_engagement', {}).get('social_propensity', 0):.2f}"
            },
            'value_metrics': {
                'conversion_rate': f"{segment_conversion_rate:.1f}%",
                'previous_purchases': f"{profile.get('transaction_history', {}).get('PreviousPurchases', 0):.1f}",
                'loyalty_points': f"{profile.get('transaction_history', {}).get('LoyaltyPoints', 0):.0f}"
            },
            'channel_preferences': {
                'best_channel': best_channel[0],
                'best_channel_conversion': f"{best_channel[1]*100:.1f}%",
                'distribution': profile.get('channel_preferences', {}).get('distribution', {})
            },
            'campaign_preferences': {
                'best_campaign': best_campaign[0],
                'best_campaign_conversion': f"{best_campaign[1]*100:.1f}%",
                'distribution': profile.get('campaign_type_preferences', {}).get('distribution', {})
            }
        }
    
    return profile_cards


def run_main():
    """
    Main function to run the entire behavioral profiling pipeline.
    
    This function loads the dataset, processes it, and returns profile data
    for all customer segments without creating visualizations.
    
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary with profile data for each segment
    """
    # 1. Load data
    df = pd.read_csv('/Users/cindy/Desktop/DSA3101-Project-3/Data/digital_marketing_campaign_dataset.csv')
    
    # 2. Process the data
    processed_df = behavioral_data_processing(df)
    
    # 3. Get segment profile data (without visualizations)
    profile_data = get_segment_profile_data(processed_df)
    
    return profile_data



profile_data = run_main()
print(profile_data) 