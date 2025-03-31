"""
Segment profiler module for customer behavioral analysis.

This module contains the SegmentProfiler class for analyzing customer segments
and calculating detailed profile metrics.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


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
    
        This method:
        1. Identifies all numeric metrics from the available columns
        2. Calculates the overall average for each metric across all segments
        3. For each segment, calculates the percentage difference from the overall average
        4. Handles division by zero cases appropriately
        
        The deviation values represent how much higher or lower (as a percentage) each 
        segment's average is compared to the overall average. Positive values indicate 
        the segment is above average, negative values indicate below average.
        
        Returns:
            Tuple containing:
                pd.DataFrame: DataFrame with percentage deviations from average for each segment and metric.
                            Rows are segment IDs, columns are metrics.
                List[str]: List of numeric metrics that were analyzed
        
        Examples:
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


def main():
    """
    Main function to run the segment profiling pipeline.
    
    This function:
    1. Parses command line arguments
    2. Processes the input data
    3. Generates segment profiles
    4. Saves the profiles to a JSON file
    
    Example usage:
        python segment_profiler.py --input data.csv --output profiles.json
    """
    import argparse
    import json
    import os
    import sys
    
    # Add parent directory to sys.path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    try:
        from data_processing import behavioral_data_processing
    except ImportError:
        # If not found in parent directory, try the current directory
        try:
            import data_processing
            behavioral_data_processing = data_processing.behavioral_data_processing
        except ImportError:
            raise ImportError("Could not import behavioral_data_processing. Please make sure data_processing.py is in the same directory or in the parent directory.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate customer segment profiles.')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='profiles.json',
                        help='Path where to save the profile JSON')
    
    args = parser.parse_args()
    
    # Process the data
    print("Processing data...")
    processed_df = behavioral_data_processing(args.input)
    
    # Get segment profile data
    print("Generating segment profiles...")
    profile_data = get_segment_profile_data(processed_df)
    
    # Save profile data to JSON
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
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    print(f"Profile data saved to {args.output}")
    return profile_data


if __name__ == "__main__":
    main()
