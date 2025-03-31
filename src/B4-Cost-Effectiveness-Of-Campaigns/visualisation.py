import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric(df, group_col, metric_col, title, ylabel):
    """Plot bar chart for a given metric."""
    print(f"Plotting bar chart for {metric_col} grouped by {group_col}...")
    df_sorted = df.sort_values(by=metric_col, ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=group_col, y=metric_col, data=df_sorted)
    plt.title(title, fontsize=16)
    plt.xlabel(group_col, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    print("Plotting completed.")

def plot_cluster_profiles(df, categorical_features, numerical_features):
    """Visualize customer segment profiles using countplots and boxplots."""
    print(f"Visualizing customer segment profiles for categorical features: {categorical_features}...")
    num_features = len(categorical_features)
    num_cols = 1  # Fixed number of columns
    num_rows = num_features

    plt.figure(figsize=(10, 6 * num_rows))
    for i, feature in enumerate(categorical_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x='CustomerSegment', hue=feature, data=df)
        plt.title(f"Distribution of {feature} Across Segments")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

    print(f"Visualizing numerical features: {numerical_features}...")
    num_features = len(numerical_features)
    num_cols = 3
    num_rows = (num_features + 2) // 3  # Adjusted for layout

    plt.figure(figsize=(12, 6 * num_rows))
    for i, feature in enumerate(numerical_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(x='CustomerSegment', y=feature, data=df)
        plt.title(f"Comparison of {feature} Across Segments")

    plt.tight_layout()
    plt.show()
    print("Cluster profile visualization completed.")
