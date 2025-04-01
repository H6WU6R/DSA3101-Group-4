import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def plot_metric(df, group_col, metric_col, title, ylabel, save_path):
    df_sorted = df.sort_values(by=metric_col, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=group_col, y=metric_col, data=df_sorted)
    plt.title(title, fontsize=16)
    plt.xlabel(group_col, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the plot after saving


def plot_all_metrics(final_dfs, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for group_col, metrics in final_dfs.items():
        plot_metric(metrics['AdSpend'], group_col, 'AdSpend', f"AdSpend by {group_col}", 'Ad Spend ($)', os.path.join(save_folder, f"{group_col}_AdSpend.png"))
        plot_metric(metrics['Conversion'], group_col, 'Conversion', f"Conversion Rate by {group_col}", 'Conversion Rate', os.path.join(save_folder, f"{group_col}_Conversion.png"))
        plot_metric(metrics['CPA'], group_col, f'{group_col}CPA', f"CPA by {group_col}", 'Cost Per Acquisition ($)', os.path.join(save_folder, f"{group_col}_CPA.png"))

def plot_countplots(marketing_df, categorical_features, save_folder):
    print("Generating countplots...")

    num_features = len(categorical_features)
    num_cols = 1  # Fixed number of columns
    num_rows = num_features  # Each feature gets its own row

    plt.figure(figsize=(10, 6 * num_rows))

    for i, feature in enumerate(categorical_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x='CustomerSegment', hue=feature, data=marketing_df)
        plt.title(f"Distribution of {feature} Across Segments")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_folder, "categorical_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Countplots saved to {save_path}")


def plot_boxplots(marketing_df, numerical_features, save_folder):
    print("Generating boxplots...")

    num_features = len(numerical_features)
    num_cols = 3  # Fixed number of columns for better layout
    num_rows = math.ceil(num_features / num_cols)

    plt.figure(figsize=(12, 6 * num_rows))

    for i, feature in enumerate(numerical_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(x='CustomerSegment', y=feature, data=marketing_df)
        plt.title(f"Comparison of {feature} Across Segments")

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_folder, "numerical_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Boxplots saved to {save_path}")
