# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors

import seaborn as sns

from scipy.stats import norm

# %%
def plot_histogram_kde_with_std(df, columns, bins=30, kde=False, normal_distribution=False, mean=False, median=False, print_measures=False):
        # Determine the number of columns and rows for the grid
    num_plots = len(columns)
    num_cols = 3  # Maximum 3 columns wide
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten to easily iterate over axes
    
    # To store handles and labels for the legend
    all_handles = []
    all_labels = []

    for i, column in enumerate(columns):
        # Extract the column data
        data = df[column]
        
        # Calculate mean, median, and standard deviation
        mean_value = data.mean()
        median_value = data.median()
        std_dev = data.std()

        # Plot the histogram
        ax1 = axes[i]
        ax1.hist(data, bins=bins, color='grey', alpha=0.6, edgecolor='black', label='Histogram (Counts)')
        ax1.set_ylabel('Frequency Count')

        # Set up secondary y-axis for KDE and Normal Distribution
        ax2 = ax1.twinx()
        ax2.set_ylabel('Density')

        if kde or normal_distribution:
            # Plot KDE line if KDE=True
            if kde:
                sns.kdeplot(data, clip=(data.min(), data.max()), ax=ax2, color='blue', linewidth=1, label='KDE')

            # Plot normal distribution curve and shaded areas if normal_distribution=True
            if normal_distribution:
                x = np.linspace(data.min(), data.max(), 1000)
                y = norm.pdf(x, mean_value, std_dev)
                ax2.plot(x, y, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Normal Distribution')

                # Colors and labels for shading
                shading_info = [
                    (mean_value - std_dev, mean_value + std_dev, 'blue', "1σ range"),
                    (mean_value - 2 * std_dev, mean_value - std_dev, 'green', "2σ range"),
                    (mean_value + std_dev, mean_value + 2 * std_dev, 'green', "2σ range"),
                    (mean_value - 3 * std_dev, mean_value - 2 * std_dev, 'orange', "3σ range"),
                    (mean_value + 2 * std_dev, mean_value + 3 * std_dev, 'orange', "3σ range")
                ]

                # Shade areas
                for lower_bound, upper_bound, color, label in shading_info:
                    if lower_bound < upper_bound:
                        ax2.fill_between(x, 0, y, where=(x > lower_bound) & (x < upper_bound), color=color, alpha=0.3)

        ax2.set_ylim(0, ax2.get_ylim()[1])

        # Add vertical lines for mean and median
        if mean:
            ax1.axvline(mean_value, color='red', linestyle=':', linewidth=3, label='Mean')
        if median:
            ax1.axvline(median_value, color='indigo', linestyle=':', linewidth=3, label='Median')

        # Collect handles and labels for the legend
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles += handles + handles2
        all_labels += labels + labels2

        # Print skewness and kurtosis if required
        if print_measures:
            print(f'{df[column].name} Skewness: {data.skew():.4F}')
            print(f'{df[column].name} Excess Kurtosis: {round(data.kurt()-3,4)}')

        # Set title and labels
        ax1.set_title(f'{column} Distribution')
        ax1.set_xlabel(column)

    # Hide any unused axes in the grid
    for j in range(i + 1, num_rows * num_cols):
        axes[j].axis('off')

    # Remove duplicate handles and labels for the legend
    unique_handles_labels = {label: handle for handle, label in zip(all_handles, all_labels)}
    unique_handles, unique_labels = zip(*unique_handles_labels.items())

    # Add the legend outside the grid
    fig.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.tight_layout()
    plt.show()

# %%
boxprops = {'facecolor':colors.to_rgba('blue', 0.3),
            'edgecolor':colors.to_rgba('black', 1),
            'linewidth':0.4,
            'linestyle':'-',
            'zorder':1
            }

whiskerprops = {'color':'black',
               'linewidth':0.6,
               'linestyle':'-',
               'alpha':1,
               'solid_capstyle':None, #(butt, round, projecting)}
               'zorder': 1
               }

capprops = {'color':'black',
               'linewidth':0.6,
               'linestyle':'-',
               'alpha':1,
               'solid_capstyle':None, #(butt, round, projecting)}
               'zorder': 1
               }

medianprops = {'color':'red',
               'linewidth':1.5,
               'linestyle':'--',
               'alpha':1,
               'zorder': 2
               }

meanprops = {'color':'mediumvioletred',
             'linestyle':':',
             'linewidth':1.9,
             'alpha':1,
             'zorder': 3}

flierprops = {'marker':'x', 
              'markersize':5,
              'markerfacecolor':colors.to_rgba('blue', 1),
              'markeredgecolor':colors.to_rgba('blue', 1),
              'markeredgewidth':0.6,
              'zorder':2
              }

# %%
def histo_with_box(df, features, bins=30, kde=False, box_mean=False, box_meanline=False, boxprops=None, medianprops=None, flierprops=None, meanprops=None):
    
 # Ensure features is a list for uniformity
    if isinstance(features, str):
        features = [features]
    
    # Determine the number of columns and rows for the grid if multiple features
    num_plots = len(features)
    num_cols = 3  # Maximum 3 columns wide
    num_rows = int(np.ceil(num_plots / num_cols))

    # Set up the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten to easily iterate over axes
    
    for i, feature in enumerate(features):
        # Plot histogram
        ax1 = sns.histplot(df, x=df[feature], bins=bins, color='grey', kde=kde, ax=axes[i])
        
        # Overlay boxplot on secondary axis
        ax2 = ax1.twinx()
        sns.boxplot(df, x=df[feature], 
                    ax=ax2,
                    width=0.3,   # Boxplot width relative to histogram width
                    notch=True, 
                    boxprops=boxprops,
                    medianprops=medianprops,
                    flierprops=flierprops,
                    showmeans=box_mean, meanline=box_meanline,
                    meanprops=meanprops)

        ax1.set_title(f'{feature} Distribution')
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Frequency')

    # Hide any unused subplots in the grid
    for j in range(i + 1, num_rows * num_cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



