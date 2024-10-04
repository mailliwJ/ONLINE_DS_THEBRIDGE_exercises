# -----------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

import seaborn as sns

# -----------------------------------------------------------------------------------------------------------------------------------

def countplot(df, cat_col = '', stat='count', palette='plasma', show_vals=False, ax=None):
    
    if stat == 'count':
        cat_sorted = df[cat_col].value_counts().index
    if stat == 'percent':
        cat_sorted = df[cat_col].value_counts(True).index
    
    if ax is None:
        plt.figure(figsize=(15,7))
        ax = plt.gca()

    sns.countplot(df, x=cat_col, hue=cat_col, order=cat_sorted, palette=palette, legend=False, stat=stat, ax=ax)
    
    if show_vals:
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    ax.set_title(f'Countplot of {cat_col}')

    if ax is None:
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def boxplots(df, cat_col = '', num_col = '', show_mean=False, palette='plasma', ax=None):
    cat_sorted = df[cat_col].value_counts().index
    
    if ax is None:
        plt.figure(figsize=(15,7))
        ax = plt.gca()   
    
    sns.boxplot(df,
                x=cat_col, y=num_col, order=cat_sorted,
                hue=cat_col, palette=palette,
                orient='v',
                notch=True,
                showmeans=True,
                ax=ax,
                meanprops={'marker':'o',
                            'markerfacecolor':'white',
                            'markeredgecolor':'green'},
                medianprops={'color':'white'})
    if show_mean:
            mean = np.mean(df[num_col])
            ax.axhline(mean, color='red', label=f'Mean {num_col}: {mean:.2f}')
            ax.legend()

    ax.set_title(f'Boxplot of {num_col} for classes in {cat_col}\nshowing mean (o) and median (-)')
    
    if ax is None:
        plt.show()
    
# -----------------------------------------------------------------------------------------------------------------------------------

def grouped_histograms(df, cat_col, num_col, group_size=None, kde=False, element='step', palette='plasma', ax=None):
    # Get unique categories
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    
    # Set default group_size to the total number of categories if it's None
    if group_size is None:
        group_size = num_cats

    # Generate colormap for better visualization
    cmap = cm.get_cmap(palette, num_cats)

    # If no axis is provided, create a new figure
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    # Plot histograms for each group of categories based on group_size
    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i + group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]

        for j, cat in enumerate(subset_cats):
            color = cmap(i + j / num_cats)  # Consistent coloring across groups
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], 
                         kde=kde, element=element,
                         label=str(cat), 
                         color=color, alpha=0.3, ax=ax)

    ax.set_title(f'Histograms of {num_col} for {cat_col} (Grouped)')
    ax.set_xlabel(num_col)
    ax.set_ylabel('Frequency')
    ax.legend()

    # Only call plt.show() if no axis was provided
    if ax is None:
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def plotAll4(data, cat_col='', num_col='', show_vals=True, show_mean = False, element='step', palette='plasma'):
    fig, axes = plt.subplots(2,2, figsize=(20,20))

    plot_histogram(data, num_col, kde=True, ax=axes[0,0])
    plot_boxplot(data, num_col, ax=axes[0,1])
    grouped_histograms(data, cat_col=cat_col, num_col=num_col, element=element, palette=palette, ax=axes[1,0])
    boxplots(data, cat_col=cat_col, num_col=num_col, show_mean=show_mean, palette=palette, ax=axes[1,1])

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def histobox2(df, feature, bins=25, kde=False, box_mean=True, hist_color=('silver', 1), 
             box_color=('blue', 0.4), element='bars', ax=None):
    # Ensure feature is a single column since plotAll4 passes one column
    if isinstance(feature, list):
        feature = feature[0]
    
    hist_color = colors.to_rgba(hist_color[0], hist_color[1])
    box_color = colors.to_rgba(box_color[0], box_color[1])
    
    # Create a new axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Plot histogram
    ax1 = sns.histplot(df, x=feature, bins=bins, color=hist_color, element=element, kde=kde, ax=ax)
    
    # Twin axis for the boxplot
    ax2 = ax1.twinx()
    sns.boxplot(df, x=feature, 
                ax=ax2,
                width=0.3,   
                notch=True, 
                boxprops={'facecolor':box_color,
                          'edgecolor':colors.to_rgba('black', 1),
                          'linewidth':0.6,
                          'linestyle':'-',
                          'zorder':1},
                medianprops={'color':'red',
                             'linewidth':1.5,
                             'linestyle':'--',
                             'alpha':1,
                             'zorder': 2},
                flierprops={'marker':'x',
                            'markersize':5,
                            'markerfacecolor':colors.to_rgba('blue', 1),
                            'markeredgecolor':colors.to_rgba('blue', 1),
                            'markeredgewidth':0.6,
                            'zorder':2},
                showmeans=box_mean)

    # Set titles and labels
    ax1.set_title(f'{feature} Distribution')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Frequency')

    # If ax is None (meaning this function created the plot), show the plot
    if ax is None:
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------


def histobox(df, features, bins=25, kde=False, box_mean=True, hist_color=('silver', 1), 
             box_color=('blue', 0.4), element='bars', axes=None):
    if isinstance(features, str):
        features = [features]
    
    hist_color = colors.to_rgba(hist_color[0], hist_color[1])
    box_color = colors.to_rgba(box_color[0], box_color[1])
    
    num_plots = len(features)

    if axes is None:
        if num_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(7, 5))
            axes = [axes]
        else:
            num_cols = 3  
            num_rows = int(np.ceil(num_plots / num_cols))

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax1 = sns.histplot(df, x=df[feature], bins=bins, color=hist_color, element=element, kde=kde, ax=axes[i])
        
        ax2 = ax1.twinx()  
        sns.boxplot(df, x=df[feature], 
                    ax=ax2,
                    width=0.3,   
                    notch=True, 
                    boxprops={'facecolor':box_color,
                              'edgecolor':colors.to_rgba('black', 1),
                              'linewidth':0.6,
                              'linestyle':'-',
                              'zorder':1},
                    medianprops={'color':'red',
                                 'linewidth':1.5,
                                 'linestyle':'--',
                                 'alpha':1,
                                 'zorder': 2},
                    flierprops={'marker':'x',
                                'markersize':5,
                                'markerfacecolor':colors.to_rgba('blue', 1),
                                'markeredgecolor':colors.to_rgba('blue', 1),
                                'markeredgewidth':0.6,
                                'zorder':2},
                    showmeans=box_mean)

        ax1.set_title(f'{feature} Distribution')
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Frequency')

    if axes is not None and num_plots > 1:
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

    if axes is None:
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def histogram_and_boxplot(df, num_col='', bins=20,
                          hist_color=('silver', 1), element='bars', kde=True, kde_color='black',
                          box_color=('blue', 0.4), flier_marker='x', marker_color='white',
                          median_color='red', mean_color = 'yellow'):
    
    hist_color = colors.to_rgba(hist_color[0], hist_color[1])
    box_color = colors.to_rgba(box_color[0], box_color[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(15,8))

    sns.histplot(df, x=df[num_col], bins=bins,
                 element=element, kde=False,
                 line_kws={'color':kde_color}, color=hist_color,
                 ax=axes[0])
    
    if kde:
        ax_kde = axes[0].twinx()
        sns.kdeplot(df[num_col], ax=ax_kde, color=kde_color, linewidth=2)
        ax_kde.spines['right'].set_visible(False)
        ax_kde.get_yaxis().set_ticks([])
        ax_kde.set_ylabel('')


    sns.boxplot(df,
                x=num_col, orient='h',
                showmeans=True, width = 0.4, ax=axes[1],
                boxprops={'facecolor':box_color,
                          'edgecolor':colors.to_rgba('black', 1),
                          'linewidth':0.6,
                          'linestyle':'-',
                          'zorder':1},
                medianprops={'color':median_color,
                             'linewidth':1.5,
                             'linestyle':'--',
                             'alpha':1,
                             'zorder': 2},
                flierprops={'marker':flier_marker,
                            'markersize':5,
                            'markerfacecolor':marker_color,
                            'markeredgecolor':'black',
                            'markeredgewidth':0.6,
                            'zorder':2},
                meanprops={'marker':'o',
                           'markerfacecolor':'white',
                           'markeredgecolor':mean_color})   
    
    mean = np.mean(df[num_col])
    median = np.median(df[num_col])

    axes[1].axvline(mean, color=mean_color, linestyle='--', zorder=3, label=f'Mean {num_col}: {mean:.2f}')
    axes[1].axvline(median, color=median_color, linestyle='--', zorder=3, label=f'Median {num_col}: {median:.2f}')
    axes[1].legend()

    axes[1].spines['left'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].get_yaxis().set_ticks([])

    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def plot_histogram(df, num_feature, bins=25, kde=False, hist_color=('silver', 1), ax=None):
    hist_color = colors.to_rgba(hist_color[0], hist_color[1])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Plot histogram
    sns.histplot(df, x=num_feature, bins=bins, color=hist_color, kde=kde, ax=ax)

    # Set title and labels
    ax.set_title(f'{num_feature} Histogram')
    ax.set_xlabel(num_feature)
    ax.set_ylabel('Frequency')

    # Remove the right spine
    ax.spines['right'].set_visible(False)

    # If ax is None, show the plot
    if ax is None:
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def plot_boxplot(df, feature, box_color=('blue', 0.4), box_mean=True, ax=None):
    box_color = colors.to_rgba(box_color[0], box_color[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the boxplot (horizontal)
    box = sns.boxplot(df, x=feature,
                      ax=ax,
                      width=0.3,
                      boxprops={'facecolor': box_color,
                                'edgecolor': colors.to_rgba('black', 1),
                                'linewidth': 0.6,
                                'linestyle': '-',
                                'zorder': 1},
                      medianprops={'color': 'red',
                                   'linewidth': 1.5,
                                   'linestyle': '--',
                                   'alpha': 1,
                                   'zorder': 2},
                      flierprops={'marker': 'x',
                                  'markersize': 5,
                                  'markerfacecolor': colors.to_rgba('blue', 1),
                                  'markeredgecolor': colors.to_rgba('blue', 1),
                                  'markeredgewidth': 0.6,
                                  'zorder': 2},
                      showmeans=box_mean)

    # Calculate the mean value
    mean_value = df[feature].mean()

    # Draw a vertical line at the mean that spans the width of the box
    ax.axvline(x=mean_value, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_value}')

    # Set title and labels
    ax.set_title(f'{feature} Boxplot with Mean Line')
    ax.set_xlabel(feature)

    # Remove left spine and ticks
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks([])

    # If ax is None, show the plot
    if ax is None:
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def categorical_relationships(df, cat_col1='', cat_col2='', relative_freq=False, show_values=False, palette='plasma', group_size=5):

    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > group_size:
        num_plots = int(np.ceil(len(unique_categories) / group_size))

        for i in range(num_plots):

            categories_subset = unique_categories[i * group_size:(i + 1) * group_size]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            plt.title(f'Relationship between {cat_col1} & {cat_col2} - Group {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frequency' if relative_freq else 'Count')
            plt.xticks(rotation=45)

            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, group_size),
                                textcoords='offset points')

            plt.show()

    else:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, palette=palette, data=count_data)

        plt.title(f'Relationship between {cat_col1} & {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frequency' if relative_freq else 'Count')
        plt.xticks(rotation=45)

        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, group_size),
                            textcoords='offset points')

        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def dispersion(df, x='', y='', s=20):
    sns.scatterplot(data=df, x=x, y=x, s=20)
    plt.title(f'Dispersion Diagram of {x} against {y}')
    plt.grid(True)
    print(f"Correlation: {df[[x, y]].corr().iloc[0, 1]}")

# -----------------------------------------------------------------------------------------------------------------------------------

def combined_plot(df, hist_features=[], hist_bins=25, kde=False, box_mean=True, 
                  hist_color=('silver', 1), box_color=('blue', 0.4), hist_element='bars',
                  scatter_x='', scatter_y='', show_corr=True, scatter_s=20):
    
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Call histobox function on the first axis (axes[0])
    if hist_features:
        histobox(df, features=hist_features, bins=hist_bins, kde=kde, box_mean=box_mean, 
                 hist_color=hist_color, box_color=box_color, element=hist_element, axes=[axes[0]])
    
    # Call dispersion function on the second axis (axes[1])
    if scatter_x and scatter_y:
        sns.scatterplot(data=df, x=scatter_x, y=scatter_y, s=scatter_s, ax=axes[1])
        axes[1].set_title(f'Dispersion Diagram of {scatter_x} against {scatter_y}')
        axes[1].grid(True)
        if show_corr:
            corr = df[[scatter_x, scatter_y]].corr().iloc[0, 1]
            axes[1].text(0.95, 0.95, f'Correlation: {corr:.2f}', ha='right', va='top', 
                        transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def heatmap(df, columns = [], cmap = 'plasma'):
    plt.figure(figsize=(15,8))
    sns.heatmap(df[columns].corr().loc[columns], annot=True, cmap=cmap)
    plt.title(f'HeatMap of selected numeric columns"')

# -----------------------------------------------------------------------------------------------------------------------------------





