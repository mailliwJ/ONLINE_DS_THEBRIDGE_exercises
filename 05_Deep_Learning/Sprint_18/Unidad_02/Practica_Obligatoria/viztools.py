# -----------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

import seaborn as sns

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import PowerTransformer

import scipy.stats as stats

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
            ax.axhline(mean, color='green', linestyle='--', label=f'Mean {num_col}: {mean:.2f}')
            ax.legend()

    ax.set_title(f'Boxplot of {num_col} for classes in {cat_col}\nshowing mean (o) and median (-)')
    
    if ax is None:
        plt.show()
    
# -----------------------------------------------------------------------------------------------------------------------------------

def grouped_histograms(df, cat_col, num_col, group_size=None, kde=False, element='step', palette='plasma', alpha=0.5, ax=None):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    
    if group_size is None:
        group_size = num_cats

    cmap = cm.get_cmap(palette, num_cats)

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i + group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]

        for j, cat in enumerate(subset_cats):
            color = cmap(i + j / num_cats) 
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], 
                         kde=kde, element=element,
                         label=str(cat), 
                         color=color, alpha=alpha, ax=ax)

    ax.set_title(f'Histograms of {num_col} for {cat_col} (Grouped)')
    ax.set_xlabel(num_col)
    ax.set_ylabel('Frequency')
    ax.legend()

    # Only call plt.show() if no axis was provided
    if ax is None:
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def plotAll4(data, cat_col='', num_col='', show_mean = False, kde=True, element='step', palette='plasma', hist_color='silver', alpha=1):
    fig, axes = plt.subplots(2,2, figsize=(20,20))

    plot_histogram(data, num_col, kde=True, hist_color=hist_color, alpha=alpha, ax=axes[0,0])
    plot_boxplot(data, num_col, show_mean=show_mean, ax=axes[0,1])
    grouped_histograms(data, cat_col=cat_col, num_col=num_col, kde=kde, element=element, palette=palette, ax=axes[1,0])
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax1 = sns.histplot(df, x=feature, bins=bins, color=hist_color, element=element, kde=kde, ax=ax)
    
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

    ax1.set_title(f'{feature} Distribution')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Frequency')

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
        data = df[feature].dropna()  
        
        mean_val = data.mean()
        median_val = data.median()
        
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

def plot_histogram(df, num_feature, bins=25, kde=False, hist_color='silver', alpha=1, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Plot histogram
    sns.histplot(df, x=num_feature, bins=bins, kde=kde, color=hist_color, alpha=alpha, ax=ax)

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

def plot_boxplot(df, feature, box_color=('blue', 0.4), box_mean=True, show_mean=False, ax=None):
    box_color = colors.to_rgba(box_color[0], box_color[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

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

    if show_mean:
        mean = df[feature].mean()
        ax.axvline(x=mean, color='green', linestyle='--', linewidth=2, label=f'Mean {feature}: {mean:.2f}')
        ax.legend()

    ax.set_title(f'{feature} Boxplot with Mean Line')
    ax.set_xlabel(feature)

    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks([])

    if ax is None:
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

def categorical_relationships(df, cat_col1='', cat_col2='', relative_freq=False, show_values=False, palette='plasma', group_size=5):

    # super important for color continuity. Turns out seaborn treats numbers and text diffferently when they are category classes
    df[cat_col1] = df[cat_col1].astype(str)
    df[cat_col2] = df[cat_col2].astype(str)

    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    miScore = mutual_info_score(df[cat_col1], df[cat_col2])

    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > group_size:
        num_plots = int(np.ceil(len(unique_categories) / group_size))

        for i in range(num_plots):

            categories_subset = unique_categories[i * group_size:(i + 1) * group_size]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=data_subset, x=cat_col1, y='count', order=categories_subset, hue=cat_col2, palette=palette)

            plt.title(f'Relationship between {cat_col1} & {cat_col2}\nMutual Info:{miScore:.5f}')
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
        ax = sns.barplot(data=count_data, x=cat_col1, y='count', hue=cat_col2, palette=palette)

        plt.title(f'Relationship between {cat_col1} & {cat_col2}\nMutual Info:{miScore:.5f}')
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
    sns.scatterplot(data=df, x=x, y=y, s=20)
    plt.title(f'Dispersion Diagram of {x} against {y}')
    plt.grid(True)
    print(f"Correlation: {df[[x, y]].corr().iloc[0, 1]}")

# -----------------------------------------------------------------------------------------------------------------------------------

def combined_plot(df, hist_features=[], hist_bins=25, kde=False, box_mean=True, 
                  hist_color=('silver', 1), box_color=('blue', 0.4), hist_element='bars',
                  scatter_x='', scatter_y='', show_corr=True, scatter_s=20):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if hist_features:
        histobox(df, features=hist_features, bins=hist_bins, kde=kde, box_mean=box_mean, 
                 hist_color=hist_color, box_color=box_color, element=hist_element, axes=[axes[0]])
    
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

def qq_plot_with_shapiro(data, feature_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    stat, p_value = stats.shapiro(data)
    alpha = 0.05
    normality_text = "Fail to reject H0: Data appears normally distributed." if p_value > alpha else "Reject H0: Data does not appear normally distributed."

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot of {feature_name}')

    # Add Shapiro-Wilk test result on the plot
    ax.text(0.95, 0.05, f'Statistic: {stat:.5f}\np-value: {p_value:.5f}\n{normality_text}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7))

    return ax

# -----------------------------------------------------------------------------------------------------------------------------------

def transform_distributions(df, num_col, transformations=('log', 'boxcox', 'sqrt', 'squared', 'yeo-johnson')):
    """Generate histobox and Q-Q plot for various transformations of the data."""
    
    # Prepare the figure layout based on the number of transformations + original
    n_rows = len(transformations) + 1  # +1 for the original data
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))

    if len(transformations) == 1:
        axes = np.expand_dims(axes, axis=0)  # Make sure axes is a 2D array if there's only one transformation

    # Row 1: Original Data
    histobox(df, features=num_col, axes=[axes[0, 0]], kde=True)  # Enabling kde for density estimate
    qq_plot_with_shapiro(df[num_col], feature_name=f'Original {num_col}', ax=axes[0, 1])

    # Initialize the PowerTransformer
    pt_boxcox = PowerTransformer(method='box-cox', standardize=False)
    pt_yeojohnson = PowerTransformer(method='yeo-johnson', standardize=False)

    # Process each transformation
    for i, transform in enumerate(transformations):
        if transform == 'log':
            # Shift data if there are non-positive values
            if (df[num_col] <= 0).any():
                shift_value = abs(min(df[num_col])) + 1
                transformed_data = np.log(df[num_col] + shift_value)
            else:
                transformed_data = np.log1p(df[num_col])
            feature_name = f'Log-transformed {num_col}'
        
        elif transform == 'squared':
            transformed_data = np.square(df[num_col])
            feature_name = f'Squared {num_col}'
        
        elif transform == 'sqrt':
            # Shift data if there are non-positive values
            if (df[num_col] < 0).any():
                shift_value = abs(min(df[num_col])) + 1
                transformed_data = np.sqrt(df[num_col] + shift_value)
            else:
                transformed_data = np.sqrt(df[num_col])
            feature_name = f'Square root of {num_col}'
        
        elif transform == 'boxcox':
            # Box-Cox requires strictly positive values
            if (df[num_col] > 0).all():
                transformed_data = pt_boxcox.fit_transform(df[[num_col]]).flatten()
            else:
                shift_value = abs(min(df[num_col])) + 1
                transformed_data = pt_boxcox.fit_transform((df[num_col] + shift_value).values.reshape(-1, 1)).flatten()
            feature_name = f'Box-Cox Transformed {num_col}'
        
        elif transform == 'yeo-johnson':
            transformed_data = pt_yeojohnson.fit_transform(df[[num_col]]).flatten()
            feature_name = f'Yeo-Johnson Transformed {num_col}'
        
        else:
            print(f"Unknown transformation: {transform}")
            continue

        # Add transformed data to DataFrame for the histobox
        df[f'{transform}_{num_col}'] = transformed_data

        # Histobox for the transformed data (i+1 to account for original data in row 0)
        histobox(df, features=f'{transform}_{num_col}', axes=[axes[i + 1, 0]], kde=True)

        # Q-Q plot with Shapiro-Wilk for the transformed data (i+1 for the same reason)
        qq_plot_with_shapiro(transformed_data, feature_name=feature_name, ax=axes[i + 1, 1])

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------