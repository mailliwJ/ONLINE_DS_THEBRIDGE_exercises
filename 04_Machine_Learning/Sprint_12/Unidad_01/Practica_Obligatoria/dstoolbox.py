# %% [markdown]
# # Data Science Tool Kit
# Author: Will Johson  
# Contact: jmailliwj@gmail.com  
# 
# ---
# The following functions have proved highly for Data Science projects from EDAs to ML model design  
# 
# ---
# 

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import seaborn as sns

from scipy.stats import pearsonr, f_oneway, mannwhitneyu, norm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# %%
def describe_df(df):
    """
    Generates a brief summary of the DataFrame's features
    
    Parameters:
    df (pd.DataFrame): DataFrame to be described and summarised.

    Returns:
    pd.DataFrame ('df_temp'): Transposed DataFrame of summary statistics for each column and rows representing:
        - DATA_TYPE: Data type of each column
        - MISSINGS (%): Percentage of missing values in each column
        - UNIQUE_VALUES: Number of unique values in each column
        - CARDIN (%): Cardinality (unique values as a percentage of total rows)
    """
    #1. Prepare data on all columns:
    #  Get data types
    DATA_TYPE = df.dtypes
    # Calculate percentage of missing values in each col
    MISSINGS = df.isna().sum()/len(df) * 100
    # Count number of unique values in each col
    UNIQUE_VALUES = df.nunique()
    # Calculate percentage cardinality for each col
    CARDIN = UNIQUE_VALUES / len(df) * 100
    
    # Make the DataFrame
    df_temp = pd.DataFrame({
        'DATA_TYPE': DATA_TYPE,
        'MISSINGS (%)': MISSINGS,
        'UNIQUE_VALUES': UNIQUE_VALUES,
        'CARDIN (%)': round(CARDIN, 2)
    })

    # Transpose DataFrame for better readability
    df_temp = df_temp.T

    # Return the transposed DataFrame
    return df_temp

# %%
def tipify_variables(df, categoric_threshold, continuous_threshold):
    
    """
    Suggests variable types for each column in a DataFrame based on cardinality 
    and percentage cardinality of each column, and specified thresholds for defining
    binary, categoric, numeric continuous and numeric discrete variables.

    Parameters:
    df (pd.DataFrame): DataFrame containing variables to be classified.
    categoric_threshold (int, inclusive): Threshold for determining whether a variable is categorical or numerical.
    continuous_threshold (float, percentage, inclusive): Threshold (percentage) for determining whether a variable is continuous or discrete
            based on percentage of unique values.
    
    Returns:
    pd.DataFrame ('df_temp'): DataFrame with columns 'nombre_variable' and 'tipo_sugerido'
    """

    # Initialize a dictionary to store col names and suggested variable types
    types_dict = {}

    # Loop through columns in the DataFrame
    for col in df.columns:
        # Calculate cardinality for each column
        cardinality = df[col].nunique()
        # Calculate percentage cardinality for each column
        percentage_cardinality = cardinality / len(df) * 100

        # Classify each variable based on cardinality and percentage cardinality
        # Binary variables: 2 unique values
        if cardinality == 2:
            tipo = 'Binary'
        # Categorical variables: unique values less than or equal to 'categoric_threshold'
        elif cardinality <= categoric_threshold:
            tipo = 'Categorical'
        # Classify numeric variables: unique values greater than 'categoric_threshold'
        elif cardinality > categoric_threshold:
            # Numeric continuous: percentage cardinality greater than or equal to 'continuous_threshold'
            if percentage_cardinality >= continuous_threshold:
                tipo = 'Numerical Continuous'
            # Numeric discrete: percentage cardinality below 'continuous_threshold'
            else:
                tipo = 'Numerical Discrete'

        # Store proposed variable types in the dictionary 'types_dict' with column names as keys
        types_dict[col] = tipo

    # Create a DataFrame from the 'types_dict' dictionary with two columns: 'nombre_variable' and 'tipo_sugerido'
    df_temp = pd.DataFrame(types_dict.items(), columns=['nombre_variable', 'tipo_sugerido'])

    # Return the DataFrame
    return df_temp

# %%
# from scipy.stats import pearsonr

def get_num_features_regression(df, target_col, corr_threshold, pvalue=None, card=20):
    
    """
    Identifies and evaluates the correlation between numeric columns in a DataFrame and a specified target column.
    Stores and returns a list of columns that have an absolute Pearson correlation stat greater than a specified threshold ('corr_threshold').
    If a p-value is specified (pvalue) then this is used to check correlations for statistical signifcance and this is accounted for in column selection.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    target_col (str): Target column to correlate with numeric columns.
    corr_threshold (float): Correlation threshold (between 0 and 1) for the correlation test.
    pvalue (float, optional, Defaul=None): Signifance level (between 0 and 1) for the correlation test.
    card (int): Cardinality threshold checks for sufficient unique values in 'target_col'


    Returns:
    list ('features_num'): A list of columns that have correlated with target column above the specified threshold 'corr_threshold'
    """

    # First carry out checks to prevent errors
    #1. check df is a dataframe
    if not isinstance(df, pd.DataFrame):
        print("First arguement must be a Pandas DataFrame.")
        return None
    
    #2. check target_col is in df
    if target_col not in df.columns:
        print(f"The column '{target_col}' is not in the the specified DataFrame.")
        return None
    
    #3. check target_col is numeric and continuous (high cardinality)
    # https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
    if not (pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > card):
        print(f"The column '{target_col}' must be a continuous numeric variable with high cardinality. \nCheck the 'card' value.")
        return None
    
    # Check corr_threshold is float between 0 and 1 (and not (0 <= corr_threshold => 1)
    if not isinstance(corr_threshold, (int, float)) or not (0 <= corr_threshold <= 1):
        print("'corr_threshold' must be a number between 0 and 1.")
        return None
    
    # Check pvalue is float between 0 and 1 (and not (0 <= pvalue => 1)
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("'pvalue' must be 'None' or a number between 0 and 1.")
            return None
        
    #2. Initialize features list to store selected numeric features
    features_num = []

    #3. Loop over all numeric columns in the dataframe
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
    for col in df.select_dtypes(include=[float, int]).columns:
        if col == target_col:
            continue

        # Calculate pearsonr corr stat and p_value
        corr, p_val = pearsonr(df[col], df[target_col])

        # Check corr stat is greater than 'corr_threshold'
        # Convert to absolute value to avoid problems with negative correlations
        if abs(corr) > corr_threshold:
            if pvalue is None:
                features_num.append(col)
            elif p_val <= 1 - pvalue:
                features_num.append(col)

    # Return the selected numeric columns list 'features_num'
    return features_num

# %%
def plot_num_features_regression(df, target_col="", card=20, columns=[], corr_threshold=0, pvalue=None):

    """
    Generates pair plots for numeric columns in a DataFrame based on their correlation with a specified target column.
    Pair plots are generated in maximum 5x5 grids.
    If specific numeric columns are not specified the function will filter the numeric columns in the DataFrame based on
    a specified correlation threshold ('corr_threshold') and optionally a p-value significance level.
    Checks the threshold conditions of specified columns and offers options to remove if columns are not valid or continue
    anyway.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing data.
    target_col (str): The target column to correlate with other numeric columns. Must be numeric continuous variable with high cardinality.
    card (int): Cardinality threshold checks for sufficient unique values in 'target_col'
    corr_threshold (float): Correlation threshold (between 0 and 1) for correlation testing if numeric columns are not specified.
    pvalue (float, optional, Defaul=None): Signifance level (between 0 and 1) for the correlation testing if numeric columns are not specified.

    Returns:
    list ('columns'): List of columns used for generating the pair plots
    """

    # First carry out checks to prevent errors
    #1. Check df is a dataframe
    if not isinstance(df, pd.DataFrame):
        print('First arguement must be a Pandas DataFrame.')
        return None

    #2. Check target_col is in DataFrame, and is numeric and continuous (high cardinality)
    if target_col not in df.columns or not (pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > card):
        print(f"The target column ('{target_col}') must be a numeric continuous variable with high cardinality.\nCheck 'card' value")
        return None
    
    #3. Check pvalue is float between 0 and 1 (and not (0 <= pvalue => 1)
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("'pvalue' must be 'None' or a number between 0 and 1.")
            return None

    # If no numeric columns are specified, get columns using function 'get_features_num_regression()' based on 'corr_threshold' and 'pvalue'
    if not columns:
        columns = get_num_features_regression(df=df, target_col=target_col, corr_threshold=corr_threshold, pvalue=pvalue)
    else:
        valid_cols = [] # Create empty list to store columns that meet threshold conditions
        for col in columns: # Loop through columns in columns list
            if col == target_col:
                continue # Skip the target column itself as already been checked for validity

            # Calculate pearsonr corr stat and p_value between column and target column
            corr, p_val = pearsonr(df[col], df[target_col])

            # Check corr stat and p-value meet specified thresholds
            if abs(corr) > corr_threshold:
                if pvalue is None or p_val <= pvalue:
                    valid_cols.append(col) # add column to valid_cols list if it meets both thresholds
                else:
                    # Warn that column does not meet the required p-value significance level
                    print(f"'{col}' did not meet the p-value signifcance level")
                    # Ask if you want to remove the column or continue anyway
                    question = input(f"Do you want to remove '{col}' from the columns list or continue anyway? Type 'remove' or 'continue'").strip().lower()

                    if question == 'continue': 
                        valid_cols.append(col) # adds column to valid_cols list if user types continue
                    else:
                        print(f"'{col}' was removed from columns list")
                        continue
            
            else:
                # Warn that column does not meet the required correlation threshold
                print(f"'{col}' did not meet the correlation threshold of {corr_threshold}.")
                # Ask if you want to remove the column or continue anyway
                question = input(f"Do you want to remove '{col}' from the columns list or continue anyway? Type 'remove' or 'continue'").strip().lower()
                if question == 'continue':
                    valid_cols.append(col) # adds column to valid_cols list if user types continue
                else:
                    print(f"'{col}' was removed from columns list")
                    continue
        
        if valid_cols: # Check there are still valid columns left in valid_cols
            columns = valid_cols # Sets columns to valid_columns after checks and warnings
        else:
            columns = get_num_features_regression(df=df, target_col=target_col, corr_threshold=corr_threshold, pvalue=pvalue)

    columns = [col for col in columns if col != target_col] # Make sure target is not in columns list to plot
    print(f"columns selected for pair plot analysis were: {columns}")
    
    # Generate pair plots in max 5x5 grids
    for i in range(0, len(columns), 4):
        sns.pairplot(df, vars=[target_col] + columns[i:i + 4])
        plt.show()

    # Return the selected numeric columns list 'columns'
    return columns

# %%
# from scipy.stats import f_oneway, mannwhitneyu
# possibile addition? --> see comment marked ???
def get_cat_features_regression(df, target_col, pvalue=0.05, card=20):
    """
    Identifies and evaluates the significance of relationship between
    categorical columns and a specified numeric target column in a DataFrame.
    Uses ANOVA for multi-cats or Mann_whitney U for binary-cats
    Stores and returns a list of columns that have show a significant 
    relationship with target column based on spcifed (optionally) pvalue.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data

    target_col : str
        Numeric target column for testing relationship with categorical columns
    
    pvalue : float, optional (default=0.05)
        Significance level (between 0 and 1) for statistical test evaluation.
    
    card : int (default=20)
        Cardinality threshold (based on unique values) to determine if a column should be considered categorical.

    Returns:
    categorical_features : list
        A list of categorical columns that have a significant relationship with target column based on pvalue arguement.
    """
    # Carry out input data checks
    #1. Check df is a dataframe
    if not isinstance(df, pd.DataFrame):
        print('First arguement must be a Pandas DataFrame.')
        return None
    
    #2. Check target column in DataFrame
    if not target_col in df.columns:
        print(f"The target column ('{target_col}') must be in the DataFrame.")
        return None
    
    # Check target column is numeric and has sufficiently high cardinality
    if not (pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > card):
        print(f"The target column ('{target_col}') must be a numeric continuous variable with high cardinality.\nCheck 'card' value")

    # Check pvalue is float between 0 and 1 (and not (0 <= pvalue => 1)
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("'pvalue' must be a 'None' or a number between 0 and 1.")
            return None
    
    # Create empty list to store columns considered to have statistically significant relationship with target column
    categorical_features = []

    # Loop through each column in the DataFrame
    for col in df.columns:
        if col == target_col: # Skip target column itself
            continue
        
        # Check the cardinality of column to decide if categorical or not
        if len(df[col].unique()) <= card: #??? Could we add if 'df[col].dtype == 'object' to this if?
            # If categorical and binary perform Mann-Whitney U test
            if df[col].nunique() == 2:
                groupA = df[df[col] == df[col].unique()[0]][target_col]
                groupB = df[df[col] == df[col].unique()[1]][target_col]

                p_val = mannwhitneyu(groupA, groupB).pvalue
            
            else:
                # If categorical with more than 2 groups, perform ANOVA test
                groups = df[col].unique()
                target_by_groups = [df[df[col] == group][target_col] for group in groups]

                p_val = f_oneway(*target_by_groups).pvalue

            # Check p-val against pvalue arguement to see if significance threshold is met
            if p_val <= pvalue:
                categorical_features.append(col) # Add to categorical_features list if deemed significant

    # Return list of categorical features
    return categorical_features

# %%
def plot_histogram_kde_with_std(df, column, bins=30, kde=False, normal_distribution=False, mean=False, median=False, print_measures=False):
    
    """
    Plots a histogram of the specified column from a DataFrame with optional KDE overlay and shaded areas 
    representing standard deviation ranges. The function allows customization of mean and median lines 
    as well as the normal distribution curve with its shaded standard deviation areas.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.

    column : str
        The column name in the DataFrame to be plotted.
    
    bins : int, optional (default=30)
        Number of bins for the histogram.
    
    kde : bool, optional (Default=False) 
        If True, overlay a Kernel Density Estimate (KDE) line on the histogram. 
    
    normal_distribution : bool, optional (default=False)
        If True, plot a normal distribution curve with shaded areas for 
        1, 2, and 3 standard deviations. 
    
    mean : bool, optional (default=False) 
        If True, adds a vertical line for the mean of the data.
    
    median : bool, optional (default=False) 
        If True, add a vertical line for the median of the data. 
    
    print_measures : bool, optional (default=False) 
        If True, print skewness and kurtosis of the data. 

    Returns:
    --------
    - None: Displays the plot and optionally prints statistical measures.
    """
    
    # Extract the column data
    data = df[column]
    
    # Calculate mean, median, and standard deviation
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot the histogram
    ax1.hist(data, bins=bins, color='grey', alpha=0.6, edgecolor='black', label='Histogram (Counts)')
    ax1.set_ylabel('Frequency Count')

    # Set up secondary y-axis for KDE and Normal Distribution
    ax2 = ax1.twinx()
    ax2.set_ylabel('Density')
    
    handles2 = []
    labels2 = []

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

            # Shade areas and get unique labels
            labels_shaded = set()
            for lower_bound, upper_bound, color, label in shading_info:
                if lower_bound < upper_bound:
                    ax2.fill_between(x, 0, y, where=(x > lower_bound) & (x < upper_bound), color=color, alpha=0.3)
                    labels_shaded.add((color, label))

            # Add legend patches to shaded areas
            patches = []
            for color, label in labels_shaded:
                patch = Patch(color=color, alpha=0.3, label=label)
                patches.append((label, patch))

            # Sort patches by the standard dev ranges in numerical order
            patches.sort(key=lambda x: int(x[0][0]))

            # Separate handles and labels after sorting
            if patches:
                labels2, handles2 = zip(*patches)
            else:
                labels2 = []
                handles2 = []

        ax2.set_ylim(0, ax2.get_ylim()[1])

    # Add third axis for mean and median lines
    ax3 = ax1.twiny()  # Create a third axis sharing the x-axis with ax1
    ax3.set_xlim(ax1.get_xlim())  # Ensure it has the same x limits
    
    # Hide the top axis by making it transparent and removing ticks and labels
    ax3.xaxis.set_visible(False)
    ax3.spines['top'].set_color('none')  # Hide the top spine
    ax3.spines['bottom'].set_color('none')  # Hide the bottom spine on the twiny axis

    # Add vertical lines for mean and median on ax3
    if mean:
        ax3.axvline(mean_value, color='red', linestyle=':', linewidth=3, label='Mean')
    if median:
        ax3.axvline(median_value, color='indigo', linestyle=':', linewidth=3, label='Median')

    handles1, labels1 = ax1.get_legend_handles_labels()

    # Combine graphical element and labels for legend
    handles = handles1 + list(handles2)
    labels = labels1 + list(labels2)

    # Ensure the mean and median appear on top
    handles += list(ax3.get_legend_handles_labels()[0])
    labels += list(ax3.get_legend_handles_labels()[1])

    # Sort legend by the standard deviation range in numerical order
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: (x[1].find('σ') != -1, x[1]))
    handles, labels = zip(*sorted_handles_labels) if sorted_handles_labels else ([], [])

    # Add legend next to the plot using handles and lavels
    fig.legend(handles, labels, bbox_to_anchor=(1.17, 0.9))

    # Print skewness and kurtosis if required
    if print_measures:
        print(f'{df[column].name} Skewness: {data.skew():.4F}')
        print(f'{df[column].name} Excess Kurtosis: {round(data.kurt()-3,4)}')

    # Add title and labels
    plt.title(f'{column} Distribution')
    ax1.set_xlabel(column)
    
    # Show the plot
    plt.show()

# %%
def plot_distribution_categorics(df, cat_cols, rel_values=False, show_values=False):
    number_cols = len(cat_cols)
    number_rows = (number_cols // 2) + (number_cols % 2)

    fig, axes = plt.subplots(number_rows, 2, figsize=(15, 5 * number_rows))
    axes = axes.flatten() 

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        if rel_values:
            total = df[col].value_counts().sum()
            series = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=series.index, y=series, ax=ax, palette='viridis', hue = series.index, legend = False)
            ax.set_ylabel('Frequency rel_values')
        else:
            series = df[col].value_counts()
            sns.barplot(x=series.index, y=series, ax=ax, palette='viridis', hue = series.index, legend = False)
            ax.set_ylabel('Frequency')

        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if show_values:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, number_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
# %%
def show_coefs(model, figsize=(12, 8), color=('blue', 1.0)):
    """
    Plot the coefficients and absolute coefficients of a linear model.

    Parameters:
    -----------
    model : object
        The linear model from which to extract the coefficients. The model must have a `coef_` attribute and a `feature_names_in_` attribute.
    
    figsize : tuple, optional (default=(12, 8))
        The size of the figure to be created, specified as (width, height) in inches.

    color : tuple, optional (default=('blue', 1.0))
        A tuple specifying the color and transparency (alpha) of the bars.
        The first element is the color name (e.g., 'blue'), and the second element is the alpha value (e.g., 0.6).

    Returns:
    --------
    df_coef : pandas.DataFrame
        A DataFrame containing the coefficients of the model, indexed by the feature names.

    Example:
    --------
    models = [lin_reg, ridgeR, lassoR, elasticnetR]
    for model in models:
        show_coefs(model, color=('indigo', 0.6))
    """
    
    # Create a DataFrame from the model coefficients
    df_coef = pd.DataFrame(model.coef_, index=model.feature_names_in_, columns=["coefs"])

    # Create the figure and subplots
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot the model coefficients
    df_coef.plot(kind="barh", ax=ax[0], legend=False, color=color[0], alpha=color[1])
    ax[0].set_title("Model Coefficients")

    # Plot the absolute values of the coefficients
    df_coef.abs().sort_values(by="coefs").plot(kind="barh", ax=ax[1], legend=False, color=color[0], alpha=color[1])
    ax[1].set_title("Absolute Values of Coefficients")
    
    # Set the title for the entire figure
    fig.suptitle(f"{model} Model Coefficients")

    # Adjust the layout
    fig.tight_layout()

    return df_coef


