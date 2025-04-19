from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
import math

from DEFINE import CAT_DATA_STRING



def catPlot(df: pd.DataFrame, catColumn: str, ax : plt.axes):
    """
    Function to create a count plot with seaborn

    @param df: a pandas dataframe
    @param catColumn: string corresponding to the name of the column with the categorical column
    @param ax, plt.axes to put the graph in a subplot
    """
    sns.set_style("whitegrid")
    sns.countplot(data=df, x=catColumn, hue=catColumn, ax=ax)


def numDataDistribution(data : pd.DataFrame, nbPlot : int = 10, nbPlotPerRow : int = 5) -> None:
    """
    Function to plots histogram of numeric values to study the distribution
    """
    nbRow : int = math.ceil((nbPlot / nbPlotPerRow)) # compute the number of rows
    colIndex: int = 0
    colors : tuple = ("red", "blue", "green")
    nbColors : int = len(colors)
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(data.columns[:nbPlot]):  # Mshow only the amount of requested data
        sns.set_style("whitegrid")
        plt.subplot(nbRow, nbPlotPerRow, i+1)
        plt.hist(data[col], color=colors[colIndex])
        plt.title(col)
        colIndex = colIndex + 1 if colIndex +1 < nbColors else 0
    plt.tight_layout()
    plt.show()


def catLegendaryDistribution(data : pd.DataFrame) -> None:
    """
    Function to plots barplot of categorical values to study the distribution
    """
    bar_colors = ["tab:red", "tab:blue"]

    sns.set_style("whitegrid")
    plt.bar(("0", "1"), [
        (data["Legendary Status"] == 0).sum(),
        (data["Legendary Status"] == 1).sum()
    ], color = bar_colors)
    plt.title("Legendary Status")
    plt.tight_layout()
    plt.show()

def catTypingDataDistribution(data: pd.DataFrame, strType: str, title: str = "", 
                             ax=None, nbPlot: int = 10, nbPlotPerRow: int = 5):
    """
    Function to plots barplot of primary Typing values to study the distribution
    
    @param data: DataFrame containing the data
    @param strType: Type to display (key in CAT_DATA_STRING)
    @param title: Title for the plot
    @param ax: Matplotlib axes to plot on (if None, creates new figure)
    """
    # Create ax if not defined
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]
    catCol = []
    catColMinimal = []
    
    for colName in data.columns:
        if colName.startswith(CAT_DATA_STRING[strType][0]):
            catCol.append(colName)
            catColMinimal.append(colName.replace(CAT_DATA_STRING[strType][1], ""))
    
    catTrue = [(data[colName] == 1).sum() for colName in catCol]

    bars = ax.bar(catColMinimal, catTrue, color=bar_colors)
    ax.set_xticklabels(catColMinimal, rotation=90)
    
    if title:
        ax.set_title(title)
    

def multiplePlot(data: pd.DataFrame):
    """
    Create a 3x3 grid of plots showing different type distributions
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # browse key and plot each graph
    for index, strType in enumerate(CAT_DATA_STRING.keys()):
        if index < len(axes):  # avoid ax overflow
            catTypingDataDistribution(data, strType, title=strType, ax=axes[index])
    
    # remove empty plot
    for i in range(len(CAT_DATA_STRING.keys()), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()





def prepareDataForMlp(df: pd.DataFrame):
    """
    Function to convert categorical columns into numeric ones and apply SMOTE processing to normalise legendary and non legendary pokemon
    """
    # Identify num and cat columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != "Legendary Status"]  # Exclure la variable cible

    # for numCol in numerical_cols:
    #     df[numCol] = normalize(df[numCol].to_frame())

    # Create a processor to transform cat values into numeric vector
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols), # ignore numerical data
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) # transform cat value into a numerica vector
        ], remainder='drop')
    

    # Divide data and the target
    X = df.drop("Legendary Status", axis=1)
    y = df["Legendary Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform cat values into num values for non target columns
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # SMOTE to get the same amount of legendary and not legendary pokemon in train dataset (test does not matter)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    return X_train_smote, y_train_smote, X_test_processed, y_test, preprocessor


def encodedDataToDataFrame(preprocessor : ColumnTransformer, data):
    """
    Function to extract compressed data columns after an ColumnTransformer to get a dataFrame
    """
    feature_names = preprocessor.get_feature_names_out() # get feature name for the table
    data_dense = data.toarray() if sp.issparse(data) else data # check if the data is compressed or not
    data_df = pd.DataFrame(data_dense, columns=feature_names)
    return data_df


# Pour conserver le préprocesseur pour une utilisation ultérieure (important!)
# from sklearn.pipeline import Pipeline
# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', your_model)  # Votre modèle MLP ici
# ])