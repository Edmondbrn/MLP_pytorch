from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
import math

from DEFINE import CAT_DATA_STRING


def lossPlot(trainLossData: list[float], predictLossdata: list[float]) -> None:
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # dataframe to handle legend
    lossDf = pd.DataFrame({
        'Epoch': range(1, len(trainLossData) + 1),
        'Train Loss': trainLossData,
        'Validation Loss': predictLossdata
    })
    
    sns.lineplot(x='Epoch', y='Train Loss', data=lossDf, linewidth=2.5, 
                 marker='o', markersize=5, label='Train Loss', color='#1E88E5')
    sns.lineplot(x='Epoch', y='Validation Loss', data=lossDf, linewidth=2.5, 
                 marker='s', markersize=5, label='Validation Loss', color='#FFC107')
    
    plt.title("Loss evolution accross training", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss value", fontsize=12)

    
    # Améliorer la légende
    plt.legend(title='Legend', fontsize=10, title_fontsize=12, 
               frameon=True, facecolor='white', edgecolor='gray')
    

    # minimum annotation
    minTrainIdx = trainLossData.index(min(trainLossData))
    minPredictIdx = predictLossdata.index(min(predictLossdata))
    
    plt.annotate(f'Min: {min(trainLossData):.4f}', 
                xy=(minTrainIdx+1, min(trainLossData)),
                xytext=(10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.annotate(f'Min: {min(predictLossdata):.4f}', 
                xy=(minPredictIdx+1, min(predictLossdata)),
                xytext=(10, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    plt.show()


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
    numerical_cols_to_normalise = [col for col in numerical_cols if "Evolution" not in col]
    # Normalize numerical columns using StandardScaler
    scaler = StandardScaler()
    df[numerical_cols_to_normalise] = scaler.fit_transform(df[numerical_cols_to_normalise])

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
    y_train_tensor = torch.FloatTensor(y_train_smote).view(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)
    
    return X_train_smote, y_train_tensor, X_test_processed, y_test_tensor, preprocessor


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