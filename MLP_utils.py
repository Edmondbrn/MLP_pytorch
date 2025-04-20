from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve

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
        "Epoch": range(1, len(trainLossData) + 1),
        "Train Loss": trainLossData,
        "Validation Loss": predictLossdata
    })
    
    sns.lineplot(x="Epoch", y="Train Loss", data=lossDf, linewidth=2.5, 
                 marker="o", markersize=5, label="Train Loss", color="#1E88E5")
    sns.lineplot(x="Epoch", y="Validation Loss", data=lossDf, linewidth=2.5, 
                 marker="s", markersize=5, label="Validation Loss", color="#FFC107")
    
    plt.title("Loss evolution accross training", fontsize=16, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss value", fontsize=12)

    
    # Améliorer la légende
    plt.legend(title="Legend", fontsize=10, title_fontsize=12, 
               frameon=True, facecolor="white", edgecolor="gray")
    

    # minimum annotation
    minTrainIdx = trainLossData.index(min(trainLossData))
    minPredictIdx = predictLossdata.index(min(predictLossdata))
    
    plt.annotate(f"Min: {min(trainLossData):.4f}", 
                xy=(minTrainIdx+1, min(trainLossData)),
                xytext=(10, -20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.annotate(f"Min: {min(predictLossdata):.4f}", 
                xy=(minPredictIdx+1, min(predictLossdata)),
                xytext=(10, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.tight_layout()
    plt.show()


def rocCurve(yTrue, yPred):
    """
    Function to plot the ROC curve of the MLP output
    It will plot the false positive rate against the true positive rate
    """
    fpr, tpr, thresholds = roc_curve(yTrue.numpy(), yPred.numpy())
    
    df = pd.DataFrame({
        "False positive rate": fpr,
        "True positive rate": tpr,
        "thresholds": thresholds
    })

    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
    return df



def precisionRecallCurve(yTrue, yPred):
    """
    Function to plot the precision against the recall to determine the best thresholds
    """
    precision, recall, thresholds = precision_recall_curve(yTrue.numpy(), yPred.numpy())
    precision =  precision[:-1]
    recall =  recall[:-1]
    
    df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "thresholds": thresholds
    })
    
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    sns.lineplot(data = df, x = "Recall", y = "Precision", marker = "o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
    return df






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
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numerical_cols = [col for col in numerical_cols if col != "Legendary Status"]  # Exclure la variable cible
    numerical_cols_to_normalise = [col for col in numerical_cols if "Evolution" not in col]
    # Normalize numerical columns using StandardScaler
    scaler = StandardScaler()
    df[numerical_cols_to_normalise] = scaler.fit_transform(df[numerical_cols_to_normalise])

    # Create a processor to transform cat values into numeric vector
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols), # ignore numerical data
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols) # transform cat value into a numerica vector
        ], remainder="drop")
    

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


def getPerformanceStatistics(yTrue : np.ndarray | pd.Series, yPred : np.ndarray) -> dict[str : float]:
    """
    Function to compute important statistic after MLP prediction
    
    @return: a dictionnary with accuracy, precision, recall, f1 and confusion matrix
    """
    accuracy = accuracy_score(yTrue, yPred) # number of correct prediction
    precision = precision_score(yTrue, yPred) # true positive ratio in prediction class
    recall = recall_score(yTrue, yPred) # proportion of legendary pokemon correctly identified
    f1 = f1_score(yTrue, yPred) # mean between precision and recal
    confusion = confusion_matrix(yTrue, yPred)
    return {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
        "confusion" : confusion
    }

def getBestThresholdsForF1(yTest, yPred) -> float:
    """
    Function to get the best thresholds to optimise F1 value (mean between recall and precision)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for threshold in thresholds:
        y_pred_t = (yPred >= threshold).float().numpy()
        f1_scores.append(f1_score(yTest.numpy(), y_pred_t))
    return thresholds[np.argmax(f1_scores)]



def encodedDataToDataFrame(preprocessor : ColumnTransformer, data):
    """
    Function to extract compressed data columns after an ColumnTransformer to get a dataFrame
    """
    feature_names = preprocessor.get_feature_names_out() # get feature name for the table
    data_dense = data.toarray() if sp.issparse(data) else data # check if the data is compressed or not
    data_df = pd.DataFrame(data_dense, columns=feature_names)
    return data_df


def formatRawCvOutput(dictFoldResults : dict):
    """
    Function to get all the important information about each fold after a cross validation

    @param dictFoldResults: dictionnary from MlpPokemon.crossValidationPredict
    """
    for fold in dictFoldResults.keys():
        output = dictFoldResults[fold]
        print(f"Fold number {fold}:\nAccuracy: {output["accuracy"]}\n Precision: {output["precision"]}\n Recall: {output["recall"]}\n F1: {output["f1"]}\n Confusion Matrix: {output["confusion"]}\n Final test loss: {output["testLoss"][-1]}")


def analyseCvOutput(foldResults : dict) -> pd.DataFrame:
    """
    Method to compute all important performance data after a cross validation
    
    @param foldResults: dictionnary from MlpPokemon.crossValidationPredict
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    resultDf = pd.DataFrame(columns=metrics)
    
    for fold, results in foldResults.items(): # store metrics for each fols in a dataFrame
        resultDf.loc[fold] = [results[metric] for metric in metrics]
    
    meanScores = resultDf.mean()
    stdScores = resultDf.std()
    
    print("=== Cross validation results===")
    for metric in metrics:
        print(f"{metric.capitalize()}: {meanScores[metric]:.4f} ± {stdScores[metric]:.4f}")
    
    return resultDf


def plotCvLearningCurves(foldResults : dict):
    """
    Function to plot the loss curves for each fold

    @param foldResults: dictionnary from MlpPokemon.crossValidationPredict
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    sns.set_style("whitegrid")
    for fold, results in foldResults.items():
        sns.lineplot(results["trainLoss"], label=fold)
    plt.title("Train loss for each fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for fold, results in foldResults.items():
        plt.plot(results["testLoss"], label=fold)
    plt.title("Test loss for each fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plotConfusionMatrices(foldResults):
    """
    Function to plot the confusion matrix for each fold

    @param foldResults: dictionnary from MlpPokemon.crossValidationPredict
    """
    nFolds = len(foldResults)
    fig, axes = plt.subplots(1, nFolds, figsize=(5*nFolds, 4))
    
    for i, (fold, results) in enumerate(foldResults.items()):
        ax : plt.axes = axes[i] if nFolds > 1 else axes
        sns.heatmap(results["confusion"], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Matrice de confusion - {fold}")
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")
        ax.set_xticklabels(["Non-Légendaire", "Légendaire"])
        ax.set_yticklabels(["Non-Légendaire", "Légendaire"])
    
    plt.tight_layout()
    plt.show()


def plotMetricsComparison(metricsDf : pd.DataFrame) -> None:
    """
    Function to plot the metrics for each folds

    @param metricsDf, dataframe from analyseCvOutput
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    metricsDf.plot(kind="bar", figsize=(10, 6))
    plt.title("Metric comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

def completeCvAnalysis(cvResults):
    """
    Global function to launch all the analysises after a cross validation

    @param cvResults: dict from MlpPokemon.crossValidationPredict
    """
    # gloabl analysis
    metrics_df = analyseCvOutput(cvResults)
    
    # loss curves
    plotCvLearningCurves(cvResults)
    
    # Confusion matrix
    plotConfusionMatrices(cvResults)
    
    # compare metrics between folds
    plotMetricsComparison(metrics_df)
    
    # best thresholds analysis
    thresholds = [float(cvResults[f]["bestThreshold"]) if "bestThreshold" in cvResults[f] else 0.5 
                  for f in cvResults]
    print(f"Seuil moyen: {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}")




