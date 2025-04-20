import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from sklearn.model_selection import KFold
from MLP_utils import getPerformanceStatistics, getBestThresholdsForF1

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import itertools
import joblib

from multiprocessing import Pool



class MlpPokemon():



    def __init__(self, X_train, y_train, X_test, y_test, preprocessor, numEpoch: int = 1000, batchSize : int = 32, hiddenDim : int = 16, lr : float = 0.001):
        
        # network configuration
        self.numEpochs = numEpoch
        self.batchSize : int = batchSize
        self.inputDim : int = X_train.shape[1]
        self.hiddenDim = hiddenDim
        self.outputDim = 1 # because we are performing a binary classification
        self.buildNetwork()

        # store original train data
        self.X_train = X_train
        self.y_train = y_train

        # Set up data for the model
        self.trainDataset = TensorDataset(self.forward(X_train), y_train)
        self.valDataset = TensorDataset(self.forward(X_test), y_test)
        self.trainLoader = DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=True)
        self.valLoader = DataLoader(self.valDataset, batch_size=self.batchSize)

        # Set up loss function and solver 
        self.lossFunction = nn.BCELoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr = lr)




    def buildNetwork(self):
        """
        Method to build the basic neural network
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(self.inputDim, self.hiddenDim),
            nn.ReLU(),
            nn.Linear(self.hiddenDim, self.hiddenDim),
            nn.ReLU(),
            nn.Linear(self.hiddenDim, self.outputDim),
            nn.Sigmoid()
        ).to(device= device)

    # def getCombinedData(self):
    #     """
    #     Merge train and test data to perform a cross validation
        
    #     Returns:
    #         Tensor tuple such as [X_tensor, y_tensor]
    #     """
    #     X_train_tensor = self.trainDataset.tensors[0]
    #     y_train_tensor = self.trainDataset.tensors[1]
    #     X_test_tensor = self.valDataset.tensors[0]
    #     y_test_tensor = self.valDataset.tensors[1]
        
    #     # merging
    #     X_combined = torch.cat([X_train_tensor, X_test_tensor], dim=0)
    #     y_combined = torch.cat([y_train_tensor, y_test_tensor], dim=0)
    
    #     return X_combined, y_combined

    def classicPredict(self):
        """
        Perform a classic prediction with the train data and the test ones
        """
        trainLosses = []
        predictLosses = []
        bestLoss : float = float('inf')
        convergenceThreshold : int = 10 # if loss is not improved after 10 epochs in a row
        stopCounter : int = 0

        for epoch in range(self.numEpochs): # number of optimisation / training session
            # train mode
            trainLoss = 0
            self.model.train() # enable train mode (activate dropout etc)
            trainLoss = self.__trainingSession(trainLoss)
            trainLosses.append(trainLoss) # store the loss to check if it decreases or not

            # test mode
            predictLoss = 0
            self.model.eval() # disable dropout etc (save memory)
            predictLoss = 0
            predictLoss, self.yPred = self.__predictingSession(predictLoss)
            predictLosses.append(predictLoss)

            # early stopping if we reached convergence
            if predictLoss < bestLoss:
                bestLoss = predictLoss
                stopCounter = 0
            else:
                stopCounter += 1
                
            if stopCounter >= convergenceThreshold:
                break
        return trainLosses, predictLosses, self.yPred


    def crossValidationPredict(self, kFold : int = 5):

        kfold = KFold(n_splits=kFold, shuffle=True)
        foldResults = {}

        # X_combined, y_combined = self.getCombinedData() # get X and y data in on structure
        # X_np = X_combined.numpy() if not sp.issparse(X_combined) else X_combined # convert to numpy format


        for fold, (trainIdx, testIdx) in enumerate(kfold.split(self.X_train)): # perform cross validation only on train data (test data will be used for the final evaluation, after optimisation)

            X_TrainFold = self.X_train[trainIdx]
            y_TrainFold = self.y_train[trainIdx]
            X_testFold = self.X_train[testIdx]
            y_testFold = self.y_train[testIdx]
            # create new loader and dataset
            trainDataSet = TensorDataset(self.forward(X_TrainFold), y_TrainFold)
            valDataSet = TensorDataset(self.forward(X_testFold), y_testFold)
            trainLoader = DataLoader(trainDataSet, batch_size = self.batchSize, shuffle=True)
            valLoader = DataLoader(valDataSet,batch_size = self.batchSize)
            # reset network and optimiser
            self.buildNetwork()
            self.optimiser = optim.Adam(self.model.parameters(), lr=0.001)
            # store new train adn value loader
            self.trainLoader, originalTrainLoader = trainLoader, self.trainLoader
            self.valLoader, originalValLoader = valLoader, self.valLoader

            cvTrainLosses, cvValLosses, yPred = self.classicPredict()
            bestThreshold = getBestThresholdsForF1(y_testFold, yPred)
            yCvPredClass = self.getPredictionClass(float(bestThreshold))
            dictStat = getPerformanceStatistics(y_testFold, yCvPredClass)
            foldResults[f"Fold_{fold}"] = dictStat | {"trainLoss": cvTrainLosses, "testLoss": cvValLosses, "bestThreshold" : bestThreshold}
            # reset train loader and value Loader
            self.trainLoader, self.valLoader = originalTrainLoader, originalValLoader
        return foldResults





    def __trainingSession(self, trainLoss) -> float:
        """
        Private method to iterate on train data to optimise the model
        """
        # iterate on train data divided on batches
        for X_TrainBatch, y_TrainBatch in self.trainLoader:
            yPred = self.model(X_TrainBatch) # predict data based on features
            loss = self.lossFunction(yPred, y_TrainBatch) # compute the loss (difference between actual value and the predictions)

            self.optimiser.zero_grad() # reset gradient
            loss.backward()
            self.optimiser.step() # optmisze the teh weights based on the loss
            trainLoss += loss.item()
        
        return trainLoss / len(self.trainLoader)



    def __predictingSession(self, predictLoss : float) -> float:
        """
        Method to compute loss on test dataset to optimise the model
        """
        allPred = []
        with torch.no_grad(): # save memory by avoiding gradient compute
            for X_batch, y_batch in self.valLoader:
                yPred = self.model(X_batch)
                loss = self.lossFunction(yPred, y_batch) # compute loss on test dataset
                predictLoss += loss.item()
                allPred.append(yPred)
        allPred = torch.cat(allPred, dim=0)
        return predictLoss / len(self.valLoader), allPred
    

    def getPredictionClass(self, threshold : float = 0.5):
        """
        Method to return the binary class prediction based on the threshold probability
        default: threshold = 0.5
        """
        return (self.yPred >=threshold).float().numpy()




    def forward(self, x):
        """
        Convert data into pyTorch Tensor if it is not the case
        """
        if not isinstance(x, torch.Tensor):
            if sp.issparse(x):
                x = torch.FloatTensor(x.toarray())
            else:
                x = torch.FloatTensor(x)
        return x
    
    def __generateParamCombi(self, paramGrid : dict[str: list[float]]) -> list[dict[str: float]]:
        """
        Function to generate all parameter combinaisons
        """
        keys = paramGrid.keys()
        values = paramGrid.values()
        paramCombinaisons = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return paramCombinaisons
    

    def optimizeHyperparameters(self, X_train, y_train, X_test, y_test, preprocessor, paramGrid=None, cv=3) -> dict[str : object]:
        """
        Optimise hyperparameter for the MLP. Each combinaison will be evaluated thanks to a cross validation. The best model will be saved.
        You will be able to test it on test data thanks to classicalPrediction method

        @param X_train: arrayLike that contains features data
        @param y_train: arrayLike that contains target data        
        @param X_test: arrayLike that contains features data to predict  (not used in this context, only to init the model)
        @param y_test: arrayLike that contains target data to predict (not used in this context, only to init the model)
        @param preprocessor: the preprocessor used to normalise data and to convert non numeric values
        @param paramgrid: a dictionnary of list to define the area of reserach
            e.g :  paramGrid = {
                            'hiddenDim': [8, 16, 32, 64],
                            'learningRate': [0.001, 0.01, 0.0001],
                            'batchSize': [16, 32, 64],
                            'numEpochs': [50, 100]
                        }
        @param cv: the number of fold to generate (default 3, if you have big dataset you can increase it)

        @returns: a dictionnary with the best parameters and the CV results for each combinaison
    
        """
        # define default grid
        if paramGrid is None:
            paramGrid = {
                'hiddenDim': [8, 16, 32, 64],
                'learningRate': [0.001, 0.01, 0.0001],
                'batchSize': [16, 32, 64],
                'numEpochs': [50, 100]
            }
        
        # generate all iteration paremeters
        paramCombinaisons =  self.__generateParamCombi(paramGrid)
        
        # to store results
        results = []
        bestF1 = -1
        
        # interate on each combinaison
        self.__optimiseParameters(X_train, y_train, X_test, y_test, preprocessor, paramCombinaisons, cv, results)
        
        # extract the best solution
        bestResults = max(results, key=lambda x: x['meanF1'])
        
        print("\n===Best model ===")
        print(f"Parameters: {bestResults['params']}")
        print(f"Mean F-1 score: {bestResults['meanF1']:.4f}")
        print(f"Mean accuracy: {bestResults['meanAccuracy']:.4f}")
        
        return {
            'bestParams': bestResults['params'],
            'bestScore': bestResults['meanF1'],
            "bestResults" : bestResults,
            'allResults': results
        }
    
    def __optimiseParameters(self, X_train, y_train, X_test, y_test, preprocessor, paramCombinaisons, cv, results):
        for i, params in tqdm(enumerate(paramCombinaisons), total = len(paramCombinaisons), desc = f"Evaluation of {len(paramCombinaisons)} parameter combinaisons..."):
            optiModel = MlpPokemon(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                preprocessor=preprocessor,
                numEpoch=params['numEpochs'],
                batchSize=params['batchSize'],
                hiddenDim=params["hiddenDim"]
            )
            
            # perform cross validation for these parameters
            cvResults = optiModel.crossValidationPredict(kFold=cv)
            
            # compute mean performance
            meanF1 = np.mean([results['f1'] for results in cvResults.values()])
            mean_accuracy = np.mean([results['accuracy'] for results in cvResults.values()])
            if meanF1 > bestF1:
                bestF1 = meanF1
                torch.save(optiModel.model.state_dict(), "models/best_model.pth")
                joblib.dump(preprocessor, "models/preprocessor.pkl") # save preprocessor to apply data transformation for real prediction
            # store results for these parameters
            results.append({
                'params': params,
                'meanF1': meanF1,
                'meanAccuracy': mean_accuracy,
                'cvResults': cvResults
            })