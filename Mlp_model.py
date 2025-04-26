import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from sklearn.model_selection import KFold
from MLP_utils import getPerformanceStatistics, getBestThresholdsForF1
from MLpModuleSkorch import MLPModule

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import itertools
import joblib
import os
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV




class MlpPokemon():



    def __init__(self, X_train, y_train, X_test, y_test, preprocessor, numEpoch: int = 1000, batchSize : int = 32, hiddenDim : int = 16, lr : float = 0.001, nbHiddenLayer : int = 1, dropout: float = 0.2):
        
        # network configuration
        self.preprocessor = preprocessor
        self.numEpochs = numEpoch
        self.dropout = dropout
        self.batchSize : int = batchSize
        self.inputDim : int = X_train.shape[1]
        self.hiddenDim = hiddenDim
        self.outputDim = 1 # because we are performing a binary classification
        self.nbHiddenLayers = nbHiddenLayer
        self.buildNetwork()

        # store original train data
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test

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
        Method to build the neural network
        """

        self.model = nn.Sequential()
        self.model.add_module("input", nn.Linear(self.inputDim, self.hiddenDim))
        self.model.add_module("actInput", nn.ReLU())
        self.model.add_module("dropoutInput", nn.Dropout(self.dropout))
        for i in range(self.nbHiddenLayers):
            self.model.add_module(f"hidden{i}", nn.Linear(self.hiddenDim, self.hiddenDim))
            self.model.add_module(f"act{i}", nn.ReLU())
            layerDropout = min(self.dropout + i * 0.05, 0.5) # higher dropout for deep layers
            self.model.add_module(f"dropout{i}", nn.Dropout(layerDropout))
        self.model.add_module("output", nn.Linear(self.hiddenDim, self.outputDim))
        self.model.add_module("outputAct", nn.Sigmoid())


    def predict(self, X_transformed : np.ndarray = None, yTrue : np.ndarray = None) -> np.ndarray:
        """
        Function to predict the legendary status from a new data set. The features needs to be transformed by using the preprocessor saved with the model

        @returns: total test loss for the prediction and the probability of the classificated data
        """

        if X_transformed is None and yTrue is None:
            outputPrediction = self.__predictingSession(0.0) # use default test value (separated before training)
        
        elif X_transformed is None or yTrue is None:
            raise Exception("Both features and target need to be defined.")
        
        else: # specific data from the user
            originalValLoader = self.valLoader
            valDataSet = TensorDataset(self.forward(X_transformed), self.forward(yTrue))
            self.valLoader = DataLoader(valDataSet, batch_size =  self.batchSize, shuffle = True)
            outputPrediction = self.__predictingSession(0.0)
            self.valLoader = originalValLoader
            

        return outputPrediction






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



    def __predictingSession(self, predictLoss : float) -> tuple[float, list[float]]:
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
    

    def getPredictionClass(self, threshold : float = 0.5, ypred = None):
        """
        Method to return the binary class prediction based on the threshold probability
        default: threshold = 0.5
        """
        ypred = self.yPred if ypred is None else ypred
        return (ypred>=threshold).float().numpy()




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
                'numEpochs': [50, 100],
                "nbHiddenLayers": [1, 2, 3, 5]
            }
        
        # generate all iteration paremeters
        paramCombinaisons =  self.__generateParamCombi(paramGrid)
        
        # to store results
        results = []
        
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
        bestF1 = 0
        for i, params in tqdm(enumerate(paramCombinaisons), total = len(paramCombinaisons), desc = f"Evaluation of {len(paramCombinaisons)} parameter combinaisons..."):
            optiModel = MlpPokemon(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                preprocessor=preprocessor,
                numEpoch=params['numEpochs'],
                batchSize=params['batchSize'],
                hiddenDim=params["hiddenDim"],
                nbHiddenLayer=params["nbHiddenLayers"]
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


    def skorchOptimise(self, paramGrid : dict[list[object]] = None, cv : int =3, nbProces : int = -1):

        # flat if necessary
        if len(self.y_train.shape) > 1 and self.y_train.shape[1] == 1:
            self.y_train = self.y_train.ravel()  # Convertir de (n_samples, 1) à (n_samples,)
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] == 1:
            self.y_test = self.y_test.ravel()  # Faire de même pour y_test

        # convert to float for skorch
        self.X_train = self.X_train.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
            
        network = NeuralNetBinaryClassifier(
            MLPModule,
            module__input_dim = self.X_train.shape[1],
            lr=0.001,
            max_epochs=100,
            batch_size=32,
            iterator_train__shuffle=True,
            optimizer=torch.optim.Adam,
            criterion=nn.BCELoss,
            callbacks=[EarlyStopping(patience=10)],
            )

        if paramGrid is None:
            paramGrid = {
                'lr': [0.001, 0.01, 0.0001],
                'max_epochs': [50, 100],
                'batch_size': [16, 32, 64],
                'module__hidden_dim': [8, 16, 32, 64],
                'module__nb_hidden_layers': [1, 2, 3, 5],
                'module__dropout_rate': [0.1, 0.2, 0.3],
            }

        gs = GridSearchCV(
            network, 
            paramGrid, 
            cv=cv, 
            scoring='f1',
            verbose=2,
            n_jobs= nbProces  # nb processes
        )

        print(f"Starting the oprimisation (GridSearchCV with {cv} folds)...")
        gs.fit(self.X_train, self.y_train)
        
        # display the best results
        print("\n=== Best modele (skorch) ===")
        print(f"Best parameters: {gs.best_params_}")
        print(f"Best F1 score: {gs.best_score_:.4f}")
        
        # save the best model
        best_model = gs.best_estimator_
        model_path = "models/best_skorch_model.pkl"
        preprocessor_path = "models/skorch_preprocessor.pkl"
    
        # create output dir if not exists
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(best_model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        

        return self.generateBestModel(gs, best_model)
        

    def generateBestModel(self, gridSearchResults, bestModel):
    #    Convert skorch variable name to the model ones
        skorch_to_mlp_mapping = {
            'lr': 'learningRate',
            'max_epochs': 'numEpochs',
            'batch_size': 'batchSize',
            'module__hidden_dim': 'hiddenDim',
            'module__nb_hidden_layers': 'nbHiddenLayers',
            'module__dropout_rate': 'dropout'
        }
        # Convert bets params from GridSearch
        best_mlp_params = {}
        for skorch_param, value in gridSearchResults.best_params_.items():
            if skorch_param in skorch_to_mlp_mapping:
                mlp_param = skorch_to_mlp_mapping[skorch_param]
                best_mlp_params[mlp_param] = value
        
        # update the model with the parameter
        self.numEpochs = best_mlp_params.get('numEpochs', 100),
        self.batchSize = best_mlp_params.get('batchSize', 32),
        self.hiddenDim = best_mlp_params.get('hiddenDim', 16),
        self.lr = best_mlp_params.get('learningRate', 0.001),
        self.nbHiddenLayers = best_mlp_params.get('nbHiddenLayers', 1)
        self.buildNetwork()
        self.optimiser = optim.Adam(self.model.parameters(), lr = self.lr)

        
        # perform the prediction and get the results
        _, _, y_pred = self.classicPredict()
        best_threshold = getBestThresholdsForF1(self.y_test, y_pred)
        y_pred_class = self.getPredictionClass(float(best_threshold))
        test_stats = getPerformanceStatistics(self.y_test, y_pred_class)
        
        print("\n=== Model performance on the test data set ===")
        print(f"F1 Score: {test_stats['f1']:.4f}")
        print(f"Accuracy: {test_stats['accuracy']:.4f}")
        print(f"Precision: {test_stats['precision']:.4f}")
        print(f"Recall: {test_stats['recall']:.4f}")
        
        return {
            'best_params': best_mlp_params,
            'best_score': gridSearchResults.best_score_,
            'grid_search_results': gridSearchResults.cv_results_,
            'best_skorch_model': bestModel,
            'best_mlp_model': self.model,
            'test_performance': test_stats
        }