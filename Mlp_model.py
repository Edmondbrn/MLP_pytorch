import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim

import scipy.sparse as sp
from tqdm import tqdm


class MlpPokemon():



    def __init__(self, X_train, y_train, X_test, y_test, preprocessor, numEpoch: int = 1000):
        
        # network configuration
        self.numEpochs = numEpoch
        self.batchSize : int = 32
        self.inputDim : int = X_train.shape[1]
        self.hiddenDim = 16
        self.outputDim = 1 # because we are performing a binary classification
        self.buildNetwork()

        # Set up data for the model
        self.trainDataset = TensorDataset(self.forward(X_train), y_train)
        self.valDataset = TensorDataset(self.forward(X_test), y_test)
        self.trainLoader = DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=True)
        self.valLoader = DataLoader(self.valDataset, batch_size=self.batchSize)

        # Set up loss function and solver 
        self.lossFunction = nn.BCELoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.001)




    def buildNetwork(self):
        """
        Method to build the basic neural network
        """

        self.model = nn.Sequential(
            nn.Linear(self.inputDim, self.hiddenDim),
            nn.ReLU(),
            nn.Linear(self.hiddenDim, self.hiddenDim),
            nn.ReLU(),
            nn.Linear(self.hiddenDim, self.outputDim),
            nn.Sigmoid()
        )

    def classicPredict(self):
        trainLosses = []
        predictLosses = []
        bestLoss : float = float('inf')
        convergenceThreshold : int = 10 # if loss is not improved after 10 epochs in a row
        stopCounter : int = 0

        for epoch in tqdm(range(self.numEpochs)): # number of optimisation / training session
            # train mode
            trainLoss = 0
            self.model.train() # enable train mode (activate dropout etc)
            trainLoss = self.__trainingSession(trainLoss)
            trainLosses.append(trainLoss) # store the loss to check if it decreases or not

            # test mode
            predictLoss = 0
            self.model.eval() # disable dropout etc (save memory)
            predictLoss = 0
            predictLoss, yPred = self.__predictingSession(predictLoss)
            predictLosses.append(predictLoss)

            # early stopping if we reached convergence
            if predictLoss < bestLoss:
                bestLoss = predictLoss
                torch.save(self.model.state_dict(), "models/best_model.pth")
                stopCounter = 0
            else:
                stopCounter += 1
                
            if stopCounter >= convergenceThreshold:
                print(f"Early stopping after {epoch+1} epochs")
                break
        return trainLosses, predictLosses, yPred




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