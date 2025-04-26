import torch.nn as nn
import torch

class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, nb_hidden_layers=1, dropout_rate=0.2):
        super().__init__()
        self.model = nn.Sequential()
        
        self.model.add_module("input", nn.Linear(input_dim, hidden_dim))
        self.model.add_module("actInput", nn.ReLU())
        self.model.add_module("dropoutInput", nn.Dropout(dropout_rate))
        
        for i in range(nb_hidden_layers):
            self.model.add_module(f"hidden{i}", nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module(f"act{i}", nn.ReLU())
            layer_dropout = min(dropout_rate + i * 0.05, 0.5) # higher dropout for deep layers
            self.model.add_module(f"dropout{i}", nn.Dropout(layer_dropout))
            
        self.model.add_module("output", nn.Linear(hidden_dim, 1))
        self.model.add_module("outputAct", nn.Sigmoid())
        
        self.to(torch.float32)
    
    def forward(self, x):
        # Conversion en float32 si nécessaire
        if hasattr(x, 'dtype') and x.dtype != torch.float32:
            x = x.float()
        return self.model(x)