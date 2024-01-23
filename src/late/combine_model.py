import torch
import torch.nn as nn
import torch.nn.functional as F

class Combine_prediction(nn.Module):
    def __init__(self):
        super(Combine_prediction, self).__init__()
        self.fc1 = nn.Linear(4, 2)  # 4 caractéristiques en entrée, 2 en sortie
        

    def forward(self,x):
        x = self.fc1(x)
 
        return x
