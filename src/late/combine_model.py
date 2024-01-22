import torch
import torch.nn as nn
import torch.nn.functional as F

class Combine_prediction(nn.Module):
    def __init__(self):
        super(Combine_prediction, self).__init__()
        # Définir les couches ici
        self.fc1 = nn.Linear(4, 10)  # 4 caractéristiques en entrée, 10 en sortie
        self.fc2 = nn.Linear(10, 2)  # 10 caractéristiques en entrée, 2 en sortie

    def forward(self,x):
        # Concaténer les vecteurs d'entrée
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
