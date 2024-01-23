import torch
import torch.nn as nn
import torch.nn.functional as F

#Modele pour late (fusionner les prédictions)
class Combine_prediction(nn.Module):
    def __init__(self):
        super(Combine_prediction, self).__init__()
        
        self.fc1 = nn.Linear(4, 2)  
        
    def forward(self, x):
        # Passage des caractéristiques en entrée à travers la couche fully connected
        x = self.fc1(x)
 
        return x
