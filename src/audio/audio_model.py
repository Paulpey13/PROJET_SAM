import torch
import torch.nn as nn
import torch.nn.functional as F

#Modele AudioCNN    

import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        # Couches de convolutions 1D
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=1)
        
        # Couche de pooling adaptatif pour obtenir une taille fixe
        self.adaptive_pool = nn.AdaptiveAvgPool1d(6)

        # Couches fully connected
        self.fc1 = nn.Linear(32 * 6, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        print(x.shape)
        # Différentes couches de convolutions avec des fonctions d'activation ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)

        # Redimensionnement pour les couches fully connected
        x = x.view(-1, 32 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
