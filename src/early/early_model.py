import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CamembertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CamembertModel

"""class EarlyFusionModel(nn.Module):
    def __init__(self, text_feature_size=768, rnn_hidden_size=192):
        super(EarlyFusionModel, self).__init__()
        
        # CamemBERT pour les caractéristiques textuelles
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        
        # Couches de convolution 1D pour les caractéristiques audio
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)

        # Vous pouvez ajuster la taille de sortie ici selon la taille de vos données
        self.adaptive_pool =  nn.AdaptiveAvgPool2d((1, 6))

        # GRU pour traiter les caractéristiques combinées
        self.rnn = nn.GRU(text_feature_size + 32 * 6, rnn_hidden_size, batch_first=True)
        
        # Couche de sortie
        self.fc1 = nn.Linear(rnn_hidden_size, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, audio_features, text_input_ids, text_attention_mask):
        # Traitement des caractéristiques audio
        audio_features = audio_features.unsqueeze(1).unsqueeze(-1)
        audio_x = F.relu(self.conv1(audio_features))
        audio_x = F.relu(self.conv2(audio_x))
        audio_x = self.adaptive_pool(audio_x)
        #audio_x = audio_x.view(audio_x.size(0), -1)  # Aplatir les caractéristiques pour le GRU

        # Traitement des caractéristiques textuelles avec CamemBERT
        text_outputs = self.camembert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_x = text_outputs[0][:, 0, :]  # Prendre le token CLS

        # Fusion des caractéristiques audio et textuelles
        combined = torch.cat((audio_x, text_x), dim=1)

        # Passage à travers le RNN
        x, _ = self.rnn(combined.unsqueeze(1))

        # Prendre le dernier état caché
        x = x[:, -1, :]

        # Passage dans la couche fully connected pour la prédiction
        x = self.fc1(x)
        x = self.fc2(x)

        return x
       
        
"""

class EarlyFusionModel(nn.Module):
    def __init__(self, text_feature_size=768):
        super(EarlyFusionModel, self).__init__()
        # CamemBERT pour les caractéristiques textuelles
        self.camembert = CamembertModel.from_pretrained('camembert-base')

        # Couches de convolution pour les caractéristiques audio
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(6)

        # Couches entièrement connectées pour la fusion
        hidden_size = 32 * 1 * 6
        self.fc1 = nn.Linear(hidden_size + text_feature_size, 2)

    def forward(self, audio_features, text_input_ids, text_attention_mask):
        # Traitement des caractéristiques audio
        audio_x = F.relu(self.conv1(audio_features))
        audio_x = F.relu(self.conv2(audio_x))
        audio_x = self.adaptive_pool(audio_x)
        audio_x = audio_x.view(audio_x.size(0), -1)

        # Traitement des caractéristiques textuelles
        text_outputs = self.camembert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_x = text_outputs[0][:, 0, :]  # Prendre le token CLS
        # Fusion des caractéristiques audio et textuelles
        combined = torch.cat((audio_x, text_x), dim=1)

        # Passage dans la couche entièrement connectée
        x = self.fc1(combined)

        return x