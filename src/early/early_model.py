import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CamembertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# Modele d'early fusion audio/text
class EarlyFusionModel(nn.Module):
    def __init__(self, text_feature_size=768):
        super(EarlyFusionModel, self).__init__()
        
        # CamemBERT pour les caractéristiques textuelles
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        
        # Couches de convolution pour les caractéristiques audio
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 6))
        
        hidden_size = 32 * 1 * 6
        self.fc1 = nn.Linear(hidden_size + text_feature_size, 2)
    
    def forward(self, audio_features, text_input_ids, text_attention_mask):
        # Traitement des caractéristiques audio
        audio_features = audio_features.unsqueeze(1).unsqueeze(-1)
        audio_x = F.relu(self.conv1(audio_features))
        audio_x = F.relu(self.conv2(audio_x))
        audio_x = self.adaptive_pool(audio_x)
        audio_x = audio_x.view(audio_x.size(0), -1)

        # Traitement des caractéristiques textuelles avec CamemBERT
        text_outputs = self.camembert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_x = text_outputs[0][:, 0, :]  # Prendre le token CLS
        
        # Fusion des caractéristiques audio et textuelles
        combined = torch.cat((audio_x, text_x), dim=1)

        # Passage dans la couche fully connected pour la prédiction
        x = self.fc1(combined)

        return x
