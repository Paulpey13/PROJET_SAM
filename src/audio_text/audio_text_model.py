import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertModel
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch.nn.functional as F


# Assuming X_audio is your audio features and text_features is your tokenized text data
# Make sure to adjust these as per your data processing
class LateFusionModel(nn.Module):
    def __init__(self):
        super(LateFusionModel, self).__init__()
        # CamemBERT for text features
        self.camembert = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 6)) # pooling adaptatif pour obtenir une taille fixe

        self.fc1 = nn.Linear(32 * 1 * 6, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)
        
        
        self.fc4= nn.Linear(4,2)
    def forward(self, audio_features, text_input_ids, text_attention_mask):
        audio_features = audio_features.unsqueeze(1).unsqueeze(-1)
        x = F.relu(self.conv1(audio_features))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)

        x = x.view(-1, 32 * 1 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Process text features
        text_logits = self.camembert(input_ids=text_input_ids, attention_mask=text_attention_mask).logits
        
        # Concatenate audio and text predictions
        x = torch.cat((x, text_logits), dim=1)
        x = self.fc4(x)
        return x