import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import load_data

# Chargement des données
transcr_path='paco-cheese/transcr'
data=load_data.load_all_ipus(folder_path=transcr_path,load_words=True)
# Remplacez cette ligne par votre propre méthode de chargement des données

# Encodage des étiquettes
label_encoder = LabelEncoder()
data['turn_after_word_encoded'] = label_encoder.fit_transform(data['turn_after_word'])

# Sélection des caractéristiques et des étiquettes
features = data[['start_ipu', 'stop_ipu', 'duration']]
labels = data['turn_after_word_encoded']

# Séparation en ensembles d'entraînement, de validation et de test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Dataset
class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[idx].values, dtype=torch.float), torch.tensor(self.labels.iloc[idx], dtype=torch.float)

train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)
test_dataset = SimpleDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.linear = nn.Linear(input_size, 4)  # Augmentation de la dimension d'embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(4, num_classes)  # Utilisation de la nouvelle dimension d'embedding

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.fc(x)
        return torch.sigmoid(x)

model = TransformerModel(input_size=3, num_heads=2, num_layers=2, num_classes=1)


# Entraînement
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# Évaluation
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')
