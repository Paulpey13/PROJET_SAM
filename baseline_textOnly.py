import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import load_data
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU ou CPU
print(torch.cuda.is_available())
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

print(X_train[:100])
print(y_train[:100])# Dataset
class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[idx].values, dtype=torch.float), torch.tensor(self.labels.iloc[idx], dtype=torch.float)

batch_size = 128

train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)
test_dataset = SimpleDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
epochs = 10
model.to(device)
# Entraînement
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Fonction pour calculer le score F1
def calculate_f1(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds)

# Entraînement avec validation
best_val_f1 = 0.0
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # Validation
    #val_f1 = calculate_f1(model, val_loader)
    #print(f'Epoch {epoch}, Loss: {loss.item()}, Validation F1: {val_f1}')
    
    # Sauvegarde du meilleur modèle
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')

# Évaluation sur l'ensemble de test
test_f1 = calculate_f1(model, test_loader)
print(f'Test F1 Score: {test_f1}')