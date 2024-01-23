from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Définition d'une classe pour le modèl LSTM.
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, num_layers=2):
        super(LSTMClassifier, self).__init__()
        # Initialisation des dimensions et du nombre de couches.
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Création d'une couche LSTM. (à augmenter si besoin, dans le notebook)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Création d'une couche linéaire pour les prédictions.
        self.hidden2label = nn.Linear(hidden_dim, label_size)

    def forward(self, batch):
        lstm_out, _ = self.lstm(batch)
        label_space = self.hidden2label(lstm_out[:, -1, :])
        # Application de la fonction sigmoid pour obtenir des scores entre 0 et 1.
        label_scores = torch.sigmoid(label_space)
        return label_scores

def train_model(model, train_data, loss_function, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        # Boucle sur les données d'entraînement.
        for X, y in train_data:
            X, y = X.to(device), y.to(device)
            model.zero_grad()
            # Obtention des prédictions du modèle.
            predictions = model(X)
            # Calcul de la perte.
            loss = loss_function(predictions, y.unsqueeze(1).float())
            loss.backward()
            # MAJ des paramètres du modèle.
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data)}")
        
def evaluate_model(model, test_data, device='cpu'):
    model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            output = model(X)
            # Assurer que les prédictions sont traitées comme une liste. (Sinon ça buguait)
            predictions = output.squeeze().tolist()
            if type(predictions) != list:
                predictions = [predictions]
            test_predictions += predictions
            labels = y.tolist()
            if type(labels) != list:
                labels = [labels]
            test_labels += labels
    return test_labels, test_predictions
