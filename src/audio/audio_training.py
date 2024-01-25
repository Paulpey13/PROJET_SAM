from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import copy
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix

# Fonction pour l'entraînement du modèle audio
def training_loop_audio(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, model_name, task, patience, save=True):
    best_f1 = 0
    patience_init = patience
    
    for epoch in range(num_epochs):
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()

        f1, conf_matrix = calculate_f1_and_confusion_matrix_audio(model, val_loader, device)
        
        if best_f1 < f1:
            best_f1 = f1
            patience = patience_init
            best_model = copy.deepcopy(model)
        else:
            patience -= 1
            
        if patience == 0:
            print("Arrêt anticipé (Early stopping)")
            break
            
        print(f'Époque [{epoch+1}/{num_epochs}], f1 sur validation: {f1}')
        
    if save:
        if not os.path.isdir(f'../modele/audio_model/{task}'):
            os.mkdir(f'../modele/audio_model/{task}')
        torch.save(best_model, f'../modele/audio_model/{task}/{model_name}')
    
    return best_model

# Fonction pour calculer F1-score et la matrice de confusion du modèle audio
def calculate_f1_and_confusion_matrix_audio(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix

# Fonction pour effectuer des prédictions avec le modèle audio
def prediction_model_audio(model, dataset, device, proba=True):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)
            if not proba:
                _, preds = torch.max(preds.data, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_preds, all_labels
