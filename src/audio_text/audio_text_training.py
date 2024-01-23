from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import copy
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix

# Fonction pour l'entraînement du modele de late fusion pour text/audio
def training_loop_audio_text(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, model_name, task, patience, save=True):
    best_val_loss = float('inf')  # Imeilleur score de perte sur la validation
    patience_counter = patience  # compteur de patience (pour early stopping)

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0  # Initialisation de la current perte

        # Boucle d'entraînement
        for i, (audio_inputs, text_inputs_dict, labels) in enumerate(train_loader):
            audio_inputs = audio_inputs.to(device)
            input_ids = text_inputs_dict['input_ids'].to(device)
            attention_mask = text_inputs_dict['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(audio_inputs, input_ids, attention_mask)  # Propagation avant
            loss = loss_fn(outputs, labels)  # Calcul de la perte
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des paramètres

            running_loss += loss.item()  # Accumulation de la perte

        avg_train_loss = running_loss / len(train_loader)  # Calcul de la perte moyenne d'entraînement
        print(f'Époque {epoch+1}/{num_epochs}, Perte d\'entraînement : {avg_train_loss:.4f}')

        # Boucle de validation
        model.eval()  
        val_loss = 0.0  # Initialisation de la perte de validation
        with torch.no_grad():
            for audio_inputs, text_inputs_dict, labels in val_loader:
                audio_inputs = audio_inputs.to(device)
                input_ids = text_inputs_dict['input_ids'].to(device)
                attention_mask = text_inputs_dict['attention_mask'].to(device)
                labels = labels.to(device)

                outputs = model(audio_inputs, input_ids, attention_mask)  # Propagation avant
                loss = loss_fn(outputs, labels)  # Calcul de la perte
                val_loss += loss.item()  # Accumulation de la perte de validation

        avg_val_loss = val_loss / len(val_loader)  # Calcul de la perte moyenne de validation
        print(f'Époque {epoch+1}/{num_epochs}, Perte de validation : {avg_val_loss:.4f}')

        # Vérification de l'early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # Mise à jour de la meilleure loss de validation
            patience_counter = patience  # Réinitialisation du compteur de patience
        else:
            patience_counter -= 1  # Réduction du compteur de patience
            if patience_counter == 0:
                print("Arrêt anticipé déclenché")
                break

        # Sauvegarde du modèle si spécifié
        if save:
            if not os.path.isdir(f'../modele/audio_text_model/{task}'):
                os.makedirs(f'../modele/audio_text_model/{task}')
            torch.save(model, f'../modele/audio_text_model/{task}/{model_name}')
            
    return model  
