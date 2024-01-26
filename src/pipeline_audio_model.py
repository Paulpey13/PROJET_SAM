import os
import sys
sys.path.insert(0, '../src')  
print(os.getcwd())
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import load_data
from load_data import *
from audio.audio_extract import extract_audio_segments, extract_features
from audio.audio_dataset import AudioDataset, collate_fn
from audio.audio_model import AudioCNN
from audio.audio_training import training_loop_audio, prediction_model_audio

from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
print(os.getcwd())

# Fonction pour charger les données audio
def load_data_audio(seed, task="yield"):
    # Chargement des données depuis un répertoire
    transcr_path = '../paco-cheese/transcr'
    data = load_all_ipus(folder_path=transcr_path, load_words=False)
    
    # Création des labels en fonction de la tâche
    if task == "yield":
        y = create_y_yield_at(data)
    elif task == "turn_after":
        print("tache turn_after" )
        y = create_y_turn_after(data)
    else:
        y = create_y_yield_at(data)
    
    # Extraction des segments audio
    audio_files_path = '../paco-cheese/audio/2_channels/'
    audio_segments = extract_audio_segments(data, audio_files_path)
    X = np.array([extract_features(segment) for segment in audio_segments])
    
    # Split des données en ensembles d'entraînement, de validation et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Split des données d'entraînement en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.001, random_state=seed)

    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, test_loader

def training_model_audio(num_epochs, seed, model_name, train_loader, val_loader, patience, class_weight=[1.0, 5.0], task="yield", save=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Using device:", device)
    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_weights = torch.tensor(class_weight, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(patience)
    # Entraînement du modèle
    model = training_loop_audio(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, model_name=model_name, task=task, patience=patience)
    
    return model
    
# Fonction pour évaluer le  modèle audio
def evaluate_model_audio(model, model_name, task, test_loader, model_save=True):
    # Évaluation du modèle
    if model_save:
        model = torch.load(f'../modele/audio_model/{task}/{model_name}')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds_audio, all_labels = prediction_model_audio(model, test_loader, device, proba=False)
    
    # Calcul du score F1
    f1 = f1_score(all_labels, all_preds_audio)
    print(f'Score F1 de test : {f1}')

    # Calcul du Kappa de Cohen
    kappa = cohen_kappa_score(all_labels, all_preds_audio)
    print(f'Score Kappa de Cohen : {kappa}')

    # Calcul et affichage de la matrice de confusion
    conf_matrix = confusion_matrix(all_labels, all_preds_audio)
    print(f'Matrice de confusion :\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Vrais positifs pour la classe 0
    detected_class_1 = conf_matrix[1, 1]  # Vrais positifs pour la classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')
    return f1, conf_matrix,kappa

# Fonction pour entraîner et évaluer le modèle audio en faisant varier les hyperparamètres
def training_eval_model_audio(num_epochs, seed, model_name, patience, class_weight=[1.0, 5.0], task="yield", save=True):
    
    train_loader, val_loader, test_loader = load_data_audio(seed, task=task) 
    
    model = training_model_audio(num_epochs, seed, model_name, train_loader, val_loader, patience=patience, class_weight=class_weight, task=task, save=save)
    f1_train, conf_matrix_train,kappa_train = evaluate_model_audio(model, model_name, task, train_loader, model_save=False)
    f1_test, conf_matrix_test,kappa_test = evaluate_model_audio(model, model_name, task, test_loader, model_save=False)
    return f1_train, conf_matrix_train,kappa_train, f1_test, conf_matrix_test,kappa_test
