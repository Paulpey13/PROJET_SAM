import os
import sys
sys.path.insert(0, '../src')
from utils import create_y_yield_at, create_y_turn_after, create_y
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.utils.rnn as rnn_utils
from load_data import load_all_ipus
from text.text_dataset import *
from text.text_extract import *
from text.text_training import *
from sklearn.metrics import f1_score, confusion_matrix,cohen_kappa_score
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import CamembertTokenizer, CamembertForSequenceClassification

# Fonction pour charger les données textuelles
def load_data_text(seed, task="yield"):
    # Chemin vers les csv
    transcr_path = '../paco-cheese/transcr'
    
    # Chargement des données
    data = load_all_ipus(folder_path=transcr_path, load_words=False)
    
    # Création des labels
    if task == "yield":
        y = create_y_yield_at(data)
    elif task == "turn_after":
        y = create_y_turn_after(data)
    else:
        y = create_y_yield_at(data)
    
    # Extraction des caractéristiques textuelles
    data_text = create_sequences(data)
    
    # Split des données en ensembles d'entraînement, de validation et de test
    X_train, X_test, y_train, y_test = train_test_split(data_text, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    
    # Paramètres pour le modèle CamemBERT
    model_name = 'camembert-base'
    max_length = 256
    batch_size = 16
    
    # Tokenizer pour CamemBERT
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    
    # Création des ensembles de données et des loader
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Fonction pour l'entraînement du modèle textuel
def training_model_text(num_epochs, seed, model_name, train_loader, val_loader, patience, class_weight=[1.0,5.0], task="yield", save=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Utilisation de l'appareil:", device)
    
    # Chargement du modèle CamemBERT
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    model.to(device)

    # Optimiseur et taux d'apprentissage
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Adaptation des poids de classe
    class_weights = torch.tensor(class_weight, device=device)
    
    # Fonction de perte
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Entraînement du modèle
    model = training_loop_text(num_epochs, optimizer, model, loss_fn, scheduler, train_loader, val_loader, device, model_name=model_name, task=task, patience=patience)
    
    return model
        
# Fonction pour évaluer le modèle textuel
def evaluate_model_text(model, model_name, task, test_loader, model_save=True):
    # Évaluation
    if model_save:
        model = torch.load(f'modele/text_model/{task}/{model_name}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    all_preds_audio, all_labels = prediction_model_text(model, test_loader, device, proba=False)
    kappa=cohen_kappa_score(all_labels, all_preds_audio)
    # Calcul du score F1 et de la matrice de confusion
    f1 = f1_score(all_labels, all_preds_audio)
    conf_matrix = confusion_matrix(all_labels, all_preds_audio)
    
    print(f'Score F1 de test: {f1}')
    print(f'Matrice de confusion:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Vrais positifs pour la classe 0
    detected_class_1 = conf_matrix[1, 1]  # Vrais positifs pour la classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')
    return f1, conf_matrix,kappa
# Fonction pour entraîner et évaluer le modèle textuel
def training_eval_model_text(num_epochs, seed, model_name, patience, class_weight=[1.0,5.0], task="yield", save=True):
    train_loader, val_loader, test_loader = load_data_text(seed, task=task)
    model = training_model_text(num_epochs, seed, model_name, train_loader, val_loader, patience, class_weight=class_weight, task=task, save=save)
    f1_train, conf_matrix_train,kappa_train = evaluate_model_text(model, model_name, task, train_loader, model_save=False)
    f1_test, conf_matrix_test,kappa_test= evaluate_model_text(model, model_name, task, test_loader, model_save=False)
    return f1_train, conf_matrix_train,kappa_train, f1_test, conf_matrix_test,kappa_test
