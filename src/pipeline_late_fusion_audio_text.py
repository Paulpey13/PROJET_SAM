import os
import sys
sys.path.insert(0, '../src')
from utils import create_y_yield_at, create_y_turn_after, create_y
print(os.getcwd())
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import load_data
from load_data import load_all_ipus
from utils import create_y_yield_at, create_y_turn_after, create_y, calculate_f1_and_confusion_matrix_audio_text
from audio.audio_extract import extract_audio_segments, extract_features
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from audio_text.audio_text_training import *
from audio_text.audio_text_model import *
from audio_text.audio_text_dataset import *
from transformers import CamembertTokenizer
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import CamembertModel, CamembertTokenizer, CamembertForSequenceClassification
from text.text_extract import create_sequences

# Fonction pour charger les données audio et textuelles
def load_data_audio_text(seed, task="yield"):
    # Chargement des données textuelles depuis paco(cheese
    transcr_path = '../paco-cheese/transcr'
    data = load_all_ipus(folder_path=transcr_path, load_words=False)
    
    # Création des label en fonction de la tâche
    if task == "yield":
        y = create_y_yield_at(data)
    elif task == "turn_after":
        y = create_y_turn_after(data)
    else:
        y = create_y_yield_at(data)
    
    # Extraction des fonctionnalités textuelles
    text_features = create_sequences(data)
    # Initialiser le tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

    # Extraction des segments audio
    audio_files_path = '../paco-cheese/audio/2_channels/'
    audio_segments = extract_audio_segments(data, audio_files_path)
    audio_features = np.array([extract_features(segment) for segment in audio_segments])
    
    # Création des ensembles de données
    model_name = 'camembert-base'

    temp_camembert = CamembertModel.from_pretrained('camembert-base')

    # Fractionnement des données
    X_audio_train, X_audio_test, text_features_train, text_features_test, y_train, y_test = train_test_split(audio_features, text_features, y, test_size=0.2, random_state=seed)

    # Initialisation du tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    max_len = 128  # ou toute longueur maximale appropriée pour vos données textuelles

    # Création des DataLoaders
    train_dataset = audio_text_Dataset(
        torch.tensor(X_audio_train, dtype=torch.float),  
        text_features_train, 
        y_train, 
        tokenizer, 
        max_len
    )
    test_dataset = audio_text_Dataset(
        torch.tensor(X_audio_test, dtype=torch.float),  
        text_features_test, 
        y_test, 
        tokenizer, 
        max_len
    )
    return train_dataset, test_dataset

# Fonction pour l'entraînement du modèle
def training_model_audio_text(num_epochs, seed, model_name, train_dataset, test_dataset, patience, class_weight=[1.0, 5.0], task="yield", save=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Utilisation de l'appareil:", device)
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    model.to(device)

    # Optimiseur et taux d'apprentissage
    optimizer = AdamW(model.parameters(), lr=2e-5)
    class_weights = torch.tensor(class_weight, device=device)  # Poids de classe pour CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    model = LateFusionModel().to(device)

    # Définition des poids de classe pour BCEWithLogitsLoss
    class_weights = torch.tensor(class_weight, device=device) 
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Configuration de l'optimiseur

    # Boucle d'entraînement
    training_loop_audio_text(num_epochs, optimizer, model, loss_fn, train_dataset, test_dataset, device, model_name=model_name, task=task, patience=patience)
    
    return model

# Fonction pour évaluer le modèle
def evaluate_model_audio(model, model_name, task, test_dataset, model_save=True):
    # Évaluation du modèle
    if model_save:
        model = torch.load(f'../modele/audio_text_model/{task}/{model_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f1, conf_matrix = calculate_f1_and_confusion_matrix_audio_text(model, test_dataset, device)
    print(f'Test F1 Score: {f1}')
    print(f'Matrice de confusion:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Classo0
    detected_class_1 = conf_matrix[1, 1]  # Classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')

# Fonction pour entraîner et évaluer le modèle audio-texte
def training_eval_model_audio_text(num_epochs, seed, model_name, patience, class_weight=[1.0, 5.0], task="yield", save=True):
    train_dataset, test_dataset = load_data_audio_text(seed, task=task)
    model = training_model_audio_text(num_epochs, seed, model_name, train_dataset, test_dataset, patience, class_weight=class_weight, task=task, save=save)
    evaluate_model_audio(model, model_name, task, test_dataset, model_save=False)
