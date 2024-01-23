
import os
import sys
sys.path.insert(0, '../src')
from utils import create_y_yield_at,create_y_turn_after,create_y
print(os.getcwd())
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import load_data
from load_data import load_all_ipus
from utils import create_y_yield_at,create_y_turn_after,create_y,calculate_f1_and_confusion_matrix_audio_text
from audio.audio_extract import extract_audio_segments,extract_features
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
from transformers import CamembertModel,CamembertTokenizer, CamembertForSequenceClassification
from text.text_extract import create_sequences


def load_data_audio_text(seed,task="yield"):
    # load data pd 
    transcr_path='../paco-cheese/transcr'
    data=load_all_ipus(folder_path=transcr_path,load_words=False)
    
    # read y data
    if task=="yield":
        y=create_y_yield_at(data)
    elif task=="turn_after":
        y=create_y_turn_after(data)
    else:
        y=create_y_yield_at(data)
    
    #extract text features
    text_features=create_sequences(data)
    # Initialize tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')


    # extract audio segments
    audio_files_path = '../paco-cheese/audio/2_channels/'
    audio_segments = extract_audio_segments(data,audio_files_path)
    audio_features = np.array([extract_features(segment) for segment in audio_segments])
    
    
    
    # create datasets
    model_name = 'camembert-base'

    temp_camembert = CamembertModel.from_pretrained('camembert-base')

    # Splitting the data
    X_audio_train, X_audio_test, text_features_train, text_features_test, y_train, y_test = train_test_split(audio_features, text_features, y, test_size=0.2, random_state=seed)

    # Initialize tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    max_len = 128  # or any appropriate max length for your text data

    # Splitting the data
    X_audio_train, X_audio_test, text_features_train, text_features_test, y_train, y_test = train_test_split(
        audio_features, text_features, y, test_size=0.2, random_state=seed
    )

    # Creating DataLoaders
    train_dataset = audio_text_Dataset(
        torch.tensor(X_audio_train, dtype=torch.float),  # Ensure correct data type for audio features
        text_features_train, 
        y_train, 
        tokenizer, 
        max_len
    )
    test_dataset = audio_text_Dataset(
        torch.tensor(X_audio_test, dtype=torch.float),  # Ensure correct data type for audio features
        text_features_test, 
        y_test, 
        tokenizer, 
        max_len
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader,test_loader

def training_model_audio_text(num_epochs,seed,model_name,train_loader,test_loader,patience,class_weight=[1.0,5.0],task="yield",save=True):
    # task = yield or turn_after
    #entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    print("Using device:", device)
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    #model=torch.load('modele/camembert_epoch_3.bin')
    model.to(device)

    # Optimiseur et taux d'apprentissage
    optimizer = AdamW(model.parameters(), lr=2e-5)
    class_weights = torch.tensor(class_weight, device=device) # adaptation des poids des classes
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    model = LateFusionModel().to(device)

    # Define class weights for BCEWithLogitsLoss

    class_weights = torch.tensor(class_weight, device=device) 
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of training epochs
    training_loop_audio_text(num_epochs, optimizer, model, loss_fn, train_loader,test_loader,device,model_name=model_name,task=task,patience=patience)
    return model

def evaluate_model_audio(model,model_name,task,test_loader,model_save=True):
    #evaluation
    if model_save:
        model=torch.load(f'../modele/audio_text_model/{task}/{model_name}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f1,conf_matrix=calculate_f1_and_confusion_matrix_audio_text(model, test_loader, device)
    print(f'Test F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Vrais positifs pour la classe 0
    detected_class_1 = conf_matrix[1, 1]  # Vrais positifs pour la classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')
    
    

def training_eval_model_audio_text(num_epochs,seed,model_name,patience,class_weight=[1.0,5.0],task="yield",save=True):
    train_loader,test_loader=load_data_audio_text(seed,task=task)
    model=training_model_audio_text(num_epochs,seed,model_name,train_loader,test_loader,patience,class_weight=class_weight,task=task,save=save)
    evaluate_model_audio(model,task,model_name,test_loader,model_save=False)
    