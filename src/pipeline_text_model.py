import os
import sys
sys.path.insert(0, '../src')
from utils import create_y_yield_at,create_y_turn_after,create_y
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.utils.rnn as rnn_utils
from load_data import load_all_ipus
from utils import create_y_yield_at,create_y_turn_after,create_y
from text.text_dataset import *
from text.text_extract import *
from text.text_training import *
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import CamembertTokenizer, CamembertForSequenceClassification

def load_data_text(seed,task="yield"):
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
    
    
    data_text=create_sequences(data)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(data_text, y, test_size=0.2, random_state=seed)
    # split train data into train and validation
    X_train, X_val, y_train,y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    

    # create datasets
    model_name = 'camembert-base'
    max_length = 256
    batch_size = 16

    # Tokenizer pour CamemBERT
    tokenizer = CamembertTokenizer.from_pretrained(model_name)

    # Création des datasets et dataloaders
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create data loaders.

    return train_loader,val_loader,test_loader
    
def training_model_text(num_epochs,seed,model_name,train_loader,val_loader,patience,class_weight=[1.0,5.0],task="yield",save=True):
    # task = yield or turn_after
    #entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    print("Using device:", device)
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    #model=torch.load('modele/camembert_epoch_3.bin')
    model.to(device)

    # Optimiseur et taux d'apprentissage
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
)
    class_weights = torch.tensor(class_weight, device=device) # adaptation des poids des classes
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    model=training_loop_text(num_epochs, optimizer, model, loss_fn, scheduler, train_loader,val_loader,device,model_name=model_name,task=task,patience=patience)
    return model
        
def evaluate_model_text(model,model_name,task,test_loader,model_save=True):
    #evaluation
    if model_save:
            model=torch.load(f'modele/text_model/{task}/{model_name}')
            
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds_audio,all_labels=prediction_model_text(model,test_loader,device,proba=False)
    f1 = f1_score(all_labels, all_preds_audio)#, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds_audio)
    print(f'Test F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Vrais positifs pour la classe 0
    detected_class_1 = conf_matrix[1, 1]  # Vrais positifs pour la classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')



def training_eval_model_text(num_epochs,seed,model_name,patience,class_weight=[1.0,5.0],task="yield",save=True):
    train_loader,val_loader,test_loader=load_data_text(seed,task=task)
    model=training_model_text(num_epochs,seed,model_name,train_loader,val_loader,patience,class_weight=class_weight,task=task,save=save)
    evaluate_model_text(model,task,model_name,test_loader,model_save=False)
    