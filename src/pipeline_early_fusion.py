
import os
import sys
sys.path.insert(0, '../src')

from src.load_data import *
from src.audio.audio_extract import *
from src.text.text_extract import *
from transformers import CamembertTokenizer
from src.utils import *
from early.early_model import *
from early.early_dataset import *
from early.early_training import *

def load_data_audio_text_early(seed,task="yield"):
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
    
    
    # extract audio segments
    audio_files_path = '../paco-cheese/audio/2_channels/'
    audio_segments = extract_audio_segments(data,audio_files_path)
    audio_features = np.array([extract_features(segment) for segment in audio_segments])
    #extract text features
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    text_features=create_sequences(data)
    
    text_features=text_features


    # Initialize tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    max_len = 128  # or any appropriate max length for your text data

    # Splitting the data train test
    X_audio_train, X_audio_test, X_text_features_train, X_text_features_test, y_train, y_test = train_test_split(
        audio_features, text_features, y, test_size=0.2, random_state=seed
    )
    # splitting the data validation
    X_audio_train, X_audio_val, X_text_features_train, X_text_features_val, y_train, y_val = train_test_split(
        X_audio_train, X_text_features_train, y_train, test_size=0.2, random_state=seed
    )


    # Creating DataLoaders
    train_dataset = audio_text_Dataset(
        torch.tensor(X_audio_train, dtype=torch.float),  # Ensure correct data type for audio features
        X_text_features_train, 
        y_train, 
        tokenizer, 
        max_len
    )
    val_dataset = audio_text_Dataset(
        torch.tensor(X_audio_val, dtype=torch.float),  # Ensure correct data type for audio features
        X_text_features_val, 
        y_val, 
        tokenizer, 
        max_len
    )
    test_dataset = audio_text_Dataset(
        torch.tensor(X_audio_test, dtype=torch.float),  # Ensure correct data type for audio features
        X_text_features_test, 
        y_test, 
        tokenizer, 
        max_len
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader,val_loader,test_loader



def training_model_audio_text_early(num_epochs,seed,model_name,train_loader,val_loader,patience,class_weight=[1.0,5.0],task="yield",save=True):
    # task = yield or turn_after
    #entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Using device:", device)
    model = EarlyFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_weights = torch.tensor(class_weight, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(patience)
    #training
    model=training_loop_audio_text_early(num_epochs, optimizer, model, loss_fn, train_loader,val_loader,device,model_name=model_name,task=task,patience=patience,save=save)
    
    return model
        
        
def evaluate_model_audio_text_early(model,test_loader):
    #evaluation
    f1,conf_matrix=prediction_audio_text_early(model,test_loader)
    print(f'Test F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Vrais positifs pour la classe 0
    detected_class_1 = conf_matrix[1, 1]  # Vrais positifs pour la classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')



def training_evaluate_model_audio_text_early(num_epochs,seed,model_name,patience,class_weight=[1.0,5.0],task="yield",save=True,all_training=False):
    train_loader,val_loader,test_loader=load_data_audio_text_early(seed,task)
    model=training_model_audio_text_early(num_epochs,seed,model_name,train_loader,val_loader,patience,class_weight=class_weight,task="yield",save=True)
    evaluate_model_audio_text_early(model,test_loader)
    