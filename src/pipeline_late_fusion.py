
import os
import sys
sys.path.insert(0, '../src')

from late import *
from pipeline_audio_model import *
from pipeline_text_model import *
from late.late_audio_text_training import *
from late.combine_model import *
def load_data_audio_text_late(seed,task="yield"):
    train_loader_audio,val_loader_audio,test_loader_audio=load_data_audio(seed,task)
    train_loader_text,val_loader_text,test_loader_text=load_data_text(seed,task)
    return train_loader_audio,val_loader_audio,test_loader_audio,train_loader_text,val_loader_text,test_loader_text

def training_audio_text_late_model(num_epochs,seed,model_name,train_loader_audio,val_loader_audio,test_loader_audio,train_loader_text,val_loader_text,test_loader_text,patience,class_weight=[1.0,5.0],task="yield",save=True,all_training=True):
    if all_training:
        model_text=training_model_text(num_epochs,seed,model_name,train_loader_text,val_loader_text,patience,class_weight=class_weight,task=task,save=save)
        model_audio=training_model_audio(num_epochs,seed,model_name,train_loader_audio,val_loader_audio,patience=patience,class_weight=class_weight,task=task,save=save)
    else:
        model_text=torch.load(f'../modele/text_model/{task}/{model_name}')
        model_audio=torch.load(f'../modele/audio_model/{task}/{model_name}')   
        
    combine_model=Combine_prediction()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("model")
    print(device)
    optimizer = AdamW(combine_model.parameters(), lr=2e-5)
    class_weights = torch.tensor(class_weight, device=device) # adaptation des poids des classes
    combine_model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    combine_model=training_late_combinaison(num_epochs, optimizer, model_audio, model_text, combine_model, loss_fn, train_loader_audio, train_loader_text, val_loader_audio, val_loader_text, device, model_name, task, patience, save=True)
        
        


def training_evaluate_model_audio_text_late(num_epochs,seed,model_name,patience,class_weight=[1.0,5.0],task="yield",save=True,all_training=False):
    train_loader_audio,val_loader_audio,test_loader_audio,train_loader_text,val_loader_text,test_loader_text=load_data_audio_text_late(seed,task)
    training_audio_text_late_model(num_epochs,seed,model_name,train_loader_audio,val_loader_audio,test_loader_audio,train_loader_text,val_loader_text,test_loader_text,patience,class_weight=class_weight,task=task,save=save,all_training=all_training)