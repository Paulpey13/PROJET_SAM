import os
import sys
sys.path.insert(0, '../src')
from late import *
from pipeline_audio_model import *
from pipeline_text_model import *
from late.late_audio_text_training import *
from late.combine_model import *
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from load_data import *
# Fonction pour charger les données audio et texte 
def load_data_audio_text_late(seed, task="yield"):
    # Chargement des données audio
    train_loader_audio, val_loader_audio, test_loader_audio = load_data_audio(seed, task)
    
    # Chargement des données textuelles
    train_loader_text, val_loader_text, test_loader_text = load_data_text(seed, task)
    
    return train_loader_audio, val_loader_audio, test_loader_audio, train_loader_text, val_loader_text, test_loader_text

# Fonction pour l'entraînement du modèle audio-texte late fusion
def training_audio_text_late_model(num_epochs, seed, model_name, train_loader_audio, val_loader_audio, test_loader_audio, train_loader_text, val_loader_text, test_loader_text, patience, class_weight=[1.0,5.0], task="turn_after", save=True, all_training=True,modele_addition=False):
    if all_training:
        # Entraînement des modèles audio et textuels
        model_text = training_model_text(1, seed, model_name, train_loader_text, val_loader_text, patience, class_weight=[1.0,3.6], task=task, save=save)
        model_audio = training_model_audio(5, seed, model_name, train_loader_audio, val_loader_audio, patience=patience, class_weight=[1.0,4.0], task=task, save=save)
    else:
        # Si les modèles ne sont pas déjà entraînés, ils sont entraînés ici
        if os.path.isfile(f'../modele/text_model/{task}/{model_name}') == False:
            print("Modèle textuel non trouvé, entraînement en cours...")
            model_text = training_model_text(num_epochs, seed, model_name, train_loader_text, val_loader_text, patience, class_weight=class_weight, task=task, save=save)
        if os.path.isfile(f'../modele/audio_model/{task}/{model_name}') == False:
            print("Modèle audio non trouvé, entraînement en cours...")
            model_audio = training_model_audio(num_epochs, seed, model_name, train_loader_audio, val_loader_audio, patience=patience, class_weight=class_weight, task=task, save=save)
        
        # Chargement des modèles
        model_text = torch.load(f'../modele/text_model/{task}/{model_name}')
        model_audio = torch.load(f'../modele/audio_model/{task}/{model_name}')
    
    combine_model = Combine_prediction()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Utilisation de l'appareil:", device)
    optimizer = AdamW(combine_model.parameters(), lr=2e-5)
    class_weights = torch.tensor(class_weight, device=device) # Adaptation des poids de classe
    combine_model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    if modele_addition==False:
        # Entraînement de la combinaison des modèles
        combine_model = training_late_combinaison(num_epochs, optimizer, model_audio, model_text, combine_model, loss_fn, train_loader_audio, train_loader_text, val_loader_audio, val_loader_text, device, model_name, task, patience, save=True)
    
    return combine_model, model_audio, model_text

# Fonction pour évaluer le modèle 
def evaluate_model_audio_text_late(combine_model, model_audio, model_text, test_loader_audio, test_loader_text,modele_addition):
    # Évaluation
    all_preds,all_labels = prediction_late_combinaison(combine_model, model_audio, model_text, test_loader_audio, test_loader_text,modele_addition)
    f1=f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    kappa=cohen_kappa_score(all_labels, all_preds)
    print(f'Test F1 Score: {f1}')
    print(f'Matrice de confusion:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]  # Vrais positifs pour la classe 0
    detected_class_1 = conf_matrix[1, 1]  # Vrais positifs pour la classe 1

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')
    return f1, conf_matrix,kappa
# Fonction pour entraîner et évaluer le modele
def training_evaluate_model_audio_text_late(num_epochs, seed, model_name, patience, class_weight=[1.0,5.0], task="yield", save=True, all_training=False,modele_addition=False):
    train_loader_audio, val_loader_audio, test_loader_audio, train_loader_text, val_loader_text, test_loader_text = load_data_audio_text_late(seed, task)
    combine_model, model_audio, model_text = training_audio_text_late_model(num_epochs, seed, model_name, train_loader_audio, val_loader_audio, test_loader_audio, train_loader_text, val_loader_text, test_loader_text, patience, class_weight=class_weight, task=task, save=save, all_training=all_training,modele_addition=modele_addition)
    print("train")
    f1_train, conf_matrix_train,kappa_train = evaluate_model_audio_text_late(combine_model, model_audio, model_text, train_loader_audio, train_loader_text,modele_addition)
    print("test")
    f1_test, conf_matrix_test,kappa_test = evaluate_model_audio_text_late(combine_model, model_audio, model_text, test_loader_audio, test_loader_text,modele_addition)
    return f1_train, conf_matrix_train,kappa_train, f1_test, conf_matrix_test,kappa_test
