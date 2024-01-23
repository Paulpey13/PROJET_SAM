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

# Fonction pour charger les données audio et textuelles
def load_data_audio_text_early(seed, task="yield"):
    # Chargement des données textuelles depuis un dossier spécifié
    transcr_path = '../paco-cheese/transcr'
    data = load_all_ipus(folder_path=transcr_path, load_words=False)
    
    # Création des label en fonction de la tâche
    if task == "yield":
        y = create_y_yield_at(data)
    elif task == "turn_after":
        y = create_y_turn_after(data)
    else:
        y = create_y_yield_at(data)
    
    # Extraction des segments audio
    audio_files_path = '../paco-cheese/audio/2_channels/'
    audio_segments = extract_audio_segments(data, audio_files_path)
    audio_features = np.array([extract_features(segment) for segment in audio_segments])
    
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    text_features = create_sequences(data)
    
    # Split des données en ensembles d'entraînement, de validation et de test
    X_audio_train, X_audio_test, X_text_features_train, X_text_features_test, y_train, y_test = train_test_split(
        audio_features, text_features, y, test_size=0.2, random_state=seed
    )
    X_audio_train, X_audio_val, X_text_features_train, X_text_features_val, y_train, y_val = train_test_split(
        X_audio_train, X_text_features_train, y_train, test_size=0.2, random_state=seed
    )

    train_dataset = audio_text_Dataset(
        torch.tensor(X_audio_train, dtype=torch.float),
        X_text_features_train,
        y_train,
        tokenizer,
        max_len=128
    )
    val_dataset = audio_text_Dataset(
        torch.tensor(X_audio_val, dtype=torch.float),
        X_text_features_val,
        y_val,
        tokenizer,
        max_len=128
    )
    test_dataset = audio_text_Dataset(
        torch.tensor(X_audio_test, dtype=torch.float),
        X_text_features_test,
        y_test,
        tokenizer,
        max_len=128
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader, test_loader

# Fonction pour l'entraînement du modèle audio-texte avec early
def training_model_audio_text_early(num_epochs, seed, model_name, train_loader, val_loader, patience, class_weight=[1.0, 5.0], task="yield", save=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Utilisation de l'appareil:", device)
    model = EarlyFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_weights = torch.tensor(class_weight, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(patience)
    
    # Entraînement
    model = training_loop_audio_text_early(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, model_name=model_name, task=task, patience=patience, save=save)
    
    return model

# Fonction pour l'évaluation du modèle
def evaluate_model_audio_text_early(model, test_loader):
    f1, conf_matrix = prediction_audio_text_early(model, test_loader)
    print(f'Test F1 Score: {f1}')
    print(f'Matrice de confusion:\n{conf_matrix}')

    total_class_0 = np.sum(conf_matrix[0])
    total_class_1 = np.sum(conf_matrix[1])
    detected_class_0 = conf_matrix[0, 0]
    detected_class_1 = conf_matrix[1, 1]

    print(f'Nombre d\'éléments de classe 0 détectés : {detected_class_0} sur {total_class_0}')
    print(f'Nombre d\'éléments de classe 1 détectés : {detected_class_1} sur {total_class_1}')

# Fonction pour entraîner et évaluer le modèle
def training_evaluate_model_audio_text_early(num_epochs, seed, model_name, patience, class_weight=[1.0, 5.0], task="yield", save=True):
    train_loader, val_loader, test_loader = load_data_audio_text_early(seed, task)
    model = training_model_audio_text_early(num_epochs, seed, model_name, train_loader, val_loader, patience, class_weight=class_weight, task=task, save=save)
    evaluate_model_audio_text_early(model, test_loader)
