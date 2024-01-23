import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Crée une liste de labels en fonction du changement de locuteur
def create_y(df):
    y = [0]  # Initialisation avec 0
    for i in range(0, len(df)-1):
        # Vérifie si le locuteur actuel est différent du précédent
        if df['speaker'][i] != df['speaker'][i+1]:
            y.append(1)  # Changement de locuteur
        else:
            y.append(0)  # Pas de changement de locuteur
    return y

# Crée une liste de label en fonction de la colonne 'yield_at_end'
def create_y_yield_at(df):
    y = [] 
    for i in range(len(df)):
        # Vérifie si 'yield_at_end' est True
        if df['yield_at_end'][i] == True:
            y.append(1)  # Changement de locuteur
        else:
            y.append(0)  # Pas de changement de locuteur
    return y

# Crée une liste de label en fonction de la colonne 'turn_after'
def create_y_turn_after(df):
    y = [] 
    for i in range(len(df)):
        # Vérifie si 'turn_after' est True
        if df['turn_after'][i] == True:
            y.append(1)  # Changement de locuteur
        else:
            y.append(0)  # Pas de changement de locuteur
    return y

# Crée une liste de label en fonction d'une fenêtre temporelle
def create_y_time(df, window_time):
    y = [0] 
    for i in range(0, len(df)-1):
        j = i + 1
        stop = False
        while j < len(df) and df['start_words'][j] < df['stop_words'][i] + window_time and not stop:
            if df['speaker'][i] != df['speaker'][j]:
                y.append(1)  # Changement de locuteur
                stop = True
            j += 1
        if not stop:
            y.append(0)  # Pas de changement de locuteur dans la fenêtre de temps
    return y

# Calcule le score F1 et la matrice de confusion pour un modèle audio
def calculate_f1_and_confusion_matrix_audio(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)
            outputs = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix

# Calcule le score F1 et la matrice de confusion pour le modèle de texte
def calculate_f1_and_confusion_matrix_text(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix

# Calcule le score F1 et la matrice de confusion pour le modèle audio-texte
def calculate_f1_and_confusion_matrix_audio_text(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for audio_inputs, text_inputs_dict, labels in data_loader:
            audio_inputs = audio_inputs.to(device)
            input_ids = text_inputs_dict['input_ids'].to(device)
            attention_mask = text_inputs_dict['attention_mask'].to(device)
            labels = labels.to(device)

            # Utilise directement la sortie du modèle comme logits
            outputs = model(audio_inputs, input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix
