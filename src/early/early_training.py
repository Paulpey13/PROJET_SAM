import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import os

# Entrainement et évaluation pour l'early
def training_loop_audio_text_early(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, model_name, task, patience, save=True):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Boucle d'entraînement
        for i, (audio_inputs, text_inputs_dict, labels) in enumerate(train_loader):
            print(f'Batch {i+1}/{len(train_loader)}')
            audio_inputs = audio_inputs.to(device)
            input_ids = text_inputs_dict['input_ids'].to(device)
            attention_mask = text_inputs_dict['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio_inputs, input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')
        
    if save:
        if os.path.isdir(f'../modele/audio_text_early_model/{task}')==False:
            os.makedirs(f'../modele/audio_text_early_model/{task}')
        torch.save(model, f'../modele/audio_text_early_model/{task}/{model_name}')
    return model

# Fonction de prédiction pour l'early audio-texte
def prediction_audio_text_early(model, test_loader):
    # Prédiction
    model.eval()
    all_preds = []
    all_labels = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i, (audio_inputs, text_inputs_dict, labels) in enumerate(test_loader):
            audio_inputs = audio_inputs.to(device)
            input_ids = text_inputs_dict['input_ids'].to(device)
            attention_mask = text_inputs_dict['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(audio_inputs, input_ids, attention_mask)
            _, preds = torch.max(outputs.data, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return f1, conf_matrix
