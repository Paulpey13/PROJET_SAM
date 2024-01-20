from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import copy
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix
def training_loop_audio_text(num_epochs, optimizer, model, loss_fn, train_loader, val_loader, device, model_name, task, patience, save=True):
    best_val_loss = float('inf')
    patience_counter = patience

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for i, (audio_inputs, text_inputs_dict, labels) in enumerate(train_loader):
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

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio_inputs, text_inputs_dict, labels in val_loader:
                audio_inputs = audio_inputs.to(device)
                input_ids = text_inputs_dict['input_ids'].to(device)
                attention_mask = text_inputs_dict['attention_mask'].to(device)
                labels = labels.to(device)

                outputs = model(audio_inputs, input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = patience 
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping triggered")
                break
        if os.path.isdir(f'../modele/audio_text_model/{task}')==False:
            os.makedirs(f'../modele/audio_text_model/{task}')
        torch.save(model, f'../modele/audio_text_modelg/{task}/{model_name}')
