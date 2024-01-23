from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import copy
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix
from late.combine_model import *

import os
import torch

def training_late_combinaison(num_epochs, optimizer, model_audio, model_text, combine_model, loss_fn, train_loader_audio, train_loader_text, val_loader_audio, val_loader_text, device, model_name, task, patience, save=True):
    best_val_loss = float('inf')
    patience_counter = patience
    model_audio=model_audio.to(device)
    model_text=model_text.to(device)
    for epoch in range(num_epochs):
        model_audio.eval()
        model_text.eval()
        combine_model.train()
        print(device)
        # Training Loop
        for (audio_inputs, labels1), (text_inputs_dict) in zip(train_loader_audio, train_loader_text):
            # Move inputs to the device
            audio_inputs = audio_inputs.to(device)
            input_ids = text_inputs_dict['input_ids'].to(device)
            attention_mask = text_inputs_dict['attention_mask'].to(device)
            labels2 = text_inputs_dict["labels"].to(device)
            # Get predictions from audio and text models
            with torch.no_grad():
                audio_pred = model_audio(audio_inputs)
                text_pred = model_text(input_ids, attention_mask).logits
                
            audio_pred=audio_pred.to(device)
            text_pred=text_pred.to(device)
            # print audio_pred.shape
            # print text_pred.shape
            
            pred=torch.cat((audio_pred, text_pred), 1)  
            # Combine and train the combine model
            combined_pred = combine_model(pred)
            
            combined_pred=combined_pred.to(device)
            labels1=labels1.to(device)
            # Here you need to have the correct labels for combined predictions
            loss = loss_fn(combined_pred, labels1)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        """
        # Validation Loop
        combine_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (audio_inputs, _), (text_inputs_dict, labels) in zip(val_loader_audio, val_loader_text):
                audio_inputs = audio_inputs.to(device)
                input_ids = text_inputs_dict['input_ids'].to(device)
                attention_mask = text_inputs_dict['attention_mask'].to(device)
                labels = labels.to(device)

                # Predictions from audio and text models
                audio_pred = model_audio(audio_inputs)
                text_pred = model_text(input_ids, attention_mask)

                # Prediction from combined model
                combined_pred = combine_model(audio_pred, text_pred)
                loss = loss_fn(combined_pred, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader_audio)  # assuming both loaders have the same length

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')
        """

    return combine_model