import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
def create_y(df):
    y = [0] 
    for i in range(0, len(df)-1):
        # Check if the current speaker is different from the previous one
        if df['speaker'][i] != df['speaker'][i+1]:
            y.append(1)  # Speaker changed
        else:
            y.append(0)  # Speaker did not change
    return y

def create_y_yield_at(df):
    y = [] 
    for i in range(len(df)):
        # Check if the current speaker is different from the previous one
        if df['yield_at_end'][i]==True:
            y.append(1)  # Speaker changed
        else:
            y.append(0)  # Speaker did not change
    return y


def create_y_turn_after(df):
    y = [] 
    for i in range(len(df)):
        # Check if the current speaker is different from the previous one
        if df['turn_after'][i]==True:
            y.append(1)  # Speaker changed
        else:
            y.append(0)  # Speaker did not change
    return y
    
def create_y_time(df,window_time):
    y = [0] 
    for i in range(0, len(df)-1):
        # Check if the current speaker is different from the previous one
        j=i+1
        stop=False
        while j<len(df)and df['start_words'][j] < df['stop_words'][i]+window_time and not stop:
            if df['speaker'][i] != df['speaker'][j]:
                y.append(1)  # Speaker changed
                stop = True
            j+=1
        if stop==False:
            y.append(0) # Speaker did not change in time windows
    return y

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

            # Directly use the output of the model as logits
            outputs = model(audio_inputs, input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix
