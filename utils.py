import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Function to create the target variable y
def create_y(df):
    y = [0] 
    for i in range(0, len(df)-1):
<<<<<<< HEAD
        # Check if the current speaker is different from the previous one and if the stop_words and stop_ipu are equal
        if df['speaker'][i] != df['speaker'][i+1] and int(df['stop_words'][i])==int(df['stop_ipu'][i]):
=======
        # Check if the current speaker is different from the previous one
        if df['speaker'][i] != df['speaker'][i+1]:
>>>>>>> 4ab7c92818ce313ade6fcf3f68bad1f9b28c005b
            y.append(1)  # Speaker changed
        else:
            y.append(0)  # Speaker did not change
    return y

<<<<<<< HEAD
# Function to calculate the F1 score and confusion matrix for the audio model
=======
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

>>>>>>> 4ab7c92818ce313ade6fcf3f68bad1f9b28c005b
def calculate_f1_and_confusion_matrix_audio(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix

# Function to calculate the F1 score and confusion matrix for the text model
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

    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return f1, conf_matrix

# Function to predict the labels for the audio model
def predition_model_audio(model,dataset,device,proba=True):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)
            if not proba:
                _, preds = torch.max(preds.data, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds,all_labels

# Function to predict the labels for the text model
def prediction_model_text(model,test_loader,device,proba=True):
    all_preds_text = []
    all_labels = []
    with torch.no_grad():
        for d in test_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits
            if not proba:
                _, preds = torch.max(preds, dim=1)

            all_preds_text.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds_text = np.concatenate(all_preds_text, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds_text,all_labels