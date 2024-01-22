
import torch
import torch.nn as nn
import os
import numpy as np
from utils import calculate_f1_and_confusion_matrix_text
import copy
def training_loop_text(num_epochs,optimizer,model,loss_fn, scheduler, train_loader,val_loader, device,model_name,task,patience,save=True):
    model.train()
    losses = []
    correct_predictions = 0
    patience_init=patience
    best_f1=0
    for epoch in range(num_epochs):
        #epoch+=1
        print(f'Epoch {epoch}')
        model.train()
        for d in train_loader:
            

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)


            model.zero_grad()

    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)


            loss = loss_fn(outputs.logits, labels)
            
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        model.eval()
        correct_predictions = 0

        f1,conf=calculate_f1_and_confusion_matrix_text(model, val_loader, device)
        if best_f1 < f1:
            best_f1 = f1
            patience=patience_init
            best_model=copy.deepcopy(model)
        else:
            patience-=1
            
        if patience==0:
            print("Early stopping")
            break
        print(f'validation  f1 score : {f1}')
        
    if save:
        if os.path.isdir(f'../modele/text_model/{task}')==False:
            os.mkdir(f'../modele/text_model/{task}')
        
        torch.save(best_model, f'../modele/text_model/{task}/{model_name}')
    return best_model



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
