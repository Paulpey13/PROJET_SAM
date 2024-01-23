from torch.utils.data import DataLoader, Dataset
import torch

# Classe de Dataset pour le texte
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Encodage du texte en utilisant le tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,  # Texte brut
            'input_ids': encoding['input_ids'].flatten(),  # Identifiants d'entrée
            'attention_mask': encoding['attention_mask'].flatten(),  # Masque d'attention
            'labels': torch.tensor(label, dtype=torch.long)  # labels sous forme de tenseur torch
        }
