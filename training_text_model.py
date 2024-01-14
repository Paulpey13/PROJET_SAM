import torch
from torch.utils.data import DataLoader, Dataset

# Function to create sequences from the text series
def create_sequences(text_series, window_size=5):
    sequences = []
    for i in range(1, window_size + 1):
        # Add the appropriate number of "start" words at the beginning when there is no previous context
        padded_sequence = ['start'] * (window_size - i) + text_series[:i]
        sequences.append(' '.join(padded_sequence))
        
    for i in range(window_size, len(text_series)):
        # Take the previous window_size words and the current word to form the sequence
        sequence = text_series[i - window_size:i+1]
        sequence = [str(word) for word in sequence]
        sequences.append(' '.join(sequence))
    return sequences

# Dataset class for the text
def create_sequences(df):
    sequences = [] 
    for i in range( len(df)):
        sequence = df['text'][i]
        if isinstance(sequence,float):
            sequence="unk"
        else:
            sequence = [str(word) for word in sequence]
        sequences.append(' '.join(sequence))
    return sequences

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

        # Encode the text using the provided tokenizer
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }