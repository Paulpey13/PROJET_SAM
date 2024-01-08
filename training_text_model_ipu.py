import torch
from torch.utils.data import DataLoader, Dataset
def create_sequences(df, window_size=5):
    sequences = []
                
    for i in range(len(df)):
        if int(df['stop_words'][i])==int(df['stop_ipu'][i]):
            # Prendre window_size mots précédents et le mot courant pour former la séquence
            sequence = df["text_ipu"][i]
            if isinstance(sequence, str):
                sequence = [str(word) for word in sequence]
                sequences.append(' '.join(sequence))
            else:
                sequences.append("unk")
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