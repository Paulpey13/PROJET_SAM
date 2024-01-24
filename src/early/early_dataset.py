from torch.utils.data import Dataset
import torch
# Définition du dataset audio/Text pour early fusion
class audio_text_Dataset(Dataset):
    def __init__(self, audio_features, texts, labels, tokenizer, max_len):
        self.audio_features = audio_features  # Caractéristiques audio
        self.texts = texts  # Textes
        self.labels = labels  # Label
        self.tokenizer = tokenizer  # Tokenizer pour le texte
        self.max_len = max_len  # Longueur maximale des séquences

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        audio_features = self.audio_features[idx].float()
        audio_features = audio_features.unsqueeze(0)  # Ajout dimension de canal
        text = str(self.texts[idx])  # Récupérer le texte
        label = self.labels[idx]  # Récupérer label

        # Encodage du texte avec le tokenizer
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

        # Renvoyer un tuple contenant l'audio, l'encodage du texte et dulabel
        return (
            audio_features,
            {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            },
            label
        )
