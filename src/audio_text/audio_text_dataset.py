import torch
from torch.utils.data import Dataset

class audio_text_Dataset(Dataset):
    def __init__(self, audio_features, texts, labels, tokenizer, max_len):
        self.audio_features = audio_features
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        audio_feature = self.audio_features[idx]
        text = str(self.texts[idx])
        label = self.labels[idx]

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

        return (
            audio_feature,
            {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            },
            label
        )
        
