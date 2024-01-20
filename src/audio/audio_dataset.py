
from torch.utils.data import Dataset, DataLoader
import torch
class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio_tensor = torch.from_numpy(self.data[idx]).float()
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        label = self.labels[idx]
        return audio_tensor, label
    


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences)
    labels = torch.tensor(labels)

    return sequences, labels