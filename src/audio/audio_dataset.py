from torch.utils.data import Dataset, DataLoader
import torch

class AudioDataset(Dataset):
    """
    Une classe personnalisée pour charger des données audio dans PyTorch.
    
    Attributs:
        data (numpy.ndarray): Un tableau numpy contenant les données audio.
        labels (list): Une liste des étiquettes correspondant à chaque échantillon audio.
    """
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Récupère un échantillon audio et son étiquette en utilisant un index.
        
        Paramètres:
            idx (int): L'index de l'échantillon à récupérer.
        
        Retourne:
            tuple: Contient l'échantillon audio sous forme de tensor PyTorch et son étiquette.
        """
        # Convertit l'échantillon audio de numpy array à PyTorch tensor et ajoute des dimensions supplémentaires nécessaires pour le modèle.
        audio_tensor = torch.from_numpy(self.data[idx]).float()
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        # Récupère l'étiquette correspondante.
        label = self.labels[idx]
        return audio_tensor, label
    

def collate_fn(batch):
    """
    Une fonction de regroupement personnalisée pour le DataLoader.
    
    Paramètres:
        batch (list): Une liste de tuples, où chaque tuple contient un échantillon audio et son étiquette.
    
    Retourne:
        tuple: Contient deux tensors PyTorch, un pour les séquences audio regroupées et un pour les étiquettes regroupées.
    """
    # Sépare les séquences audio et les étiquettes, puis empile les séquences pour créer un batch.
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences)
    
    # Convertit les étiquettes en tensor PyTorch.
    labels = torch.tensor(labels)

    return sequences, labels
