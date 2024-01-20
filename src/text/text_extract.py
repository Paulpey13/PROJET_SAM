import torch
from torch.utils.data import DataLoader, Dataset
def create_sequences_mots(text_series, window_size=5):
    sequences = []
    for i in range(1, window_size + 1):
        # Ajout du nombre approprié de mots "debut" au début quand on a pas de contexte précédent
        padded_sequence = ['debut'] * (window_size - i) + text_series[:i]
        sequences.append(' '.join(padded_sequence))
        
    for i in range(window_size, len(text_series)):
        # Prendre window_size mots précédents et le mot courant pour former la séquence
        sequence = text_series[i - window_size:i+1]
        sequence = [str(word) for word in sequence]
        sequences.append(' '.join(sequence))
    return sequences

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

