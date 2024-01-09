from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch
import torch.nn.utils.rnn as rnn_utils
import os
import gc
from pydub import AudioSegment
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 6)) # pooling adaptatif pour obtenir une taille fixe

        self.fc1 = nn.Linear(32 * 1 * 6, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)

        x = x.view(-1, 32 * 1 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


# Cette fonction prend une entrée de dyade et renvoie le chemin du fichier audio correspondant
def find_audio_file(dyad,first_speaker,audio_files_path):
    dyad=dyad.split('\\')[1] 
    if isinstance(first_speaker, float):
        first_speaker="NA" #on gére le cas ou le dataframe a enregistré NaN au lieu de NA
    second_speaker=dyad.replace(first_speaker,"")
    for file_name in os.listdir(audio_files_path):
        if first_speaker in file_name and second_speaker in file_name:
            return os.path.join(audio_files_path, file_name)
    return None

# Cette fonction extrait les segments audio en utilisant les informations du dataframe fournis dans le projet
def extract_audio_segments(df,audio_files_path):
    audio_segments = []
    audio_file_path = ""
    nombre_boucle=0
    for index, row in df.iterrows():
        nombre_boucle+=1
        first_speaker=str(row['speaker'])
        if isinstance(row['speaker'], float):
            first_speaker="NA"
            
            
        if first_speaker not in audio_file_path:
            # Si le fichier audio n'est pas chargé, on le charge
            audio = None
            gc.collect()
            audio_file_path = find_audio_file(row['dyad'],first_speaker,audio_files_path)
            if audio_file_path is None:
                print("Audio file not found for dyad {}".format(row['dyad']))
                audio_file_path = ""
                continue
            audio = AudioSegment.from_file(audio_file_path)
        if audio_file_path!="":
            start_ms = int(row['start_ipu'] * 1000)
            end_ms = int(row['stop_words'] * 1000)
            segment = audio[start_ms:end_ms]
            audio_segments.append(segment)
    return audio_segments

def extract_features(audio_segment):
    # conversion de l'audio segment en numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768
    elif audio_segment.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648

    # extraction des MFCCs
    mfccs = librosa.feature.mfcc(y=samples, sr=audio_segment.frame_rate, n_mfcc=13)
    
    # moyenne des MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean
