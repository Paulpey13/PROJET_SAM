import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from pydub import AudioSegment
import librosa

# Define the architecture of the audio CNN model
=======


>>>>>>> 4ab7c92818ce313ade6fcf3f68bad1f9b28c005b
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 6)) # Adaptive pooling to get a fixed size
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

# Define the audio dataset
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

# Define the collate function for the dataloader
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences)
    labels = torch.tensor(labels)
    return sequences, labels

# Function to find the corresponding audio file for a given dyad
def find_audio_file(dyad, first_speaker, audio_files_path):
    dyad = dyad.split('\\')[1] 
    if isinstance(first_speaker, float):
        first_speaker = "NA" # Handle the case where the dataframe recorded NaN instead of NA
    second_speaker = dyad.replace(first_speaker, "")
    for file_name in os.listdir(audio_files_path):
        if first_speaker in file_name and second_speaker in file_name:
            return os.path.join(audio_files_path, file_name)
    return None

<<<<<<< HEAD
# Function to extract audio segments using the information from the provided dataframe
def extract_audio_segments(df, audio_files_path):
=======
# Cette fonction extrait les segments audio en utilisant les informations du dataframe fournis dans le projet
def extract_audio_segments_mots(df,audio_files_path):
>>>>>>> 4ab7c92818ce313ade6fcf3f68bad1f9b28c005b
    audio_segments = []
    audio_file_path = ""
    for index, row in df.iterrows():
        first_speaker = str(row['speaker'])
        if isinstance(row['speaker'], float):
            first_speaker = "NA"
        if first_speaker not in audio_file_path:
            # If the audio file is not loaded, load it
            audio = None
            gc.collect()
            audio_file_path = find_audio_file(row['dyad'], first_speaker, audio_files_path)
            if audio_file_path is None:
                print("Audio file not found for dyad {}".format(row['dyad']))
                audio_file_path = ""
                continue
            audio = AudioSegment.from_file(audio_file_path)
        if audio_file_path != "":
            start_ms = int(row['start_ipu'] * 1000)
            end_ms = int(row['stop_words'] * 1000)
            segment = audio[start_ms:end_ms]
            audio_segments.append(segment)
    return audio_segments

<<<<<<< HEAD
# Function to extract MFCC features from an audio segment
=======
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
            start_ms = int(row['start'] * 1000)
            end_ms = int(row['stop'] * 1000)
            segment = audio[start_ms:end_ms]
            audio_segments.append(segment)
    return audio_segments

>>>>>>> 4ab7c92818ce313ade6fcf3f68bad1f9b28c005b
def extract_features(audio_segment):
    # Convert the audio segment to a numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768
    elif audio_segment.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648
    # Extract the MFCCs
    mfccs = librosa.feature.mfcc(y=samples, sr=audio_segment.frame_rate, n_mfcc=13)
    # Average the MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean