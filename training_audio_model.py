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
        
        # Utilisation d'un pooling adaptatif pour obtenir une taille fixe
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 6))

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
        # Convertir numpy.ndarray en tenseur PyTorch
        audio_tensor = torch.from_numpy(self.data[idx]).float()

        # Reshape pour obtenir une forme [1, 1, 13] qui représente [channels, height, width]
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        label = self.labels[idx]
        return audio_tensor, label
    


def collate_fn(batch):
    # Extraction des séquences et des étiquettes
    sequences, labels = zip(*batch)

    # Conversion en tenseurs PyTorch
    sequences = torch.stack(sequences)
    labels = torch.tensor(labels)

    return sequences, labels


# Cette fonction prend une entrée de dyade et renvoie le chemin du fichier audio correspondant
def find_audio_file(dyad,first_speaker,audio_files_path):
    dyad=dyad.split('\\')[1]
    #first_speaker=str(first_speaker)
    if isinstance(first_speaker, float):
        first_speaker="NA"
        print(first_speaker)
    second_speaker=dyad.replace(first_speaker,"")
    #if "LS" in dyad:
    #    print("trouvé")
    # Trouver le fichier audio qui contient l'identifiant de la dyade
    for file_name in os.listdir(audio_files_path):
        #if("LS" in dyad):
            #print(first_speaker)
            #print(second_speaker)
            #print(file_name)
        if first_speaker in file_name and second_speaker in file_name:
            return os.path.join(audio_files_path, file_name)
    return None

# Cette fonction extrait les segments audio en utilisant les informations de la dyade et les timestamps
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
            # Si le fichier audio n'a pas encore été chargé, chargez-le
            audio = None
            gc.collect()
            audio_file_path = find_audio_file(row['dyad'],first_speaker,audio_files_path)
            if audio_file_path is None:
                print("Audio file not found for dyad {}".format(row['dyad']))
                audio_file_path = ""
                continue
            audio = AudioSegment.from_file(audio_file_path)
            #print(index)
            print(audio_file_path)  
            print(len(audio))
        if audio_file_path!="":
            start_ms = int(row['start_ipu'] * 1000)
            end_ms = int(row['stop_words'] * 1000)
            segment = audio[start_ms:end_ms]
            audio_segments.append(segment)
            # Vous pouvez également enregistrer le segment si nécessaire
            # segment.export('segment_{}.wav'.format(index), format='wav')
        if audio_file_path=="":
            print("trouvé")
            print(audio_file_path)
    return audio_segments

def extract_features(audio_segment):
    # Convert PyDub audio segment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    # Normalize the audio samples to floating-point values
    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768
    elif audio_segment.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648

    # Use librosa to extract MFCCs
    mfccs = librosa.feature.mfcc(y=samples, sr=audio_segment.frame_rate, n_mfcc=13)
    
    # Average the MFCCs over time
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean
