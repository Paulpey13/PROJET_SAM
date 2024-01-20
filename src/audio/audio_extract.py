import os
import gc
from pydub import AudioSegment
import numpy as np
import librosa



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
def extract_audio_segments_mots(df,audio_files_path):
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