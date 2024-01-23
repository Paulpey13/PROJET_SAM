import os
import gc
from pydub import AudioSegment
import numpy as np
import librosa

def find_audio_file(dyad, first_speaker, audio_files_path):
    """
    Trouve le fichier audio correspondant à une dyade spécifique.

    Paramètres:
        dyad (str): Identifiant de la dyade.
        first_speaker (str or float): Identifiant du premier interlocuteur. Peut être NA si inconnu.
        audio_files_path (str): Chemin du répertoire contenant les fichiers audio.

    Retourne:
        str: Chemin du fichier audio correspondant, ou None si non trouvé.
    """
    dyad = dyad.split('\\')[1]
    
    # Gérer le cas où first_speaker est NaN (stocké comme float dans le dataframe)
    if isinstance(first_speaker, float):
        first_speaker = "NA"
    
    # Déterminer l'identifiant du deuxième interlocuteur
    second_speaker = dyad.replace(first_speaker, "")
    
    # Parcourir les fichiers dans le répertoire et trouver le fichier correspondant
    for file_name in os.listdir(audio_files_path):
        if first_speaker in file_name and second_speaker in file_name:
            return os.path.join(audio_files_path, file_name)
    
    # Retourner None si aucun fichier correspondant n'est trouvé
    return None

def extract_audio_segments_mots(df, audio_files_path):
    """
    Extrait les segments audio en fonction des informations fournies dans un dataframe.

    Paramètres:
        df (DataFrame): Dataframe contenant les informations des segments à extraire.
        audio_files_path (str): Chemin du répertoire contenant les fichiers audio.

    Retourne:
        list: Liste des segments audio extraits.
    """
    audio_segments = []
    audio_file_path = ""
    nombre_boucle = 0
    
    for index, row in df.iterrows():
        nombre_boucle += 1
        first_speaker = str(row['speaker'])
        
        #Gérer le cas où la colonne 'speaker' est NaN
        if isinstance(row['speaker'], float):
            first_speaker = "NA"
        
        # Charger le fichier audio si le premier interlocuteur change
        if first_speaker not in audio_file_path:
            audio = None
            gc.collect()  #Libérerla mémoire si nécessaire
            audio_file_path = find_audio_file(row['dyad'], first_speaker, audio_files_path)
            
            if audio_file_path is None:
                print(f"Audio file not found for dyad {row['dyad']}")
                audio_file_path = ""
                continue
            
            #Charger le fichier audio avec pydub
            audio = AudioSegment.from_file(audio_file_path)
        
        if audio_file_path != "":
            # Extraire le segment audio basé sur les timestamps obteinus
            start_ms = int(row['start_ipu'] * 1000)
            end_ms = int(row['stop_words'] * 1000)
            segment = audio[start_ms:end_ms]
            audio_segments.append(segment)
    
    return audio_segments

def extract_audio_segments(df, audio_files_path):
    """
    Extrait les segments audio en fonction des informations fournies dans un dataframe.
    
    Paramètres:
        df (DataFrame): Dataframe contenant les informations des segments à extraire.
        audio_files_path (str): Chemin du répertoire contenant les fichiers audio.
    
    Retourne:
        list: Liste des segments audio extraits.
    """
    audio_segments = []  # Liste pour stocker les segments audio extraits
    audio_file_path = ""  # Chemin vers le fichier audio actuel
    nombre_boucle = 0  # Compteur pour suivre le nombre d'itérations

    for index, row in df.iterrows():
        nombre_boucle += 1
        first_speaker = str(row['speaker'])  
        if isinstance(row['speaker'], float):
            first_speaker = "NA"
        
        # Charger le fichier audio si le premier interlocuteur change
        if first_speaker not in audio_file_path:
            audio = None  
            gc.collect()  
            
            audio_file_path = find_audio_file(row['dyad'], first_speaker, audio_files_path)
            if audio_file_path is None:
                print(f"Audio file not found for dyad {row['dyad']}")
                audio_file_path = ""  # Réinitialiser le chemin du fichier audio
                continue
            
            # Charger le fichier audio avec pydub
            audio = AudioSegment.from_file(audio_file_path)
        
        # Si un fichier audio est chargé
        if audio_file_path != "":
            # Convertir les timestamps de début et de fin de segment en millisecondes
            start_ms = int(row['start'] * 1000)
            end_ms = int(row['stop'] * 1000)
            
            # Extraire le segment audio et l'ajouter à la liste des segments
            segment = audio[start_ms:end_ms]
            audio_segments.append(segment)
    
    return audio_segments


def extract_features(audio_segment):
    """
    Extrait les caractéristiques (MFCC) d'un segment audio.

    Paramètres:
        audio_segment (AudioSegment): Le segment audio à partir duquel extraire les caractéristiques.

    Retourne:
        np.ndarray: Vecteur des moyennes des MFCCs du segment audio.
    """
    # Convertir l'AudioSegment en un array numpy
    samples = np.array(audio_segment.get_array_of_samples())

    # Normaliser les samples en fonction de la largeur de l'échantillon
    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768
    elif audio_segment.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648

    # Extraire les MFCCs en utilisant librosa
    mfccs = librosa.feature.mfcc(y=samples, sr=audio_segment.frame_rate, n_mfcc=13)
    
    # Calculer la moyenne des MFCCs pour chaque coefficient
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean


