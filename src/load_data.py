"""
Download examples:

Whole archive:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output archive.zip

Specific files with path:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download?path=%2Fvideo%2F&files=openface.zip' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output video/openface.zip
"""


import pandas as pd
import numpy as np
import os
from glob import glob

def load_all_ipus(folder_path:str='transcr', load_words:bool=False):
    """Load all csv and concatenate
    """
    file_list = glob(os.path.join(folder_path, f"*_merge{'_words' if load_words else ''}.csv"))
    # Load all csv files
    data = []
    for file in file_list:
        df = pd.read_csv(file, na_values=['']) # one speaker name is 'NA'
        df['dyad'] = file.split('/')[-1].split('_')[0]
        data.append(df)
            
    #a remettre si besoin
    data = pd.concat(data, axis=0).reset_index(drop=True)
    #print(data.shape)
    plabels = [col for col in data.columns if not any([col.startswith(c) 
            for c in ['dyad', 'ipu_id','speaker','start','stop','text', 'duration']])]
    #print(data[plabels].sum(axis=0) / data.shape[0])
    return data

def filter_after_jokes(df_ipu:pd.DataFrame):
    """First few ipus are useless / common to all conversations"""
    jokes_end = df_ipu[df_ipu.text.apply(lambda x: False if isinstance(x, float) else (
                ('il y avait un âne' in x) or ("qui parle ça c'est cool" in x)))].groupby('dyad').agg(
                {'ipu_id':'max'}).to_dict()['ipu_id']
    return df_ipu[df_ipu.apply(lambda x: x.ipu_id > jokes_end.get(x.dyad,0), axis = 1)], jokes_end

# Crée une liste de labels en fonction du changement de locuteur
def create_y(df):
    y = [0]  # Initialisation avec 0
    for i in range(0, len(df)-1):
        # Vérifie si le locuteur actuel est différent du précédent
        if df['speaker'][i] != df['speaker'][i+1]:
            y.append(1)  # Changement de locuteur
        else:
            y.append(0)  # Pas de changement de locuteur
    return y

# Crée une liste de label en fonction de la colonne 'yield_at_end'
def create_y_yield_at(df):
    y = [] 
    for i in range(len(df)):
        # Vérifie si 'yield_at_end' est True
        if df['yield_at_end'][i] == True:
            y.append(1)  # Changement de locuteur
        else:
            y.append(0)  # Pas de changement de locuteur
    return y

# Crée une liste de label en fonction de la colonne 'turn_after'
def create_y_turn_after(df):
    y = [] 
    for i in range(len(df)):
        # Vérifie si 'turn_after' est True
        if df['turn_after'][i] == True:
            y.append(1)  # Changement de locuteur
        else:
            y.append(0)  # Pas de changement de locuteur
    return y