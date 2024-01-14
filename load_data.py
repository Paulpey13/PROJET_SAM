import pandas as pd
import numpy as np
import os
from glob import glob

"""
Download examples:

Whole archive:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output archive.zip

Specific files with path:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download?path=%2Fvideo%2F&files=openface.zip' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output video/openface.zip
"""

# Function to load all csv files and concatenate them
def load_all_ipus(folder_path:str='transcr', load_words:bool=False):
    """
    Load all csv files from a specified folder and concatenate them into a single DataFrame.
    If 'load_words' is True, it loads files with '_words' in their names.
    """
    # Get the list of all csv files in the folder
    file_list = glob(os.path.join(folder_path, f"*_merge{'_words' if load_words else ''}.csv"))
    
    # Initialize an empty list to store the data
    data = []

    # Load each csv file and append it to the data list
    for file in file_list:
        df = pd.read_csv(file, na_values=['']) # one speaker name is 'NA'
        df['dyad'] = file.split('/')[-1].split('_')[0]
        data.append(df)

    # A remettre si besoin
    # data = pd.concat(data, axis=0).reset_index(drop=True).drop(columns=['?'])
    # print(data.shape)

    # Get the list of column names that do not start with certain prefixes
    # plabels = [col for col in data.columns if not any([col.startswith(c) for c in ['dyad', 'ipu_id','speaker','start','stop','text', 'duration']])]
    # print(data[plabels].sum(axis=0) / data.shape[0])

    return data

# Function to filter out the initial common conversations
def filter_after_jokes(df_ipu:pd.DataFrame):
    """
    Filter out the initial common conversations from the DataFrame.
    The initial few ipus are common to all conversations and are therefore not useful for analysis.
    """
    # Find the end of the common conversations
    jokes_end = df_ipu[df_ipu.text.apply(lambda x: False if isinstance(x, float) else (
                ('il y avait un âne' in x) or ("qui parle ça c'est cool" in x)))].groupby('dyad').agg(
                {'ipu_id':'max'}).to_dict()['ipu_id']

    # Filter out the common conversations
    return df_ipu[df_ipu.apply(lambda x: x.ipu_id > jokes_end.get(x.dyad,0), axis = 1)], jokes_end
