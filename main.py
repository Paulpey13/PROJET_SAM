import load_data

transcr_path='paco-cheese/transcr'
data=load_data.load_all_ipus(folder_path=transcr_path,load_words=True)

#print(data.tail(10))