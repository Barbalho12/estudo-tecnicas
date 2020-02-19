import glob
import pandas as pd
import os

def readDataframes(folder):
    """Ler os dataframes a partir dos arquivos CSV na pasta passada como argumento"""
    dataframes = {}
    files = sorted(glob.glob(folder+'/*.csv'))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        dataframes[name] = pd.read_csv(f) 
    return dataframes


dataframes = readDataframes('csv')


for key, dt in dataframes.items():
    print(key, dt.shape)
    
