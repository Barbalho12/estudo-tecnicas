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

def resumeDataframes(dataframes):
    for key, dt in dataframes.items():
        print(key, dt.shape)

def holdout(dataframes, frac=0.2):
    holdoutDataframes = {}
    for key, dt in dataframes.items():
        holdoutDataframes[key] = dt.sample(frac=frac, replace=True, random_state=1)
    return holdoutDataframes

def crosValidation(dataframes, kFold=10):
    crosValidationDataframes = {}
    #TODO implementar
    return crosValidationDataframes


dataframes = readDataframes('csv')

resumeDataframes(dataframes)

# Método holdout: 70% (treinamento) 30% (teste).
holdoutDataframes70 = holdout(dataframes, 0.3)

# Método holdout: 60% (treinamento) 40% (teste).
holdoutDataframes60 = holdout(dataframes, 0.4)

# Método k fold cross validation: 10 fold cros validation.
crosValidationDataframes = crosValidation(dataframes, 10)




