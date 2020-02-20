# coding=utf-8
import glob
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def readDataframes(folder):
    """Ler os dataframes a partir dos arquivos CSV na pasta passada como argumento"""
    dataframes = {}
    files = sorted(glob.glob(folder+'/*.csv'))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        dataframes[name] = pd.read_csv(f) 
    return dataframes

def crosValidation(dataframes, kFold=10):
    crosValidationDataframes = {}
    #TODO implementar
    return crosValidationDataframes

dataframes = readDataframes('csv')

# TODO Adincionar métodos
classifiers = {
    'DT' : {
        'Configuração 1' : DecisionTreeClassifier(), 
        'Configuração 2' : DecisionTreeClassifier(criterion='gini', max_depth=3),
        'Configuração 3' : DecisionTreeClassifier(criterion="entropy", max_depth=3)
    },
    'DecisionTree' : {
        'Configuração 1' : GaussianNB()
    },
    'MLP' : {
        'Configuração 1' : MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=300, activation = 'logistic')
    }
}

def holdout(df, test_size):
    cls_name = df.columns[-1]
    cls = df[cls_name].values
    df = df.drop(columns=[cls_name])
    dt_train, dt_test, cls_train, cls_test = train_test_split(df, cls, test_size=test_size)
    return {'dt_train':dt_train, 'dt_test':dt_test, 'cls_train':cls_train, 'cls_test':cls_test}

def calculateScore(method, dataset):
    clf = method.fit(dataset['dt_train'],dataset['cls_train'])
    return clf.score(dataset['dt_test'], dataset['cls_test'])

for kClassifier, classifier in classifiers.items():
    print(kClassifier)
    for key, dt in dataframes.items():
        dt = dt.apply(preprocessing.LabelEncoder().fit_transform)
        print(key)
        i_60 = holdout(dt, 0.4)
        i_70 = holdout(dt, 0.3)
        # crosv = crosValidation(dt, 10)

        for kMethod, method in classifier.items():
            print(kMethod)
            a = calculateScore(method, i_60)
            b = calculateScore(method, i_70)
            # c = calculateScore(method, crosv)
            # print('{}\t{}\t{}'.format(a,b,c))