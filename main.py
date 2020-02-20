# coding=utf-8
import glob
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

"""Desabilitando warns para kfold =10"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def readDataframes(folder):
    """Ler os dataframes a partir dos arquivos CSV na pasta passada como argumento"""
    dataframes = {}
    files = sorted(glob.glob(folder+'/*.csv'))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        dataframes[name] = pd.read_csv(f) 
    return dataframes

dataframes = readDataframes('csv')

"""Classificadores usados e suas configurações"""
classifiers = {
    'DT' : {
        'Configuração 1' : DecisionTreeClassifier(), 
        'Configuração 2' : DecisionTreeClassifier(criterion='gini', max_depth=3),
        'Configuração 3' : DecisionTreeClassifier(criterion="entropy", max_depth=3)
    },
    'NB' : {
        'Configuração 1' : GaussianNB(),
        'Configuração 2' : MultinomialNB()
    },
    'MLP' : {
        'Configuração 1' : MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=300, activation = 'logistic'),
        'Configuração 2' : MLPClassifier(hidden_layer_sizes=100, max_iter=100, activation = 'logistic'),
        'Configuração 3' : MLPClassifier(hidden_layer_sizes=50, max_iter=50, activation = 'tanh')
    }
}

"""Aplica holdout e retorna conjuntos treino/teste"""
def holdout(df, test_size):
    cls_name = df.columns[-1]
    cls = df[cls_name].values
    df = df.drop(columns=[cls_name])
    dt_train, dt_test, cls_train, cls_test = train_test_split(df, cls, test_size=test_size)
    return {'dt_train':dt_train, 'dt_test':dt_test, 'cls_train':cls_train, 'cls_test':cls_test}

"""Calcula o score recebendo os conjuntos de treino/teste e o modelo"""
def calculateScore(method, dataset):
    clf = method.fit(dataset['dt_train'],dataset['cls_train'])
    return clf.score(dataset['dt_test'], dataset['cls_test'])

"""Calcula o score recebendo o dataframe e aplicando kfold"""
def calculateScoreKF(method, df, cv):
    cls_name = df.columns[-1]
    cls = df[cls_name].values
    df = df.drop(columns=[cls_name])
    scores = cross_val_score(method, df, cls, cv=cv)
    return max(scores)

"""Executa o experimento para todos os datasets/classificadore/configurações/méotdo"""
for kClassifier, classifier in classifiers.items():
    print(kClassifier)
    for key, dt in dataframes.items():
        dt = dt.apply(preprocessing.LabelEncoder().fit_transform)
        i_60 = holdout(dt, 0.4)
        i_70 = holdout(dt, 0.3)
        print('{}\t'.format(key), end = '')
        for kMethod, method in classifier.items():
            a = calculateScore(method, i_60)
            b = calculateScore(method, i_70)
            c = calculateScoreKF(method, dt, 10)
            print('{0:.4f}\t{0:.4f}\t{0:.4f}\t'.format(a,b,c), end = '')
        print()
