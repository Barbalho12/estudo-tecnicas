# coding=utf-8
import glob
import pandas as pd
import os
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing


def readDataframes(folder):
    """Ler os dataframes a partir dos arquivos CSV na pasta passada como argumento"""
    dataframes = {}
    files = sorted(glob.glob(folder+'/*.csv'))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        dataframes[name] = pd.read_csv(f) 
    return dataframes

# def resumeDataframes(dataframes):
#     for key, dt in dataframes.items():
#         print(key, dt.shape)
        # print(dt.columns[-1])

# def holdout(dataframes, frac=0.2):
#     holdoutDataframes = {}
#     for key, dt in dataframes.items():
#         holdoutDataframes[key] = dt.sample(frac=frac, replace=True, random_state=1)
#     return holdoutDataframes

def crosValidation(dataframes, kFold=10):
    crosValidationDataframes = {}
    #TODO implementar
    return crosValidationDataframes

dataframes = readDataframes('csv')

# resumeDataframes(dataframes)


# TODO Adincionar métodos
# classifiers = [
#     [
#         DecisionTreeClassifier(), 
#         DecisionTreeClassifier(criterion='gini', max_depth=3),
#         DecisionTreeClassifier(criterion="entropy", max_depth=3)
#     ],
#     [
#         GaussianNB()
#     ]
# ]

# TODO Preprocessing
# mm_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = mm_scaler.fit_transform(X_train)
# mm_scaler.transform(X_test)

classifiers = [[DecisionTreeClassifier()]]

def holdout(dataframe, test_size):
    cls_name = dt.columns[-1]
    cls = dt[cls_name].values
    df = dt.drop(columns=[cls_name])
    print(cls)
    print(df)
    dt_train, dt_test, cls_train, cls_test = train_test_split(df, cls, test_size=test_size)
    return {'dt_train':dt_train, 'dt_test':dt_test, 'cls_train':cls_train, 'cls_test':cls_test}

def calculateScore(method, dataset):
    clf = method.fit(dataset['dt_train'],dataset['cls_train'])
    y_pred = clf.predict(dataset['cls_test'])
    return metrics.accuracy_score(dataset['dt_test'], y_pred)

for classifier in classifiers:
    for key, dt in dataframes.items():
        i_60 = holdout(dt, 0.4)
        i_70 = holdout(dt, 0.3)
        # crosv = crosValidation(dt, 10)

        for method in classifier:
            # try:
                # print(key)
            a = calculateScore(method, i_60)
                # b = calculateScore(method, i_70)
                # c = calculateScore(method, crosv)
                # print('{}\t{}\t{}'.format(a,b,c))
                # print('ok')
            # except Exception as e:
                # pass
                # print("Unexpected error:", e)


# Árvore de decisão (mínimo 3):
#     DecisionTreeClassier();
#     DecisionTreeClassier(criterion=“gini", max_depth=3)
#     DecisionTreeClassier(criterion="entropy", max_depth=3)
#     Defaults: criterion=“gini”; max_depth=none
# Naive Bayes (mínimo 2):
#     GaussianNB()
#     Outras configurações.
# MultiLayerPerceptron (mínimo 3):
    # MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=300, activation = 'logistic')
    # Camadas escondidas (hidden_layer_sizes);
    # Número de épocas (max_iter);
    # Função de ativação (activation).
        # logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 +exp(-x)).
        # ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).