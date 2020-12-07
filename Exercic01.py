# Fonte: https://scikit-learn.org/stable/modules/cross_validation.html


from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import ArithmeticFunctions as art
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

skf = StratifiedKFold(n_splits=10, random_state=23, shuffle=True)

dados = pd.read_csv("data/ObesityDataSet.csv", encoding='utf-8')
Y = dados['NObeyesdad']
X = dados.drop(columns=['NObeyesdad'])

le = preprocessing.LabelEncoder()
for column_name in X.columns:
    if X[column_name].dtype == object:
        X[column_name] = le.fit_transform(X[column_name])
    else:
        pass

AcuraciaKNN = []
AcuraciaAD = []
AcuraciaRNA = []

PrecisaoKNN = []
PrecisaoAD = []
PrecisaoRNA = []

RecallKNN = []
RecallAD = []
RecallRNA = []

for train, test in skf.split(X, Y):  # este loop tem o número de splits do objeto skf = 10
    # print("treino")
    # print(train)  # ===> possui a posição dos objetos do conjunto original para compor o conjunto de treino
    # print("teste")
    # print(test)  # ===> possui a posição dos objetos do conjunto original para compor o conjunto de teste
    KNN = KNeighborsClassifier(n_neighbors=2)
    AD = DecisionTreeClassifier()
    mlp = MLPClassifier(activation='logistic',
                        solver='adam',
                        alpha=1e-5,
                        learning_rate='invscaling',
                        max_iter=2000,
                        hidden_layer_sizes=(200,))

    X_train, X_test = X.values[train], X.values[test]
    y_train, y_test = Y[train], Y[test]

    # print(X_train)

    KNN.fit(X_train, y_train)

    resultKNN = KNN.predict(X_test)
    acuraciaKNN = accuracy_score(y_test, resultKNN)
    recallKNN = recall_score(y_test, resultKNN, average='macro')
    precisionKNN = precision_score(y_test, resultKNN, average='macro')

    AcuraciaKNN.append(acuraciaKNN)
    RecallKNN.append(recallKNN)
    PrecisaoKNN.append(precisionKNN)

    AD.fit(X_train, y_train)
    resultAD = AD.predict(X_test)

    acuraciaAD = accuracy_score(y_test, resultAD)
    recallAD = recall_score(y_test, resultAD, average='macro')
    precisionAD = precision_score(y_test, resultAD, average='macro')

    AcuraciaAD.append(acuraciaAD)
    RecallAD.append(recallAD)
    PrecisaoAD.append(precisionAD)

    mlp.fit(X_train, y_train)
    resultRNA = mlp.predict(X_test)

    acuraciaRNA = accuracy_score(y_test, resultRNA)
    recallRNA = recall_score(y_test, resultRNA, average='macro')
    precisionRNA = precision_score(y_test, resultRNA, average='macro')

    AcuraciaRNA.append(acuraciaRNA)
    RecallRNA.append(recallRNA)
    PrecisaoRNA.append(precisionRNA)

    # print(X_test)
    # Montar X_train e y_train
    # Montar X_test e y_test

    # Executar os 3 algoritmos e as variações
    # Calcular as métricas acurácia, precisão e recall dos 3 algoritmos e variações
    # Ex: 3 algs - K-NN com K=1, AD, RNA com 200 neurônios

AvgAcuraciaKNN = art.Average(AcuraciaKNN)
AvgRecallKNN = art.Average(RecallKNN)
AvgPrecisionKNN = art.Average(PrecisaoKNN)
StdAcuraciaKNN = art.standardDeviation(AcuraciaKNN)
StdRecallKNN = art.standardDeviation(RecallKNN)
StdPrecisionKNN = art.standardDeviation(PrecisaoKNN)

AvgAcuraciaAD = art.Average(AcuraciaAD)
AvgRecallAD = art.Average(RecallAD)
AvgPrecisionAD = art.Average(PrecisaoAD)
StdAcuraciaAD = art.standardDeviation(AcuraciaAD)
StdRecallAD = art.standardDeviation(RecallAD)
StdPrecisionAD = art.standardDeviation(PrecisaoAD)

AvgAcuraciaRNA = art.Average(AcuraciaRNA)
AvgRecallRNA = art.Average(RecallRNA)
AvgPrecisionRNA = art.Average(PrecisaoRNA)
StdAcuraciaRNA = art.standardDeviation(AcuraciaRNA)
StdRecallRNA = art.standardDeviation(RecallRNA)
StdPrecisionRNA = art.standardDeviation(PrecisaoRNA)




print("KNN: ")
print("Acuracia")
print(AcuraciaKNN)
print(AvgAcuraciaKNN)
print([AvgAcuraciaKNN,StdAcuraciaKNN])
print("Recall")
print(RecallKNN)
print([AvgRecallKNN,StdRecallKNN])
print("Precisão")
print(PrecisaoKNN)
print([AvgPrecisionKNN,StdPrecisionKNN])
print("")
print("Arvore de decisão:")
print("Acuracia")
print(AcuraciaAD)
print([AvgAcuraciaAD,StdAcuraciaAD])
print("Recall")
print(RecallAD)
print([AvgRecallAD,StdRecallAD])
print("Precisão")
print(PrecisaoAD)
print([AvgPrecisionAD,StdPrecisionAD])
print("")
print("RNA:")
print("Acuracia")
print(AcuraciaRNA)
print([AvgAcuraciaRNA,StdAcuraciaRNA])
print("Recall")
print(RecallRNA)
print([AvgRecallRNA,StdRecallRNA])
print("Precisão")
print(PrecisaoRNA)
print([AvgPrecisionRNA,StdPrecisionRNA])

# Saída do loop: 3 vetores de acurácia com 10 valores para cada algoritmo / 3 vetores de precisão / 3 vetores de recall
# Calcular média e desvio padrão das 3 métricas
