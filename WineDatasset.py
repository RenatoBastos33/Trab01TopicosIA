import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import Graficos as grf

TOTAL_KFOLDS = 4

kfolds = [i + 1 for i in range(TOTAL_KFOLDS)]

# Metricas utilizadas no cross validate
metricas = ['accuracy', 'balanced_accuracy']

# Array dos nomes dos classificadores usados
classificadores = ['K-Nearest Neighbors', 'Decision tree']

# Lendo o csv
dados = pd.read_csv("data/winequality-white.csv", sep=';', encoding='utf-8')

# Separando a coluna target(X) dos dados de predição(Y)
Y = dados['quality']
X = dados.drop(columns=['quality'])

# print(X)
print(Y.values[0])

# Cria os arrays com os numeros de vizinhos que vão ser testados e os arrays de resultados
neighbors = [3, 5, 7, 9, 11, 13, 15]
resultsKNN = []
acuraciaKNN = []

# Separa o dataset em treino e validação
x_treino, x_validacao, y_treino, y_validacao = train_test_split(X, Y, test_size=0.33, random_state=23)

i = 0
# For variando o numero de neighbors em um knn e gerando os numeros para comparação.
for neighborsNumber in neighbors:
    # Treinando o calssificador
    knn = KNeighborsClassifier(n_neighbors=neighborsNumber)
    scores = cross_validate(knn, x_treino, y_treino, cv=TOTAL_KFOLDS, scoring=metricas)
    # print(scores)
    resultsKNN.append(scores)
    knn.fit(x_treino, y_treino)

    # Usando o classificador para prever os resultados deos dados de validação
    results = knn.predict(x_validacao)

    # Gerando a acuracia com o resultado da predição
    acuracia = accuracy_score(y_validacao, results)
    acuraciaKNN.append(acuracia)
    i = i + 1
    # print("Acuracia em KNN com " + str(neighborsNumber) + " vizinhos: ")
    # print(acuracia)

print("Acuracia KNN: ")
print(acuraciaKNN)

# Gera o grafico de barras comparando a acuracia entre os resultados do KNN
grf.mostrarGraficoBarras(neighbors, acuraciaKNN, "Acurácia variando os vizinhos ", "Vizinhos")

melhorAcuracia = acuraciaKNN[0]
melhorAcuraciaIndex = 0

for j in range(len(acuraciaKNN)):
    if (acuraciaKNN[j] > melhorAcuracia):
        melhorAcuracia = acuraciaKNN[j]
        melhorAcuraciaIndex = j

clf = DecisionTreeClassifier()
scores = cross_validate(clf, x_treino, y_treino, cv=TOTAL_KFOLDS, scoring=metricas)
print(scores)
clf.fit(x_treino, y_treino)
resultsCLF = clf.predict(x_validacao)
acuraciaCLF = accuracy_score(y_validacao, resultsCLF)

# Criando um grafico de linha comparando o resultado dos treinamentos do KNN e decision tree
grf.mostrarGraficoLinhas(resultsKNN[melhorAcuraciaIndex]['test_accuracy'],
                         scores['test_accuracy'],
                         'K-Nearest Neighbors',
                         'Decision tree',
                         kfolds,
                         [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                         "Numero do K fold", "Acurácia")

# Criando grafico para comparar a acuracia do KNN com o Decision tree
grf.mostrarGraficoBarras(classificadores,
                         [acuraciaKNN[melhorAcuraciaIndex], acuraciaCLF],
                         "Acurácia na validação",
                         "Classificadores")

grf.mostrarGraficoLinhas(resultsKNN[melhorAcuraciaIndex]['fit_time'],
                         resultsKNN[melhorAcuraciaIndex]['score_time'],
                         'Fit time',
                         'Score time',
                         kfolds,
                         [0.000, 0.005, 0.000, 0.010, 0.015, 0.020, 0.025, 0.030, 0.030, 0.035],
                         "Fit time vs Score time", "Tempo(ms)")

print("Acuracia em Decision tree: ")
print(acuraciaCLF)
