import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

TOTAL_KFOLDS = 4

kfolds = [i + 1 for i in range(TOTAL_KFOLDS)]

# Metricas utilizadas no cross validate
metricas = ['accuracy', 'balanced_accuracy']

# Lendo o csv
dados = pd.read_csv("data/winequality-white.csv", sep=';', encoding='utf-8')

# Separando a coluna target(X) dos dados de predição(Y)
Y = dados['quality']
X = dados.drop(columns=['quality'])

neighbors = [3, 5, 7, 9, 11, 13, 15]
x_treino, x_validacao, y_treino, y_validacao = train_test_split(X, Y, test_size=0.33, random_state=23)

# For variando o numero de neighbors em um knn e gerando os numeros para comparação.
for neighborsNumber in neighbors:

    #Treinando o calssificador
    knn = KNeighborsClassifier(n_neighbors=neighborsNumber)
    scores = cross_validate(knn, x_treino, y_treino, cv=TOTAL_KFOLDS, scoring=metricas)
    print(scores)
    knn.fit(x_treino, y_treino)

    #Usando o classificador para prever os resultados deos dados de validação
    results = knn.predict(x_validacao)

    #Gerando a acuracia com o resultado da predição
    acuracia = accuracy_score(y_validacao, results)

    print("Acuracia em KNN com " + str(neighborsNumber) + " vizinhos: ")
    print(acuracia)

clf = DecisionTreeClassifier()
scores = cross_validate(clf, x_treino, y_treino, cv=TOTAL_KFOLDS, scoring=metricas)
print(scores)
clf.fit(x_treino, y_treino)
resultsCLF = clf.predict(x_validacao)
acuraciaCLF = accuracy_score(y_validacao, resultsCLF)

print("Acuracia em Decision tree: ")
print(acuraciaCLF)
