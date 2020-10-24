import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.pyplot import bar

#Cria um gráfico a partir dos parâmetros de abscissas e ordenadas passados como parâmetro e salva num arquivo PNG na raiz do programa.
def mostrarGraficoLinhas(y_knn, y_dt, x_kfolds, y_values, x_label, y_label):
    figure(num=None, figsize=(8, 6))
    plt.plot(x_kfolds, y_dt, '.', label='Decision tree',linestyle='-')  # Recebe dois arrays, um sera do K e outro dos resultados
    plt.plot(x_kfolds, y_knn, '.', label='K-Nearest Neighbors', linestyle='-')
    plt.ylim(0, 1)
    plt.yticks(y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs. " + y_label)
    plt.legend()
    plt.savefig('teste-' + y_label + '.png')
    plt.show()

#Cria um gráfico de barras a aprtir dos parâmetros passados.
def mostrarGraficoBarras(classif, accs, label,xlabel):
    x_pos = np.arange(len(classif))
    accs = np.array(accs)
    plt.bar(x_pos, accs,  align='center', alpha=0.4)
    plt.xticks(x_pos, classif)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.ylabel('Acurácia')
    plt.xlabel(xlabel)
    plt.title(label)
    plt.savefig(label.strip().replace(" ","_") + '.png')
    plt.show()