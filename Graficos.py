import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.pyplot import bar


# Cria um gráfico a partir dos parâmetros de abscissas e ordenadas passados como parâmetro e salva num arquivo PNG na raiz do programa.
def mostrarGraficoLinhas(line1, line2, label_line1, label_line2, x_kfolds, y_values, x_label, y_label, ):
    figure(num=None, figsize=(8, 6))
    plt.plot(x_kfolds, line1, '.', label=label_line1, linestyle='-')
    plt.plot(x_kfolds, line2, '.', label=label_line2,
             linestyle='-')  # Recebe dois arrays, um sera do K e outro dos resultados
    # plt.ylim(0, 1)
    if len(y_values) > 0:
        print("Entrou no ticks")
        plt.yticks(y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs. " + y_label)
    plt.legend()
    plt.savefig('teste-' + y_label + '.png')
    plt.show()


def mostrarGraficoLinhasMult(lines, labels, x_kfolds, y_values, x_label, y_label, ):
    figure(num=None, figsize=(10, 8))
    for i in range(len(lines)):
        plt.plot(x_kfolds, lines[i], '.', label=labels[i], linestyle='-')
    # plt.ylim(0, 1)
    if len(y_values) > 0:
        print("Entrou no ticks")
        plt.yticks(y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs. " + y_label)
    plt.legend()
    plt.savefig('teste-' + y_label + '.png')
    plt.show()


# Cria um gráfico de barras a aprtir dos parâmetros passados.
def mostrarGraficoBarras(classif, accs, label, xlabel, yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    x_pos = np.arange(len(classif))
    accs = np.array(accs)
    plt.bar(x_pos, accs, align='center', alpha=0.4)
    plt.xticks(x_pos, classif)
    plt.yticks(yticks)
    plt.ylabel('Acurácia')
    plt.xlabel(xlabel)
    plt.title(label)
    plt.savefig(label.strip().replace(" ", "_") + '.png')
    plt.show()
