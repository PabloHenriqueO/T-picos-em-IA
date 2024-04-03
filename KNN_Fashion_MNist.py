import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# Carregar o conjunto de dados Fashion MNIST
(x_treino, y_treino), (x_teste, y_teste) = fashion_mnist.load_data()

# Normalizar os valores dos pixels para o intervalo [0, 1]
x_treino = x_treino / 255.0
x_teste = x_teste / 255.0

# Inicializar os valores de K a serem testados
valores_k = [3, 5, 7]

for k in valores_k:
    print(f"Analisando K = {k}")
    # Inicializar o classificador KNN
    knn = KNeighborsClassifier(n_neighbors=k)

    # Dividir os dados de teste em subconjuntos
    x_teste_sub, _, y_teste_sub, _ = train_test_split(x_teste, y_teste, test_size=0.2, random_state=42)

    # Ajustar o modelo aos dados de treinamento normalizados
    x_treino_flatten = x_treino.reshape(-1, 28*28)
    knn.fit(x_treino_flatten, y_treino)

    # Fazer previsões nos dados de teste normalizados
    x_teste_sub_flatten = x_teste_sub.reshape(-1, 28*28)
    y_pred = knn.predict(x_teste_sub_flatten)

    # Construir a matriz de confusão
    matriz_confusao = confusion_matrix(y_teste_sub, y_pred)

    # Exibir a matriz de confusão
    plt.figure()
    plot_confusion_matrix(knn, x_teste_sub_flatten, y_teste_sub)
    plt.title(f'Matriz de Confusão para K = {k}')
    plt.show()

    # Avaliar o modelo usando validação cruzada
    scores = cross_val_score(knn, x_treino_flatten, y_treino, cv=5)
    print(f"Pontuações de validação cruzada para K = {k}: {scores}")
    print(f"Pontuação média de validação cruzada para K = {k}: {np.mean(scores)}")