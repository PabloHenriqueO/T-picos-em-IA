import numpy as np
import pandas as pd
import seaborn as s
import matplotlib.pyplot as plote
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import multivariate_normal

#Carregamento da base de dados
wine = load_wine()

#Criação do dataframe
data = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
data['target'] = wine['target']

#Separação das classes de treinamento e testes
indices = np.random.permutation(data.shape[0])
div = int(0.2 * len(indices))  #20% para treinamento e 80% para testes
desen_id, test_id = indices[:div], indices[div:]
cj_desen, cj_test = data.loc[desen_id, :], data.loc[test_id, :]

#Extração de features e rótulos para treinamento e teste
xd = cj_desen.drop('target', axis=1).to_numpy()
yd = cj_desen.target.to_numpy()
xt = cj_test.drop('target', axis=1).to_numpy()
yt = cj_test.target.to_numpy()

#Cálculo da probabilidade a priori de cada classe
classes = len(np.unique(yd))
probabilidadeC = np.zeros(classes)
for i in range(classes):
    indiceC = np.where(yd == i)
    probabilidadeC[i] = len(indiceC[0]) / len(yd)

#Inicialização da matriz para armazenar as probabilidades de classe
matrizProbabilidade = np.zeros((xt.shape[0], classes))

#Cálculo das probabilidades de classe usando distribuição gaussiana multivariada
for i in range(classes):
    indiceC = np.where(yd == i)
    dadosDeTreino = xd[indiceC]
    media = np.mean(dadosDeTreino, axis=0)
    covariancia = np.cov(dadosDeTreino, rowvar=False)

    for j in range(xt.shape[0]):
        observation = xt[j]
        matrizProbabilidade[j, i] = multivariate_normal.pdf(observation, mean=media, cov=covariancia, allow_singular=True) * probabilidadeC[i]

#Previsão das classes com base nas probabilidades
y_pred = np.argmax(matrizProbabilidade, axis=1)

#Cálculo da acurácia das previsões
accuracy = accuracy_score(yt, y_pred)

#Geração da matriz de confusão
cm1 = confusion_matrix(yt, y_pred)

#Visualização da matriz de confusão
plote.figure(figsize=(8, 6))
s.heatmap(cm1, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
plote.xlabel('Previsões')
plote.ylabel('Valores Verdadeiros')
plote.title('Matriz de Confusão')
plote.show()

print(f'Acurácia: {accuracy:.2f}')
