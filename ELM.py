#Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers

#Preparação do dataset
data = pd.read_csv("brodatz_img_matrix.csv") #Lendo o CSV

x_data = data.iloc[:, 1:].values.astype('float32') #Features
labels = data.iloc[:, 0].values.astype('int32')    #labels

CLASSES = 111    #Número de classes
#Definindo a matriz de labels - one-hot-encode (quantidade de dados X quantidade de classes)
y_data = np.zeros([labels.shape[0], CLASSES])
for i in range(labels.shape[0]):
        y_data[i][labels[i]-1] = 1
        

#Dividindo o dataset em treino e teste
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

#Quantidade de features em cada dado
INPUT_LENGHT = x_train.shape[1] # 256 (256 tons de cinza) 
val = np.arange(1,1001) #Vamos rodar o modelo para 1000 quantidade de neurônios 

#Treinando a camada oculta
def input_to_hidden(x):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) #Função de ativação (ReLU)
    return a
    

#Função para predição
def predict(x):
    x = input_to_hidden(x)
    y = np.dot(x, Wout)
    return y
    
acc_vec = []
#executando para quantidade de neuronios entre 1 e 1000
for j in val:
    HIDDEN_UNITS = j #quantidade de neuronios na camada oculta
    perf = []
    #Executando 10 vezes para estabilizar o resultado
    for h in range(10):
        Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS]) #Pesos de entrada
        X = input_to_hidden(x_train) #Output da camada oculta
    
        Xt = np.transpose(X)
        #inversa de Moore-Penrose - com regularização de Tikhonov
        Wout = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, y_train)) 
        
        #predição
        y = predict(x_test)
        correct = 0
        total = y.shape[0]
        for i in range(total):
            predicted = np.argmax(y[i])
            test = np.argmax(y_test[i]) #Selecionando o neuronio de output com maior peso
            #Contagem de predição correta
            correct = correct + (1 if predicted == test else 0)
        
        #Acurácia
        perf.append(correct/total)

    #Média de acurácia após rodar 10x
    acc_vec.append(np.mean(perf))
    
    
#Plotando a acurácia em função da quantidade de neuronios na camada oculta
fig = plt.figure(figsize=(8,8))
plt.grid(True)
plt.xlabel('Number of neurons in the hidden layer')
plt.ylabel('Accuracy')
plt.plot(val,acc_vec)
plt.show()
plt.savefig('hist_brodatz-performance.png')

#Selecionando a melhor acurácia
best_accur = max(acc_vec)
print(best_accur)
