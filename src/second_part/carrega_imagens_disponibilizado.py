import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
#ATENÇÃO:
#Salve este algoritmo no mesmo diretório no qual a pasta chamada RecFac está.


#A tarefa nessa etapa é realizar o reconhecimento facial de 20 pessoas

#Dimensões da imagem. Você deve explorar esse tamanho de acordo com o solicitado no pdf.
dimensao = 50 #50 signica que a imagem terá 50 x 50 pixels. ?No trabalho é solicitado para que se investigue dimensões diferentes:
# 50x50, 40x40, 30x30, 20x20, 10x10 .... (tua equipe pode tentar outros redimensionamentos.)

#Criando strings auxiliares para organizar o conjunto de dados:
pasta_raiz = "RecFac"
caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
caminho_pessoas.pop(0)

C = 20 #Esse é o total de classes 
X = np.empty((dimensao*dimensao,0)) # Essa variável X será a matriz de dados de dimensões p x N. 
Y = np.empty((C,0)) #Essa variável Y será a matriz de rótulos (Digo matriz, pois, é solicitado o one-hot-encoding).
for i,pessoa in enumerate(caminho_pessoas):
    imagens_pessoa = os.listdir(pessoa)
    for imagens in imagens_pessoa:

        caminho_imagem = os.path.join(pessoa,imagens)
        imagem_original = cv2.imread(caminho_imagem,cv2.IMREAD_GRAYSCALE)
        imagem_redimensionada = cv2.resize(imagem_original,(dimensao,dimensao))

        #A imagem pode ser visualizada com esse comando.
        # No entanto, o comando deve ser comentado quando o algoritmo for executado
        # cv2.imshow("eita",imagem_redimensionada)
        # cv2.waitKey(0)

        #vetorizando a imagem:
        x = imagem_redimensionada.flatten()

        #Empilhando amostra para criar a matriz X que terá dimensão p x N
        X = np.concatenate((
            X,
            x.reshape(dimensao*dimensao,1)
        ),axis=1)
        

        #one-hot-encoding (A EQUIPE DEVE DESENVOLVER)
        y = -np.ones((C,1))
        y[i,0] = 1

        Y = np.concatenate((
            Y,
            y
        ),axis=1)
    

    


# Normalização dos dados (A EQUIPE DEVE ESCOLHER O TIPO E DESENVOLVER):

# Início das rodadas de monte carlo
#Aqui podem existir as definições dos hiperparâmetros de cada modelo.

rodadas = 50

for i in range(rodadas):
    pass
    #Embaralhar X e Y

    #Particionar em Treino e Teste (80/20)


    #Treinameno Modelo Perceptron Simples 

    #Teste Modelo Perceptron Simples



    #Treinameno Modelo ADALINE

    #Teste Modelo Modelo ADALINE



    #Treinameno Modelo MLP Com topologia já definida

    #Teste Modelo MLP Com topologia já definida



#MÉTRICAS DE DESEMPENHO para cada modelo:
#Tabela
#Matriz de confusão
#Curvas de aprendizagem