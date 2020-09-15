import numpy as np
from PIL import Image
def matrixcompress (B,k):                        # B=matriz original, k=número de autovetores da reconstrução
    A = B-B.mean(axis=1, keepdims=True)          # Subtração da média de cada linha
    n = len(A[0])                                # Número de colunas
    m = len(A[:,0])                              # Número de linhas
    if k>min(m,n):                               # Mais autovetores que linhas ou colunas = inválido
        print("Número inválido de autovetores!")
    else:
        AT = np.transpose(A)                     # Transposta de A
        C = (A.dot(AT))/(n-1)                    # Matriz de Covariância
        avalC, avetC = np.linalg.eig(C)          # Autovalores, autovetores da matriz de covariância
        idx = np.argsort(avalC)                  # Ordenando autovalores
        avalC = avalC[idx].real                  # em ordem crescente e
        avetC = avetC[:,idx].real                # os autovetores associados
        U = np.array([])                         # Matrizes vazias
        V = np.array([])                         # para a concatenação
        for i in range(1,k+1):                   # Concatenando os k
            x = avetC[:,m-i]                     # autovetores associados
            U = np.concatenate([U,x])            # aos k maiores autovalores
        U = U.reshape(k,m).T                     # Ajustando as dimensões da matriz de projeção
        for j in range(0,n):
            y = A[:,j]                           # Vetor coluna original   
            yrec = U.dot(np.transpose(U).dot(y)) # Reconstruindo cada coluna da matriz original
            V = np.concatenate([V,yrec])         # Concatenando as reconstruções
        V = V.reshape(n,m).T                     # Ajustando as dimensões da matriz reconstruída
        return V+B.mean(axis=1, keepdims=True)   # Matriz reconstruída

img = Image.open('input.png')                    # Abrindo a imagem original
imgcor = Image.Image.split(img)                  # Separando os canais de cor
R = matrixcompress(np.asarray(imgcor[0]),300)    # Comprimindo o canal vermelho
G = matrixcompress(np.asarray(imgcor[1]),300)    # Comprimindo o canal verde
B = matrixcompress(np.asarray(imgcor[2]),300)    # Comprimindo o canal azul
RGBmat = np.dstack((R,G,B))                      # Juntanto os canais comprimidos
RGB = Image.fromarray(np.uint8(RGBmat))          # Convertendo de matriz para imagem
RGB.save('output300.png')                        # Salvando a imagem final