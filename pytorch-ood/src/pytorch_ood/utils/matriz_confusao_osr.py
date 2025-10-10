import matplotlib.pyplot as plt
import torch
import numpy as np



class Matriz_confusao_osr:
    def __init__(self,predict,target_test,target_original,UUC_classes,col_labels):
        self.predict = predict
        self.target_test = target_test+1
        self.target_original=target_original+1
        self.UUC_classes = np.array(UUC_classes)+1
        self.col_labels = col_labels
        self.matriz=None
        self.mapa_de_linhas = self.mapear_classes()

    def mapear_classes(self):
        mapa_de_linhas = {}

        # Linha 0 reservada para "unknown" (classes fora de UUC_classes)
        linha_idx = 1  # Começa em 1 para as conhecidas

        for c in np.unique(self.target_original):
            if c not in self.UUC_classes and c!=0:
                mapa_de_linhas[c] = linha_idx
                linha_idx += 1
            else:
                mapa_de_linhas[c] = 0

        return mapa_de_linhas

    def computa_matriz(self):
        print(self.mapa_de_linhas)

        colunas = len(np.unique(self.target_original))
        linhas = len(np.unique(self.target_test))

        self.matriz=np.zeros((linhas,colunas)) # colunas: Omniglot, M0,M1,...,M9 -- targets reais
                                                #linhas: Unknown, classes conhecidas (MNIST - UUC) -- predicoes
        
        for predict, target_original in zip(self.predict,self.target_original):
            predict = int(predict)
            target_original = int(target_original)
            linha = self.mapa_de_linhas[predict]
            coluna = target_original
            self.matriz[linha][coluna]+=1
    
    def exibe_matriz(self):
        if self.matriz is None:
            print("Matriz não computada ainda.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(self.matriz, interpolation='nearest', cmap='Blues')
        fig.colorbar(cax)

        ax.set_title("Matriz de Confusão OSR", pad=20)

        

        # Eixo Y = previsão (linha 0 é "desconhecido")
        linhas_ordenadas = sorted(self.mapa_de_linhas.items(), key=lambda x: x[1])
        row_labels = ['Desconhecido'] + [str(classe) for classe, idx in linhas_ordenadas if idx != 0]

        ax.set_xticks(np.arange(len(self.col_labels)))
        ax.set_xticklabels(self.col_labels)

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        # Coloca os labels do eixo X no topo
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        # Adiciona os números nas células
        for i in range(self.matriz.shape[0]):
            for j in range(self.matriz.shape[1]):
                valor = int(self.matriz[i, j])
                if valor > 0:
                    ax.text(j, i, str(valor), ha='center', va='center', color='white')

        ax.set_xlabel("Classe Real (Topo)")
        ax.set_ylabel("Classe Prevista")
        plt.tight_layout()
        plt.show()
        plt.savefig("fshdf")
            
            
                