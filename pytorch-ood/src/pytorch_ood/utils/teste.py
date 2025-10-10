from matriz_confusao_osr import *

a = Matriz_confusao_osr([0,1,0,4,4,5,6,7,5,0,0],[0,1,0,3,4,5,6,7,0,0,0],[0,1,2,3,4,5,6,7,8,9,10],[2,8,9,10],col_labels=["omniglot",1,2,3,4,5,6,7,8,9,10])
a.computa_matriz()
a.exibe_matriz()