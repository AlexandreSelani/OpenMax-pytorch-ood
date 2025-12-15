"""
OpenMax
==============================

:class:`OpenMax <pytorch_ood.detector.OpenMax>` was originally proposed
for Open Set Recognition but can be adapted for Out-of-Distribution tasks.

.. warning:: OpenMax requires ``libmr`` to be installed, which is broken at the moment. You can only use it
   by installing ``cython`` and ``numpy``, and ``libmr`` manually afterwards.


"""

"""label = -1 --> desconhecido
label >= 0 --> conhecido

O primeiro item dos logits é referente ao score para classe desconhecida, o restante é para as classes conhecidas

TODO: 
    REDE NEURAL MINHA: feito 
    LOOP DE TREINAMENTO E TESTES: feito
    ANALISE GRAFICA? feito
"""
from torch.utils.data import DataLoader,ConcatDataset
from torchvision.datasets import MNIST,Omniglot,ImageFolder
import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.dataset.img import PanicumDataset_pytorchOOD
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import PlainCNN_panicum
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, metricasImplementadas,AnaliseGrafica_OpenMax,Matriz_confusao_osr_dataset_outlier_cumulativa as mc
from pytorch_ood.utils.aux_dataset import *
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc
import copy
seed = 42
fix_random_seed(seed)

device = "cuda:0"

ood_metrics = OODMetrics()


def test(test_loader,detector):
    predicts=[]
    labels=[]

    for X, y in test_loader:
        
        #score eh a ativacao de todas as classes apos a openmax
        with torch.no_grad():
            score = detector(X.to(device))
            #print(score)
            max_values, predicted = torch.max(score, dim=1)
            predict = torch.where(max_values >= detector.epsilon, predicted, torch.zeros_like(predicted))

        
        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
    
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    #ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadas(predict=predicts, label=labels)

    #print(ood_metrics.compute())
    print(predicts,labels)
    return metricas._metricas(),predicts,labels
    
def train(train_loader,model,criterion,optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/(batch_idx+1), correct/total

def confusion_matrix(test_loader,targets_original,nome_classes_originais,UUC_classes,detector):
    predicts=[]
    labels=[]

    for X, y in test_loader:
        #score eh a ativacao de todas as classes apos a openmax
        with torch.no_grad():
            score = detector(X.to(device))
            
            max_values, predicted = torch.max(score, dim=1)
            predict = torch.where(max_values >= detector.epsilon, predicted, torch.zeros_like(predicted))

        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
    
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()

    matriz_confusao = mc(predicts,labels,targets_original,UUC_classes,nome_classes_originais)
    matriz_confusao.computa_matriz()
    matriz_confusao.exibe_matriz()

def validation(val_loader,model,criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return val_loss/(batch_idx+1), correct/total



def main():
    nomeDataset = "mnist_omniglot"
    # Diretório de saída
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_mnist_omniglot/Resultados OpenMax/"
    os.makedirs(output_dir, exist_ok=True)
    #analiseGrafica = AnaliseGrafica_OpenMax(nomeDataset)

    lr=0.0001
    epochs = 70
    bs=32

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(64),   # MUITO melhor que 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

    mnist_train = MNIST(root="./data/",download=True,transform=transform,train=True)
    mnist_test = MNIST(root="./data/",download=True,transform=transform,train=False)
    omniglot = Omniglot(root="./data/",download=True,transform=transform,target_transform=ToUnknown())

    
    model = alexnet()
    model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=10
    )
    model = model.to(device)

    train_dataset,val_dataset = validation_split(0.1,mnist_train)
    test_dataset = ConcatDataset([omniglot,mnist_test])
        

    train_dataloader = DataLoader(train_dataset,batch_size=bs,shuffle=True,num_workers=4)
    val_dataloader = DataLoader(val_dataset,batch_size=bs,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(test_dataset,batch_size=bs,shuffle=False,num_workers=4)

    print(f"Tamanho Validacao: {len(val_dataset)}")
    print(f"Tamanho treino: {len(train_dataset)}")
    print(f"Tamanho teste mnist: {len(mnist_test)}")
    print(f"Tamanho teste omniglot: {len(omniglot)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    t_loss,t_acc=[],[]
    v_loss,v_acc=[],[]

    for epoch in range(epochs):
    
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
        val_loss, val_acc = validation(val_dataloader,model,criterion)
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        v_loss.append(val_loss)
        v_acc.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
        print(f"VALIDATION || Epoch {epoch+1}/{epochs} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

    plt.figure(figsize=(8,5))
    e = range(0,epochs)
    plt.plot(e, t_loss, 'red', label='Train Loss')
    plt.plot(e, v_loss, 'orange', label='Val Loss')
    plt.plot(e, t_acc, 'blue', label='Train Acc')
    plt.plot(e, v_acc, 'purple', label='Val Acc')

    plt.xlabel('Épocas')
    plt.xticks(e)
    plt.ylabel('Valor')
    plt.ylim(0,2)
    plt.title(f'Evolução de Loss e Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    #DESCOMENTAR PARA PLOTAR
    
    
    plt.savefig(output_dir+f"Validacao.png")
        
    


    # Configurações
    min_tailsize = 0
    max_tailsize = 600
    step_tail = 100
    min_alpha = 1
    max_alpha = 10
    epsilons = [0.2,0.3,0.5,0.7,0.9]

    for alpha in range(min_alpha,max_alpha+1):

        for epsilon in epsilons:
            
            results_by_tail = {}
            
            matrizes_confusao = [None for _ in range(min_tailsize, max_tailsize+1, step_tail)]
            #print(matrizes_confusao)
        
            torch.cuda.empty_cache()
            gc.collect()
            
            model.eval()

            all_targets = np.array([])

            for (_, y) in test_dataloader:
                for target in y:
                    all_targets= np.append(all_targets,target.detach().cpu())

            
            for idx,tail in enumerate(range(min_tailsize, max_tailsize+1, step_tail)):

                # Fit e Teste
                # Nota: O Fit ainda pode ser demorado, mas economizamos o load do modelo
                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                
                # Ajuste (Fit) - Geralmente precisa passar os dados de treino para calcular os centros/weibulls
                detector.fit(train_dataloader, device=device)
                
                metricas,predicts,targets_test = test(test_dataloader, detector)
                
                matriz = mc(predicts,targets_test,all_targets,[],["Omniglot",0,1,2,3,4,5,6,7,8,9])
                matriz.computa_matriz()
                matrizes_confusao[idx] = matriz
            
                # Guardar métricas deste fold para este tail
                results_by_tail[tail] = {
                "tail": tail,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0]
                }


                del detector
                gc.collect()
                torch.cuda.empty_cache()
            
            final_data = []

            for tail in sorted(results_by_tail.keys()):
                metrics = results_by_tail[tail]
                final_data.append(metrics)

            df = pd.DataFrame(final_data)

            
            # Nome dinâmico correto
            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            filename_csv = f"Resultados_grid_mnist_omniglot_tail_{min_tailsize}_{max_tailsize}_eps_{epsilon}_alpha_{alpha}.csv"
            organized_dir = os.path.join(output_dir, pasta)

            os.makedirs(organized_dir, exist_ok=True)

            csv_path = os.path.join(organized_dir,filename_csv)
            
            for idx,m in enumerate(matrizes_confusao):
                m.exibe_matriz(dir=organized_dir,name=f"tail_{idx*step_tail}")

            df.to_csv(csv_path, index=False)
            print(f"Arquivo salvo: {csv_path}")


    
if __name__ == '__main__':
    main()



