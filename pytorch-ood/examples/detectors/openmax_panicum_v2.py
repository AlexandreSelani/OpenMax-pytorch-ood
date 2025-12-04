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
from torchvision.models import resnet50,ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.dataset.img import PanicumDataset_pytorchOOD
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import PlainCNN_panicum
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, metricasImplementadas,AnaliseGrafica_OpenMax,Matriz_confusao_osr_dataset_outlier as mc
from pytorch_ood.utils.aux_dataset import *
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc
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
            
            max_values, predicted = torch.max(score, dim=1)
            predict = torch.where(max_values >= detector.epsilon, predicted, torch.zeros_like(predicted))

        
        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
    
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    #ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadas(predict=predicts, label=labels)

    #print(ood_metrics.compute())
    return metricas._metricas()
    

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
    nomeDataset = "panicum"

    analiseGrafica = AnaliseGrafica_OpenMax(nomeDataset)

    lr=0.001
    epochs = 3
    bs=4

    weights = ResNet50_Weights.DEFAULT
    
    panicum_kkc = ImageFolder(root="/home/alexandreselani/Desktop/Dataset_panicum/Dataset/Treino/",transform=weights.transforms())
    panicum_uuc = ImageFolder(root="/home/alexandreselani/Desktop/Dataset_panicum/Dataset/Teste/",transform=weights.transforms(),target_transform=ToUnknown())

    modelos = []
    fold_test_dataloaders = []
    fold_train_dataloaders = []

    SKFold = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

    for i, (train_idx, test_idx) in enumerate(SKFold.split(X=panicum_kkc.samples, y=panicum_kkc.targets)):
        torch.cuda.empty_cache()
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        model = model.to(device)

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            stratify=[panicum_kkc.targets[i] for i in train_idx],
            random_state=42,
        )

        
        val_dataset = Subset(panicum_kkc,val_idx)
        train_dataset = Subset(panicum_kkc,train_idx)

        test_kkc_dataset = Subset(panicum_kkc,test_idx)
        test_uuc_dataset = random_dataset(panicum_uuc,len(test_idx))
        test_dataset = ConcatDataset([test_kkc_dataset, test_uuc_dataset])

        train_dataloader = DataLoader(train_dataset,batch_size=bs,shuffle=True)
        val_dataloader = DataLoader(val_dataset,batch_size=bs,shuffle=True)
        test_dataloader = DataLoader(test_dataset,batch_size=bs,shuffle=False)

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
        plt.title(f'Evolução de Loss e Acurácia - Fold {i}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        

        dir = f"/home/alexandreselani/Desktop/OpenMax-pytorch-ood/resultados/Curvas validacao/"
        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir+f"Fold {i}.png")
        
        modelos.append(model)
        fold_test_dataloaders.append(test_dataloader)
        fold_train_dataloaders.append(train_dataloader)
    
    
    min_tailsize=0
    max_tailsize=1001
    alpha=1
    epsilons = [0.3,0.35,0.4]
    
    for epsilon in epsilons:

        f1s = {}
        f1s_std = {}

        accs = {}
        accs_std = {}

        uuc_accs = {}
        uuc_accs_std = {}

        inners = {}
        inners_std = {}

        outers = {}
        outers_std = {}

        halfpoints = {}
        halfpoints_std = {}

        val_accs = {}
        val_accs_std = {}

        val_losses = {}
        
        for tail in range(min_tailsize,max_tailsize+1,100):
            
            print(f"TAIL = {tail}")
            all_f1 = []
            all_acc=[]
            all_uuc_acc=[]
            all_inner=[]
            all_outer=[]
            all_halfpoint=[]
            all_val_acc=[]
            all_val_loss=[]

            for m in range(len(modelos)):
                all_targets = np.array([])
                torch.cuda.empty_cache()
                gc.collect()

                model = modelos[m]
                
                test_dataloader = fold_test_dataloaders[m]
                train_dataloader = fold_train_dataloaders[m]

                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1,epsilon=epsilon)
                detector.fit(train_dataloader, device=device)
                metricas = test(test_dataloader,detector)

                for (x, y) in test_dataloader:
                    for target in y:
                        all_targets= np.append(all_targets,target.detach().cpu())

                

                print(metricas)
                all_f1.append(metricas["F1 macro"])
                all_acc.append(metricas["accuracy"][0])
                all_uuc_acc.append(metricas["UUC Accuracy"][0])
                all_inner.append(metricas["inner metric"][0])
                all_outer.append(metricas["outer metric"][0])
                all_halfpoint.append(metricas["halfpoint"][0])


            f1s[tail]        = np.array(all_f1).mean()
            f1s_std[tail]    = np.array(all_f1).std()

            accs[tail]       = np.array(all_acc).mean()
            accs_std[tail]   = np.array(all_acc).std()

            inners[tail]     = np.array(all_inner).mean()
            inners_std[tail] = np.array(all_inner).std()

            outers[tail]     = np.array(all_outer).mean()
            outers_std[tail] = np.array(all_outer).std()

            uuc_accs[tail]       = np.array(all_uuc_acc).mean()
            uuc_accs_std[tail]   = np.array(all_uuc_acc).std()

            halfpoints[tail]        = np.array(all_halfpoint).mean()
            halfpoints_std[tail]    = np.array(all_halfpoint).std()
            
        
        df = pd.DataFrame({
        "tail": list(f1s.keys()),
        
        "f1_macro_medio": list(f1s.values()),
        #"f1_macro_std": list(f1s_std.values()),
        
        "acc_medio": list(accs.values()),
        #"acc_std": list(accs_std.values()),
        
        "uuc_acc_medio": list(uuc_accs.values()),
        #"uuc_acc_std": list(uuc_accs_std.values()),
        
        "inner_medio": list(inners.values()),
        #"inner_std": list(inners_std.values()),
        
        "outer_medio": list(outers.values()),
        #"outer_std": list(outers_std.values()),
        
        "halfpoint_medio": list(halfpoints.values()),
        #"halfpoint_std": list(halfpoints_std.values())
    })


        dir = f"/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum/Resultados OpenMax/"
        os.makedirs(dir, exist_ok=True)
        df.to_csv(f"/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum/Resultados OpenMax/Resultados_grid_search_panicum_tail_0_1000_epsilon_{epsilon}_alpha_{alpha}.csv", index=False)


    
if __name__ == '__main__':
    main()

