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
    nomeDataset = "panicum"
    # Diretório de saída
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum/Resultados OpenMax/"
    os.makedirs(output_dir, exist_ok=True)
    #analiseGrafica = AnaliseGrafica_OpenMax(nomeDataset)

    lr=0.0001
    epochs = 40
    bs=16

    weights = ResNet50_Weights.DEFAULT
    
    panicum_kkc = ImageFolder(root="/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Todas/KKC/",transform=weights.transforms())
    panicum_uuc = ImageFolder(root="/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Todas/UUC/",transform=weights.transforms(),target_transform=ToUnknown())

    modelos = []
    fold_test_dataset = []
    fold_train = []

    SKFold = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

    for i, (train_idx, test_idx) in enumerate(SKFold.split(X=panicum_kkc.samples, y=panicum_kkc.targets)):
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        model = model.to(device)

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            stratify=[panicum_kkc.targets[v] for v in train_idx],
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
        plt.ylim(0,2)
        plt.title(f'Evolução de Loss e Acurácia - Fold {i}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        #DESCOMENTAR PARA PLOTAR
        
        
        plt.savefig(output_dir+f"Fold {i}.png")
        
        modelos.append(model.cpu().state_dict())
        fold_test_dataset.append(copy.deepcopy(test_dataset))
        fold_train.append(train_idx)
        del model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader
        torch.cuda.empty_cache()
        gc.collect()
    
    
    for i,(train_fold, test_fold) in enumerate(zip(fold_train,fold_test_dataset)):
        print(f"Fold {i}")
        print(f"Tamanho treino: {len(train_fold)}")
        print(f"Tamanho teste: {len(test_fold)}")


    # Configurações
    min_tailsize = 0
    max_tailsize = 60
    step_tail = 10
    alpha = 1
    epsilons = [0.33, 0.35, 0.37, 0.4]

    
    for epsilon in epsilons:
        
        results_by_tail = {}
        
        matrizes_confusao = [None for _ in range(min_tailsize, max_tailsize+1, step_tail)]
        #print(matrizes_confusao)
        for m in range(len(modelos)):
            print(f"Processando Fold {m+1}/{len(modelos)} para Epsilon {epsilon}...")
            
            
            torch.cuda.empty_cache()
            gc.collect()
            model = resnet50()
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(modelos[m]) 
            model.to(device)
            model.eval()

            
            test_loader = DataLoader(fold_test_dataset[m], batch_size=bs, shuffle=False)
            train_loader = DataLoader(Subset(panicum_kkc, fold_train[m]), batch_size=bs, shuffle=False)

            all_targets = np.array([])

            for (_, y) in test_loader:
                for target in y:
                    all_targets= np.append(all_targets,target.detach().cpu())

           
            for idx,tail in enumerate(range(min_tailsize, max_tailsize+1, step_tail)):
                
                
                if tail not in results_by_tail:
                    results_by_tail[tail] = {'f1': [], 'acc': [], 'uuc_acc': [], 'inner': [], 'outer': [], 'half': []}

                # Fit e Teste
                # Nota: O Fit ainda pode ser demorado, mas economizamos o load do modelo
                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                
                # Ajuste (Fit) - Geralmente precisa passar os dados de treino para calcular os centros/weibulls
                detector.fit(train_loader, device=device)
                
                metricas,predicts,targets_test = test(test_loader, detector)
                
                matriz=None
                if(matrizes_confusao[idx] is None):
                    matriz = mc(predicts,targets_test,all_targets,[],["Panicum","Solo","Milho"])
                    matriz.computa_matriz()
                    matrizes_confusao[idx] = matriz
                else:
                    matriz = matrizes_confusao[idx]
                    matriz.set_data(predicts,targets_test,all_targets)
                    matriz.computa_matriz()

                # Guardar métricas deste fold para este tail
                results_by_tail[tail]['f1'].append(metricas["F1 macro"])
                results_by_tail[tail]['acc'].append(metricas["accuracy"][0])
                results_by_tail[tail]['uuc_acc'].append(metricas["UUC Accuracy"][0])
                results_by_tail[tail]['inner'].append(metricas["inner metric"][0])
                results_by_tail[tail]['outer'].append(metricas["outer metric"][0])
                results_by_tail[tail]['half'].append(metricas["halfpoint"][0])

            del model, detector
            torch.cuda.empty_cache()

        
        
        final_data = []
        
        # Ordenar por tail para o CSV ficar bonito
        for tail in sorted(results_by_tail.keys()):
            metrics = results_by_tail[tail]
            
            row = {
                "tail": tail,
                "f1_macro_medio": np.mean(metrics['f1']),
                "f1_macro_std": np.std(metrics['f1']), # É bom ter o desvio padrão
                "acc_medio": np.mean(metrics['acc']),
                "acc_std": np.std(metrics['acc']),
                "uuc_acc_medio": np.mean(metrics['uuc_acc']),
                "uuc_acc_std": np.std(metrics['uuc_acc']),
                "inner_medio": np.mean(metrics['inner']),
                "inner_std": np.std(metrics['inner']),
                "outer_medio": np.mean(metrics['outer']),
                "outer_std": np.std(metrics['outer']),
                "halfpoint_medio": np.mean(metrics['half']),
                "halfpoint_std": np.std(metrics['half'])
            }
            final_data.append(row)

        df = pd.DataFrame(final_data)
        
        # Nome dinâmico correto
        pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
        filename_csv = f"Resultados_grid_panicum_tail_{min_tailsize}_{max_tailsize}_eps_{epsilon}_alpha_{alpha}.csv"
        organized_dir = os.path.join(output_dir, pasta)

        os.makedirs(organized_dir, exist_ok=True)

        csv_path = os.path.join(organized_dir,filename_csv)
        
        for idx,m in enumerate(matrizes_confusao):
            m.exibe_matriz(dir=organized_dir,name=f"tail_{idx*step_tail}")

        df.to_csv(csv_path, index=False)
        print(f"Arquivo salvo: {csv_path}")


    
if __name__ == '__main__':
    main()

