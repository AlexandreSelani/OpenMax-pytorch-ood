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
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST,Omniglot
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.dataset.img import MNIST_OSR_TRAIN
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import PlainCNN
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, metricasImplementadas,AnaliseGrafica_OpenMax,Matriz_confusao_osr as mc
from pytorch_ood.utils.aux_dataset import *

seed=777
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

    
def main():
    UUC_classes = [7,8,9]

    nomeDataset = f"Mnist (UUC:{UUC_classes}+ omniglot"

    analiseGrafica = AnaliseGrafica_OpenMax(nomeDataset)
    
    MNIST_trans = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
    ])
    
    bs = 128

    #conjunto de treino mnist sem as classes desconhecidas
    mnist_train = MNIST_OSR_TRAIN(root="data", train=True, download=True, transform=MNIST_trans,UUC_classes=UUC_classes)

    #conjuntos de treino e de validacao apos split
    dataset_train,dataset_val = validation_split(0.05,mnist_train,seed)

    #conjunto de teste mnist (target de classes desconhecidas = -1)
    dataset_mnist_test = MNIST_OSR_TRAIN(root="data", train=False, download=True, transform=MNIST_trans,UUC_classes=UUC_classes)

    #targets originais do conjunto de teste para matriz de confusao
    targets_antigos = dataset_mnist_test.get_targets_antigos()  

    dataset_out_test = Omniglot(
         root="data", download=True,background=False, transform=MNIST_trans, target_transform=ToUnknown()
     )
    
    targets_omniglot = torch.tensor([t for _, t in dataset_out_test])
    targets_antigos = torch.cat((targets_antigos, targets_omniglot))

    print(targets_antigos)
    novo_tamanho=10000
    #ajuste de tamanho do dataset de outliers
    novo_dataset_out_test = random_dataset(dataset_out_test,novo_tamanho)

    print(f"tamanho treino mnist {len(dataset_train)}\n tamanho teste mnist {len(dataset_mnist_test)}\n tamanho teste omniglot {len(novo_dataset_out_test)}")

    train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)

    val_loader = DataLoader(dataset_val, batch_size=bs, shuffle=False)

    #IMPORTANTE PARA A MATRIZ DE CONFUSAO: SHUFFLE DEVE SER FALSE PARA QUE NAO HAJA DIFERENCA NA ORDEM DAS LABELS ORIGINAIS E DAS LABELS APOS A DEFINICAO DAS CLASSES DESCONHECIDAS
    test_loader = DataLoader(dataset_mnist_test+dataset_out_test, batch_size=bs,shuffle=False)

    #obtencao dos targets do conjunto de testes apos definicao das classes desconhecidas
    test_dataloader_targets=[]

    for i,(_,y) in enumerate(test_loader):
        test_dataloader_targets.append(y)

    test_dataloader_targets = torch.cat(test_dataloader_targets) 
    print(test_dataloader_targets)
    #definicao de parametros de treino
    lr=0.0003
    epochs = 500
    criterion = nn.CrossEntropyLoss()

    #parametros do openmax
    tailsize=20
    alpha=4
    epsilon=0.5

    #criacao do modelo de rede neural
    model = PlainCNN(num_classes=10-len(UUC_classes)).to(device)
    detector = OpenMax(model, tailsize=tailsize, alpha=alpha, euclid_weight=1,epsilon=epsilon)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    detector = OpenMax(model, tailsize=tailsize, alpha=alpha, euclid_weight=1,epsilon=epsilon)

    #loop treino/validacao/teste
    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validation(val_loader,model,criterion)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}") 
           
        detector.fit(train_loader, device=device)#ajuste do openmax
        metricas = test(test_loader,detector)
        scheduler.step()
        print(metricas)

        analiseGrafica.addEpoch(metricas,epoch,train_loss=train_loss,train_acc=train_acc,val_loss=val_loss,val_acc=val_acc)

    nome_classes_originais = ["omniglot",0,1,2,3,4,5,6]
    confusion_matrix(test_loader,targets_antigos,nome_classes_originais,UUC_classes,detector)

    analiseGrafica.mostraGrafico(tail=tailsize,alpha=alpha,epsilon=epsilon)
        

    
if __name__ == '__main__':
    main()

