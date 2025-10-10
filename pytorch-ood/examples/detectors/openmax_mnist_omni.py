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
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import PlainCNN
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, metricasImplementadas,AnaliseGrafica_OpenMax
from pytorch_ood.utils.aux_dataset import *

seed = 777
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
    ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadas(predict=predicts, label=labels)

    print(ood_metrics.compute())
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
# %%

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
    nomeDataset = "Mnist + omniglot"

    analiseGrafica = AnaliseGrafica_OpenMax(nomeDataset)

    lr=0.05
    epochs = 5
    bs=128

    # Setup preprocessing and data
    MNIST_trans = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
    ])


    mnist = MNIST(root="data", train=True, download=True, transform=MNIST_trans)

    dataset_train,dataset_val = validation_split(0.1,mnist,seed)

    dataset_in_test = MNIST(root="data", train=False, download=True, transform=MNIST_trans)

    dataset_out_test = Omniglot(
         root="data", download=True,background=False, transform=MNIST_trans, target_transform=ToUnknown()
     )
    
    novo_tamanho=10000
    novo_dataset_out_test = random_dataset(dataset_out_test,novo_tamanho)

    print(f"tamanho treino mnist {len(dataset_train)}\n tamanho teste mnist {len(dataset_in_test)}\n tamanho teste omniglot {len(novo_dataset_out_test)}")
    train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset_val,batch_size=bs,shuffle=False)


    test_loader = DataLoader(dataset_in_test+novo_dataset_out_test, batch_size=bs,shuffle=False)

    

    tailsize=20
    alpha=4
    epsilon=0.5
    model = PlainCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    detector = OpenMax(model, tailsize=tailsize, alpha=alpha, euclid_weight=1,epsilon=epsilon)

    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validation(val_loader,model,criterion)

        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
        
        
           
        detector.fit(train_loader, device=device)
        metricas = test(test_loader,detector)
        scheduler.step()
        print(metricas)
        analiseGrafica.addEpoch(metricas,epoch,train_loss=train_loss,train_acc=train_acc,val_loss=val_loss,val_acc=val_acc)


    analiseGrafica.mostraGrafico(tail=tailsize,alpha=alpha,epsilon=epsilon)
        

    
if __name__ == '__main__':
    main()

