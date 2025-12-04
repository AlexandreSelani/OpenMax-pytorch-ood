from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from pytorch_ood.utils import AnaliseGrafica,fix_random_seed
from pytorch_ood.utils.aux_dataset import validation_split
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import *
from sklearn.model_selection import train_test_split,StratifiedKFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
seed=777
fix_random_seed(seed)

device = "cuda:0"


def test(test_loader, model):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted')


    print(f"Test accuracy: {acc:.4f} | F1-score: {f1:.4f}")

    return acc, f1,precision,recall


def train(train_loader,model,criterion,optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs)
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


weights = ResNet50_Weights.IMAGENET1K_V2
transforms = weights.transforms()

imagens = ImageFolder('/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Todas',transform=transforms)

bs = 8
lr = 0.0001
epochs=100


results = pd.DataFrame(columns=["fold","accuracy","F1 score","precision","recall"])

kfold = StratifiedKFold(n_splits=5)
for i, (train_index, test_index) in enumerate(kfold.split(imagens.imgs, imagens.targets)):
    # coloque logo após obter train_index, test_index no seu loop KFold
    train_index = train_index.tolist() if hasattr(train_index, "tolist") else list(train_index)
    test_index  = test_index.tolist()  if hasattr(test_index, "tolist")  else list(test_index)

    overlap = set(train_index).intersection(set(test_index))
    print("Tamanho treino:", len(train_index), "tamanho teste:", len(test_index), "overlap:", len(overlap))
    if len(overlap) > 0:
        print("Exemplo de índices overlapped:", list(overlap)[:10])

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.fc = nn.Linear(model.fc.in_features, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(f"Fold {i}:")
    train_imgs = Subset(imagens,train_index)
    train_imgs,val_imgs = validation_split(0.2,train_imgs)

    test_imgs = Subset(imagens,test_index)
    
    print("tamanho treino ",len(train_imgs),"\ntamanho validacao ",len(val_imgs),"\ntamanho teste:",len(test_imgs))

    train_loader = DataLoader(train_imgs,batch_size=bs,shuffle=True)
    test_loader = DataLoader(test_imgs,batch_size=bs,shuffle=False)
    val_loader = DataLoader(val_imgs,batch_size=bs,shuffle=False)

    t_loss,t_acc=[],[]
    v_loss,v_acc=[],[]

    for epoch in range(epochs):
        train_loss,train_acc = train(train_loader,model,criterion,optimizer)
        val_loss,val_acc = validation(val_loader,model,criterion)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
        print(f"VALIDATION| Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

        t_loss.append(train_loss)
        t_acc.append(train_acc)
        v_loss.append(val_loss)
        v_acc.append(val_acc)

    torch.cuda.empty_cache()

    acc,f1,precision,recall = test(test_loader,model)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")   

    results.loc[len(results)] = [i,acc,f1,precision,recall]

    plt.figure(figsize=(8,5))
    e = range(0,epochs)
    plt.plot(e, t_loss, 'red', label='Train Loss')
    plt.plot(e, v_loss, 'orange', label='Val Loss')
    plt.plot(e, t_acc, 'blue', label='Train Acc')
    plt.plot(e, v_acc, 'purple', label='Val Acc')

    plt.xlabel('Épocas')
    plt.xticks(e)
    plt.ylabel('Valor')
    plt.title('Evolução de Loss e Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"Fold {i}")

results.loc[len(results)] = ["MEDIA",results["accuracy"].mean(),results["F1 score"].mean(),results["precision"].mean(),results["recall"].mean()]

results.to_csv("Resultados.csv",index=False)


