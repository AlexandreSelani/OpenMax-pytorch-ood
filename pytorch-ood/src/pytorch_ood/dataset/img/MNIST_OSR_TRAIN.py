from torchvision.datasets import MNIST
from typing import Union, Callable, Optional
from pathlib import Path
import torch

class MNIST_OSR_TRAIN(MNIST):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        UUC_classes: tuple[int] = None  
    ) -> None:
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

      
        self.classes_removidas = torch.tensor(UUC_classes)

        if train:
            
            mask = ~torch.isin(self.targets, self.classes_removidas)#True se um elemento de targets NAO esta em classes removidas e False caso contrario
            self.targets = self.targets[mask]
            self.data = self.data[mask]
            

        else:
            print(f"self.targets = {self.targets[:50]}")
            self.targets = torch.tensor([
                x.item() if x not in UUC_classes else -1
                for x in self.targets
            ])
            print(f"self.targets NOVO = {self.targets[:50]}")
