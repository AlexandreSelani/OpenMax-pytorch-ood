from PIL import Image
from torchvision.datasets import VisionDataset
import numpy as np
from .CarregaImagens import CarregaImagens

class PanicumDataset_pytorchOOD(CarregaImagens):
    """
    Classe usada para carregar o dataset panicum ja segmentado, PARA USO NO PYTORCH-OOD. As imagens de treino e teste vao estar em pastas diferentes
    """
    def __init__(self, root,transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        self.rotulos=self.rotulos-1