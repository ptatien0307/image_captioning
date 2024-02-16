import torch
from torch.nn import Module, Sequential
from torchvision.models import resnet50, ResNet50_Weights

from abc import ABC, abstractmethod
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseEncoder(Module, ABC):
    def __init__(self):
        super(BaseEncoder, self).__init__()

        # Load pretrained model and remove last fc layer
        pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = Sequential(*list(pretrained_model.children())[:-2]).to(device)

        # Freeze layer
        for param in self.model.parameters():
            param.requires_grad = False

    @abstractmethod
    def forward(self, images):
        pass