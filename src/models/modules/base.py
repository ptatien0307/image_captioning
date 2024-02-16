import torch
from torch.nn import Module, Sequential
from torch.nn import Module, Linear
from torchvision.models import resnet50, ResNet50_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionEncoder(Module):
    def __init__(self):
        super(AttentionEncoder, self).__init__()

        # Load pretrained model and remove last fc layer
        pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = Sequential(*list(pretrained_model.children())[:-2]).to(device)

        # Freeze layer
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.to(device)

        features = self.model(images)                                       # (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)                             # (batch_size, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))   # (batch_size, 49, 2048)
        return features

class NormalEncoder(Module):
    def __init__(self, encoder_dim):
        super(NormalEncoder, self).__init__()
        self.encoder_dim = encoder_dim

        # Load pretrained model and remove last fc layer
        pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(pretrained_model.children())[:-1]).to(device)

        # Freeze layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a linear layer add the end of model
        self.linear = torch.nn.Linear(2048, self.encoder_dim).to(device)
        self.drop = torch.nn.Dropout(0.3)


    def forward(self, images):
        images = images.to(device)
        
        # Forward pass
        features = self.model(images)                     # (batch_size, 2048, 1, 1)
        features = features.view(images.shape[0], 1, -1)  # (batch_size, 1, 2048)
        features = self.linear(self.drop(features))       # (batch_size, 1, 512)
        features = features.squeeze(1)                    # (batch_size, 512)
        return features

class TransformerEncoder(Module):
    def __init__(self, encoder_dim, d_model):
        super(TransformerEncoder, self).__init__()

        # Load pretrained model and remove last fc layer
        pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = Sequential(*list(pretrained_model.children())[:-2]).to(device)

        # Freeze layer
        for param in self.model.parameters():
            param.requires_grad = False

        self.linear = Linear(encoder_dim, d_model).to(device)
    def forward(self, images):
        images = images.to(device)

        features = self.model(images)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)
        features = self.linear(features)
        return features # (batch_size, 49, d_model)