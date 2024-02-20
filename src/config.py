from dataclasses import dataclass
from torchvision import transforms

@dataclass
class InitInjectConfig():
    csv_file = "dataset/captions.txt"
    max_length = 50
    freq_threshold = 5

    batch_size = 50
    num_epochs = 50
    embedding_dim = 300
    encoder_dim = 512
    decoder_dim = 512
    num_layers = 4

    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(232, antialias=True),
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
