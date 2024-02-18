import cv2
import torch
from torchvision import transforms

from src.dataset import Vocabulary
from src.utils import load_model, plot_result, plot_attention
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    # Load model
    model = load_model("runs/models/transformer.pth")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(232, antialias=True),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

    # Predict caption
    with torch.no_grad():
        image = cv2.imread('images/667626_18933d713e.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_input = transform(image).unsqueeze(0).to(device)
        caption = model.generate_caption(image_input)

        plot_result(image, caption)
