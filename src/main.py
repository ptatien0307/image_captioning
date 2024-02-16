import cv2
import torch
from torchvision import transforms

from dataset import Vocabulary
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from utils import load_model, plot_result, plot_attention



# Load model
model = load_model("runs/models/bahdanau.pth")
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
    image_input = transform(image).unsqueeze(0)
    caption, context = model.generate_caption(image_input)

    plot_result(image, caption)
