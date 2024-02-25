


import cv2
import torch
from torchvision import transforms

import uvicorn
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, UploadFile, File
from src.utils import load_model
from src.dataset import Vocabulary


app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory='templates')
model = load_model("runs/models/transformer-v2.pth")
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(232, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/upload-file")
async def create_upload_file(request: Request, files: List[UploadFile]=File(...)):
    for file in files:
        contents = await file.read()
        # save the file
        with open("images/uploaded_image.jpg", "wb") as f:
            f.write(contents)


    # Predict caption
    with torch.no_grad():
        image = cv2.imread('images/uploaded_image.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_input = transform(image).unsqueeze(0)
        caption = model.generate_caption(image_input)
        caption = caption[:-5]

    return templates.TemplateResponse("index.html", {"request": request, 
                                                     "image": 'uploaded_image.jpg',
                                                     'caption': caption})

if __name__ == '__main__':


    uvicorn.run(app, host="127.0.0.1", port=8000)