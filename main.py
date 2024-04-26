


import cv2
import torch
from torchvision import transforms

import uvicorn
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, UploadFile, File
from src.utils import load_model, load_vit_gpt
from src.dataset import Vocabulary


app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory='templates')
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
    # Read uploaded image
    for file in files:
        contents = await file.read()
        with open("images/uploaded_image.jpg", "wb") as f:
            f.write(contents)

    # Select model
    form_data = await request.form()
    method = form_data['method']

    if method == '1':
        model = load_model("runs/models/transformer.pth").to(device)
        model.eval()
    if method == '2':
        model = load_model("runs/models/par_inject.pth").to(device)
        model.eval()
    if method == '3':
        model = load_model("runs/models/init_inject.pth").to(device)
        model.eval()
    if method == '4':
        model = load_model("runs/models/bahdanau.pth").to(device)
        model.eval()
    if method == '5':
        model = load_model("runs/models/luong.pth").to(device)
        model.eval()
    if method == '6':
        model, tokenizer, image_processor = load_vit_gpt("runs/models/vit-gpt")
        model = model.to(device)

    # Predict caption
    if method == '6':
        image = cv2.imread('images/uploaded_image.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixel_values = image_processor(image, return_tensors="pt").pixel_values # Preprocess
        pixel_values = pixel_values.to(device)

        # Autoregressively generate caption (uses greedy decoding by default)
        generated_ids = model.generate(pixel_values)
        caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        with torch.no_grad():
            image = cv2.imread('images/uploaded_image.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_input = transform(image).unsqueeze(0).to(device)
            caption = model.generate_caption(image_input)

    
    return templates.TemplateResponse("index.html", {"request": request, 
                                                     "image": 'uploaded_image.jpg',
                                                     'caption': caption})

if __name__ == '__main__':


    uvicorn.run(app, host='0.0.0.0', port=8080)