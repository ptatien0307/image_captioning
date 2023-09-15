from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import cv2
import os



IMAGEDIR = "images/"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-files")
async def create_upload_files(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    # save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    # image = cv2.imread(os.path.join('images/', file.filename))
    caption = 'CAPTIdasdasadON'

    show = file.filename
    return templates.TemplateResponse("index.html", {"request": request, "show": show, "caption": caption})