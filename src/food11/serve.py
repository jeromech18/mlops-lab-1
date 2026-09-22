import io
import os

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from food11.data import CATEGORIES

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = FastAPI()
model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model("models:/food11@champion")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).numpy()  # shape (1, 3, 224, 224)

    logits = model.predict(input_tensor)
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())

    return {
        "category": CATEGORIES[predicted_idx],
        "confidence": confidence,
    }