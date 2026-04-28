import io
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

MODEL_PATH = "./animal_predict_model.pth"
CLASSES    = ["bear", "cat", "dog", "fox"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

app = FastAPI()


class PredictReq(BaseModel):
    image_url: str


@app.post("/predict")
async def predict(req: PredictReq):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(req.image_url)
            resp.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 URL을 가져올 수 없습니다.")

    try:
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 열 수 없습니다.")

    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]

    return {
        "similarities": {cls.upper(): round(p.item() * 100, 1) for cls, p in zip(CLASSES, probs)},
    }


@app.get("/health")
def health():
    return {"status": "ok"}
