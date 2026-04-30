import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from face_utils import detect_and_crop_face

MODEL_PATH = "./model/animal_predict_model.pth"
CLASSES    = ["bear", "cat", "dog", "fox"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()


def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    face, detected = detect_and_crop_face(image)
    if detected:
        print(f"얼굴 검출 ✓  크롭 사이즈: {face.size}")
    else:
        print("얼굴 검출 실패 — 원본 이미지로 추론")

    x = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    pred = probs.argmax().item()
    print(f"예측: {CLASSES[pred]}")
    print("확률:")
    for cls, p in zip(CLASSES, probs):
        print(f"  {cls}: {p.item()*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python inference.py <이미지 경로>")
        sys.exit(1)
    predict(sys.argv[1])
