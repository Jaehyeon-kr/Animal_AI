import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

SRC_DIR  = "./animal_data/train"
DST_DIR  = "./animal_data/cropped"
CLASSES  = ["bear", "cat", "dog", "fox"]
CONF_TH  = 0.05      # 일단 매우 낮게
PADDING  = 0.15      # 박스 양옆으로 15% 여유

# 모델 로드 (최초 1회 다운로드)
print("모델 다운로드/로드 중...")
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt",
)
model = YOLO(model_path)
print(f"모델 로드 완료: {model_path}\n")


def expand_box(x1, y1, x2, y2, w, h, pad):
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(w, int(x2 + px)),
        min(h, int(y2 + py)),
    )


for cls in CLASSES:
    src = os.path.join(SRC_DIR, cls)
    dst = os.path.join(DST_DIR, cls)
    os.makedirs(dst, exist_ok=True)

    if not os.path.isdir(src):
        print(f"[{cls}] 폴더 없음")
        continue

    files = sorted(f for f in os.listdir(src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    total, hit = len(files), 0

    for fname in files:
        path = os.path.join(src, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        w, h = img.size

        results = model.predict(path, conf=CONF_TH, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        # 가장 confidence 높은 박스 1개만 사용
        confs = boxes.conf.cpu().numpy()
        idx = confs.argmax()
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, PADDING)

        crop = img.crop((x1, y1, x2, y2))
        crop.save(os.path.join(dst, fname))
        hit += 1

    rate = hit / total * 100 if total else 0
    print(f"[{cls}] 검출/전체 = {hit}/{total} ({rate:.1f}%)")

print("\n완료")
print(f"검출된 얼굴: {DST_DIR}/<class>/")
print("검출 안 된 이미지는 label.py로 수동 처리하면 돼")
