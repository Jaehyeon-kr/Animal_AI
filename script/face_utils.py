from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

print("YOLOv8-Face 모델 로딩 중...")
_MODEL_PATH = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt",
)
_MODEL = YOLO(_MODEL_PATH)
print("YOLOv8-Face 모델 로드 완료")


def detect_and_crop_face(image: Image.Image, conf: float = 0.25, padding: float = 0.15):
    """
    얼굴 검출 후 가장 큰 얼굴을 크롭해서 리턴.
    검출 실패 시 원본 이미지를 그대로 리턴.

    Returns:
        (PIL.Image, bool): (크롭된 또는 원본 이미지, 검출 성공 여부)
    """
    w, h = image.size
    results = _MODEL.predict(image, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image, False

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = areas.argmax()
    x1, y1, x2, y2 = xyxy[idx]

    bw, bh = x2 - x1, y2 - y1
    px, py = bw * padding, bh * padding
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px))
    y2 = min(h, int(y2 + py))

    return image.crop((x1, y1, x2, y2)), True
