FROM python:3.10-slim

WORKDIR /app

# 1) CPU-only torch 먼저 설치 (CUDA wheel은 ~2GB라 피해야 함)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) 나머지 의존성은 requirements.txt에서 (이미 설치된 torch는 스킵됨)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) 앱 코드 + 모델
COPY api.py .
COPY model/animal_predict_model.pth ./model/animal_predict_model.pth

EXPOSE 8000

# Railway는 $PORT를 동적으로 주입하므로 sh -c로 받아서 사용
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
