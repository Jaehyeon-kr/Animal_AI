FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi uvicorn[standard] pillow httpx

COPY animal_predict_model.pth .
COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
