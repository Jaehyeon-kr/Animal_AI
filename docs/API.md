# Animal Face API

동물상 분류 REST API입니다. 이미지를 업로드하면 bear / cat / dog / fox 중 하나를 예측합니다.

---

## Base URL

```
http://<서버주소>:8000
```

---

## Endpoints

### `POST /predict`

이미지 URL을 전달하여 동물상을 예측합니다.

**Request**

| 항목 | 내용 |
|------|------|
| Method | POST |
| Content-Type | application/json |
| Body | `image_url` — 이미지 URL (string) |

**Example**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```

**Response `200 OK`**

```json
{
  "similarities": {
    "CAT": 88.0,
    "DOG": 7.0,
    "FOX": 3.0,
    "BEAR": 2.0
  }
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `similarities` | object | 각 클래스별 유사도 (0~100) |

**Response `400 Bad Request`**

```json
{
  "detail": "이미지 URL을 가져올 수 없습니다."
}
```

---

### `GET /health`

서버 상태를 확인합니다.

**Example**

```bash
curl http://localhost:8000/health
```

**Response `200 OK`**

```json
{
  "status": "ok"
}
```
