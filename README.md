# bg-removal-service

Background removal service for product images. Runs both models on CPU — zero GPU impact if you're running other GPU workloads on the same machine.

## Models

| Model | Function | Device | RAM |
|-------|----------|--------|-----|
| IS-Net (rembg) | Automatic background removal | CPU | ~200MB |
| SAM ViT-L | Interactive refinement with point prompts | CPU | ~1.5GB |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status |
| `/remove-background` | POST | Automatic removal (IS-Net) |
| `/remove-background/refine` | POST | Refinement with user points (SAM) |

### POST /remove-background

```bash
curl -X POST http://localhost:8002/remove-background \
  -F "image=@photo.jpg" \
  -F "background=white" \
  -o result.jpg
```

### POST /remove-background/refine

The refinement endpoint starts from the IS-Net mask as a base. For each user point, SAM segments the local region (always the smallest mask) and subtracts or adds from the base mask.

```bash
curl -X POST http://localhost:8002/remove-background/refine \
  -F "image=@photo.jpg" \
  -F 'points=[{"x": 300, "y": 250, "label": 0}]' \
  -F "background=white" \
  -o result_refined.jpg
```

Points:
- `label=0` — "this is background" (remove from mask)
- `label=1` — "this is object" (add to mask)

### Available backgrounds

- `white` (255, 255, 255) — marketplace standard
- `gray` (224, 224, 224) — internal catalog

## Setup

```bash
# 1. Create venv
python3 -m venv ~/bg-removal-env
source ~/bg-removal-env/bin/activate

# 2. Install PyTorch (adjust for your CUDA version or use CPU)
pip install torch torchvision

# 3. Install dependencies
## Response Format

### `/remove-background`
Returns the processed image as a binary file.

| Field | Type | Description |
|-------|------|-------------|
| Response Body | `binary` | Processed image file |
| Content-Type | `image/jpeg` | Output format |

### `/health`
```json
{
  "status": "ok",
  "models": {
    "isnet": "loaded",
    "sam": "loaded"
  }
}
```

## Error Codes

| Status Code | Meaning |
|-------------|---------|
| `200` | Success |
| `400` | Bad request — missing or invalid image/points |
| `422` | Unprocessable entity — invalid input format |
| `500` | Internal server error — model failure |

## Python Client Examples

```python
# Automatic background removal
import requests

with open("photo.jpg", "rb") as image_file:
    response = requests.post(
        "http://localhost:8002/remove-background",
        files={"image": image_file},
        data={"background": "white"}
    )

with open("result.jpg", "wb") as output:
    output.write(response.content)

print("Status:", response.status_code)
```

```python
# Refinement with point prompts
import requests
import json

points = [
    {"x": 300, "y": 250, "label": 0},  # background
    {"x": 150, "y": 100, "label": 1}   # object
]

with open("photo.jpg", "rb") as image_file:
    response = requests.post(
        "http://localhost:8002/remove-background/refine",
        files={"image": image_file},
        data={
            "points": json.dumps(points),
            "background": "white"
        }
    )

with open("result_refined.jpg", "wb") as output:
    output.write(response.content)
```
