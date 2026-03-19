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
pip install -r requirements.txt

# 4. Download SAM ViT-L checkpoint
mkdir -p ~/models
wget -P ~/models/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# 5. Install systemd service (optional)
sudo cp systemd/bg-removal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bg-removal

# 6. Verify
curl http://localhost:8002/health
```

## Systemd service

Edit `systemd/bg-removal.service` to match your paths and username before installing.

## Logs

```bash
sudo journalctl -u bg-removal -f
```

## License

MIT
