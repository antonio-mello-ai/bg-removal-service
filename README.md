---
title: Background Removal Service
kind: source_doc
area: engineering
project: bg-removal-service
collection: bg-removal-service
owner: maintainers
status: current
canonical: README.md
globalRef: qmd://bg-removal-service/README.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-28
sourceRefs: []
related:
  - docs/index.md
  - docs/fluxos-negocio.md
  - docs/arquitetura.md
  - docs/operacao.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Background Removal Service

A small FastAPI reference service for removing image backgrounds automatically
and refining the result with point prompts.

This repository is maintained as an open-source showcase. It does not represent
an active hosted service operated by the maintainers.

## How it works

| Model | Role | Execution selected by the code |
|---|---|---|
| IS-Net via `rembg` | Automatic foreground mask | CPU |
| SAM ViT-L | Point-guided mask refinement | CPU |

The API supports two flows:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Reports whether both models are loaded |
| `/remove-background` | POST | Removes the background automatically |
| `/remove-background/refine` | POST | Adjusts the automatic mask with user points |

Both image endpoints currently return JPEG with a solid white or gray
background.

## Quick start

Prerequisites:

- Python 3.10+
- enough memory for IS-Net and SAM ViT-L
- a locally downloaded SAM ViT-L checkpoint

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p models
wget -P models/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

SAM_CHECKPOINT="$PWD/models/sam_vit_l_0b3195.pth" \
uvicorn api:app --host 127.0.0.1 --port 8002
```

Verify:

```bash
curl http://127.0.0.1:8002/health
```

Automatic removal:

```bash
curl -X POST http://127.0.0.1:8002/remove-background \
  -F "image=@photo.jpg" \
  -F "background=white" \
  -o result.jpg
```

Point-guided refinement:

```bash
curl -X POST http://127.0.0.1:8002/remove-background/refine \
  -F "image=@photo.jpg" \
  -F 'points=[{"x":300,"y":250,"label":0}]' \
  -F "background=white" \
  -o result-refined.jpg
```

- `label=0`: remove the selected region from the foreground mask.
- `label=1`: add the selected region to the foreground mask.

## Security boundary

The application has no built-in authentication, rate limiting, or TLS. Keep it
bound to loopback for local use. Before exposing it to a network, place it
behind an authenticated reverse proxy and define request-size, concurrency,
timeout, logging, and retention controls. See [operations](docs/operacao.md).

## Documentation

Start at [docs/index.md](docs/index.md).

## License

MIT
