---
title: Architecture
kind: architecture
area: engineering
project: bg-removal-service
collection: bg-removal-service
owner: maintainers
status: current
canonical: docs/arquitetura.md
globalRef: qmd://bg-removal-service/docs/arquitetura.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-28
sourceRefs:
  - api.py
  - config.py
  - models.py
related:
  - docs/fluxos-negocio.md
  - docs/operacao.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Architecture

## Components

- `api.py`: FastAPI lifecycle, validation, and HTTP endpoints.
- `models.py`: IS-Net and SAM loading, inference, mask editing, and image
  composition.
- `config.py`: environment-driven server and checkpoint settings plus fixed
  limits and output options.
- `requirements.txt`: Python dependencies.
- `systemd/bg-removal.service`: generic self-hosting template.

## Runtime flow

At startup, the application loads the `rembg` session and the SAM ViT-L
checkpoint. Both global model handles are then reused by requests.

For automatic removal, `rembg` returns an RGBA image. For refinement, the alpha
channel becomes the base mask and SAM adjusts local regions selected by the
client. The final RGBA image is composited over a solid RGB background and
encoded as JPEG.

## State and dependencies

The service itself is stateless: it does not persist uploads or results. Model
weights are local runtime dependencies. Startup fails if the SAM checkpoint is
missing or incompatible.

The inference code explicitly requests CPU execution. The current dependency
manifest still contains accelerator-oriented packages inherited from the
prototype; packaging should be validated for the target platform before a
production deployment.

## Trust boundary

FastAPI is the application boundary, not a complete public edge. Authentication,
TLS, rate limiting, concurrency protection, request timeouts, observability,
and abuse controls belong to the deployment layer unless implemented in a
future version.
