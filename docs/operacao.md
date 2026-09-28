---
title: Operations
kind: runbook
area: operations
project: bg-removal-service
collection: bg-removal-service
owner: maintainers
status: current
canonical: docs/operacao.md
globalRef: qmd://bg-removal-service/docs/operacao.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-28
sourceRefs:
  - README.md
  - config.py
  - systemd/bg-removal.service
related:
  - docs/arquitetura.md
  - docs/fluxos-negocio.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Operations

## Support status

This is a reference implementation and showcase project. The maintainers do
not provide a public hosted endpoint and do not claim production availability.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BG_REMOVAL_HOST` | `0.0.0.0` | Application bind address when the app reads config directly |
| `BG_REMOVAL_PORT` | `8002` | Application port |
| `SAM_CHECKPOINT` | `~/models/sam_vit_l_0b3195.pth` | SAM ViT-L checkpoint path |

Prefer an explicit loopback bind for development:

```bash
SAM_CHECKPOINT="$PWD/models/sam_vit_l_0b3195.pth" \
uvicorn api:app --host 127.0.0.1 --port 8002
```

## Deployment checklist

- Pin and scan dependencies for the target platform.
- Confirm CPU/GPU package compatibility and model licenses.
- Put the API behind TLS and authentication.
- Enforce body-size, rate, concurrency, and timeout limits at the edge.
- Restrict network access to known clients.
- Avoid logging uploaded image contents or sensitive metadata.
- Define model download provenance and checksum verification.
- Exercise health and both inference flows before routing traffic.

## systemd template

`systemd/bg-removal.service` is intentionally generic. Replace placeholders and
review its bind address, user, paths, resource limits, and hardening directives
before installation. Copying it does not make the service production-ready.

## Validation

```bash
curl --fail http://127.0.0.1:8002/health
```

Then run one automatic request and one refinement request with non-sensitive
fixtures. A healthy response alone does not validate output quality.

## Troubleshooting

- Startup failure: verify `SAM_CHECKPOINT`, file permissions, memory, and
  dependency compatibility.
- Invalid image: confirm the format is supported and the file is below the
  configured 20 MB limit.
- Slow refinement: SAM ViT-L is CPU-bound in the current implementation.
- Exposure risk: stop the process and correct the proxy/network boundary before
  resuming service.
