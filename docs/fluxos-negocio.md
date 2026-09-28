---
title: Supported flows
kind: source_doc
area: product
project: bg-removal-service
collection: bg-removal-service
owner: maintainers
status: current
canonical: docs/fluxos-negocio.md
globalRef: qmd://bg-removal-service/docs/fluxos-negocio.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-28
sourceRefs:
  - api.py
  - models.py
related:
  - README.md
  - docs/arquitetura.md
  - docs/operacao.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Supported flows

## Automatic background removal

1. A client uploads a JPEG, PNG, or WebP image.
2. The API validates the upload size and decodes the image.
3. IS-Net produces an automatic foreground mask.
4. The service composites the foreground over the selected solid background.
5. The API returns a JPEG response.

The supported backgrounds are `white` and `gray`.

## Point-guided refinement

1. A client uploads the original image and a non-empty JSON list of points.
2. The service first generates the automatic IS-Net mask.
3. SAM receives each point and returns candidate local regions.
4. The smallest candidate region is used for a targeted edit.
5. A point with `label=0` subtracts the region; `label=1` adds it.
6. The service composites and returns the refined JPEG.

Coordinates are expressed in pixels in the uploaded image coordinate system.
The current request limit is 50 points.

## Health check

`GET /health` reports whether the IS-Net and SAM model handles were loaded. It
does not prove the quality of an inference or the readiness of an external
proxy, queue, storage layer, or deployment.

## Out of scope

- a hosted API operated by the maintainers
- user accounts, authentication, billing, or persistent storage
- asynchronous job processing
- transparent PNG/WebP output
- batch processing

Potential extensions are tracked as GitHub Issues rather than in a parallel
roadmap document.
