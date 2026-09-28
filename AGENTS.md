---
title: AGENTS.md — Background Removal Service
kind: policy
area: engineering
project: bg-removal-service
collection: bg-removal-service
owner: maintainers
status: current
canonical: AGENTS.md
globalRef: qmd://bg-removal-service/AGENTS.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-28
sourceRefs: []
related:
  - README.md
  - docs/fluxos-negocio.md
  - docs/arquitetura.md
  - docs/operacao.md
  - docs/index.md
supersedes:
  - CLAUDE.md
  - GEMINI.md
supersededBy: []
sensitivity: public
---
# AGENTS.md — Background Removal Service

## Purpose

This repository is a public reference implementation for automatic and
point-guided image background removal. It is not the source of truth for any
private deployment, customer environment, internal hostname, or infrastructure
topology.

## Repository rules

- Keep examples generic and safe for a public repository.
- Never add private hostnames, IP addresses, usernames, filesystem paths,
  credentials, customer names, or deployment-specific secrets.
- Preserve the two public API flows documented in `docs/fluxos-negocio.md`.
- Treat the bundled systemd unit as a template. It must not encode a real user,
  host, or environment.
- Do not claim that this repository is currently deployed or operated by its
  maintainers.
- Roadmap and delivery status live in GitHub Issues and pull requests. Do not
  create `roadmap.md`, `docs/backlog.md`, or `CHANGELOG.md`.

## Local setup

Run once after cloning:

```bash
git config core.hooksPath .githooks
```

The pre-push hook requires a `VERSION` bump for code changes. Documentation-only
changes do not require a version bump.

## Verification

- Run `git diff --check` for every change.
- For code changes, add or update tests and exercise the affected endpoint.
- Keep `VERSION`, the FastAPI version metadata, and release notes in the GitHub
  release aligned when publishing a release.

## Active documentation

- `README.md`: public entry point and quick start
- `docs/fluxos-negocio.md`: supported user flows
- `docs/arquitetura.md`: components and data flow
- `docs/operacao.md`: self-hosting and operational boundaries
- `docs/index.md`: documentation index
