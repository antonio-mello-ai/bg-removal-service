# CLAUDE.md - bg-removal-service

Servico de remocao de fundo de imagens com IA. Deploy na VM 151 (llm-gpu-linux, RTX 5090).

## Stack

- **FastAPI** (porta 8002)
- **rembg** + IS-Net (remocao automatica, ONNX Runtime CPU)
- **SAM ViT-L** (refinamento interativo, PyTorch CPU)
- **Venv separado**: `/home/amello/bg-removal-env`

## Arquitetura

Dois modos de operacao:
1. **Automatico** (`/remove-background`): IS-Net remove fundo sem interacao
2. **Refinamento** (`/remove-background/refine`): parte da mascara IS-Net como base, usa SAM para segmentar a regiao local de cada clique do usuario, e subtrai (label=0) ou adiciona (label=1) da mascara. Sempre seleciona a menor mascara SAM para operacoes cirurgicas.

## Infraestrutura

- **VM 151**: GPU RTX 5090 (32GB VRAM), vLLM usa 92% VRAM. BG Removal roda 100% CPU para nao competir
- **Consumidores**: ERP Desmanches (VM 210), qualquer servico via HTTP
- **Systemd**: `bg-removal.service`

## Setup obrigatório após clone

Executar uma vez: `git config core.hooksPath .githooks`

Isso ativa o pre-push hook que exige bump do arquivo `VERSION` quando há mudanças de código.

## Regras

- Output sempre JPEG (quality 95) com fundo solido (white ou gray)
- Modelos carregados no startup (nao lazy load)
- Ambos modelos rodam em CPU (zero VRAM)
- Manter compatibilidade com API do ERP (nao quebrar contratos)
- SAM sempre usa a menor mascara (argmin area) — tanto remocao quanto adicao sao operacoes de detalhe
