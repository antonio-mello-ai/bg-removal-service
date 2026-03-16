# bg-removal-service

Servico de remocao de fundo de imagens para produtos/pecas automotivas. Roda na **VM 151** (llm-gpu-linux) com RTX 5090.

## Modelos

| Modelo | Funcao | VRAM |
|--------|--------|------|
| IS-Net (rembg) | Remocao automatica de fundo | ~100MB |
| SAM ViT-L | Refinamento interativo com point prompts | ~1.2GB |

## Endpoints

| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/health` | GET | Status dos modelos |
| `/remove-background` | POST | Remocao automatica (IS-Net) |
| `/remove-background/refine` | POST | Refinamento com pontos do usuario (SAM) |

### POST /remove-background

```bash
curl -X POST http://llm-gpu-linux:8002/remove-background \
  -F "image=@foto_peca.jpg" \
  -F "background=white" \
  -o resultado.jpg
```

### POST /remove-background/refine

```bash
curl -X POST http://llm-gpu-linux:8002/remove-background/refine \
  -F "image=@foto_peca.jpg" \
  -F 'points=[{"x": 300, "y": 250, "label": 0}]' \
  -F "background=white" \
  -o resultado_refinado.jpg
```

Points:
- `label=0` — "isto e fundo" (remover)
- `label=1` — "isto e objeto" (manter)

### Backgrounds disponiveis

- `white` (255, 255, 255) — padrao para marketplaces
- `gray` (224, 224, 224) — catalogo interno

## Setup (VM 151)

```bash
# 1. Criar venv
python3 -m venv /home/amello/bg-removal-env
source /home/amello/bg-removal-env/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Baixar SAM ViT-L checkpoint
mkdir -p /home/amello/models
wget -P /home/amello/models/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# 4. Copiar codigo
cp -r . /home/amello/bg-removal-service/

# 5. Instalar systemd service
sudo cp systemd/bg-removal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bg-removal

# 6. Verificar
curl http://localhost:8002/health
```

## Ajuste necessario no vLLM

Reduzir `--gpu-memory-utilization` de 0.90 para 0.85 em `/etc/systemd/system/vllm.service` para liberar VRAM para os modelos de segmentacao.

```bash
sudo sed -i 's/--gpu-memory-utilization 0.90/--gpu-memory-utilization 0.85/' /etc/systemd/system/vllm.service
sudo systemctl daemon-reload
sudo systemctl restart vllm
```

## Acesso

- Local: `http://192.168.68.193:8002`
- Tailscale: `http://llm-gpu-linux.tail0ccdbc.ts.net:8002`

## Logs

```bash
sudo journalctl -u bg-removal -f
```
