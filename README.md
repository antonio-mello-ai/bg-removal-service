# bg-removal-service

Servico de remocao de fundo de imagens para produtos/pecas automotivas. Roda na **VM 151** (llm-gpu-linux). Ambos modelos rodam em CPU — zero impacto no vLLM.

## Modelos

| Modelo | Funcao | Device | RAM |
|--------|--------|--------|-----|
| IS-Net (rembg) | Remocao automatica de fundo | CPU | ~200MB |
| SAM ViT-L | Refinamento interativo com point prompts | CPU | ~1.5GB |

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

O endpoint de refinamento parte da mascara IS-Net como base. Para cada ponto do usuario, SAM segmenta a regiao local (sempre a menor mascara) e subtrai ou adiciona da mascara base.

```bash
curl -X POST http://llm-gpu-linux:8002/remove-background/refine \
  -F "image=@foto_peca.jpg" \
  -F 'points=[{"x": 300, "y": 250, "label": 0}]' \
  -F "background=white" \
  -o resultado_refinado.jpg
```

Points:
- `label=0` — "isto e fundo" (remover da mascara)
- `label=1` — "isto e objeto" (adicionar a mascara)

### Backgrounds disponiveis

- `white` (255, 255, 255) — padrao para marketplaces
- `gray` (224, 224, 224) — catalogo interno

## Setup (VM 151)

```bash
# 1. Criar venv
python3 -m venv /home/amello/bg-removal-env
source /home/amello/bg-removal-env/bin/activate

# 2. Instalar PyTorch nightly cu128 (Blackwell sm_120)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Instalar demais dependencias
pip install -r requirements.txt

# 4. Baixar SAM ViT-L checkpoint
mkdir -p /home/amello/models
wget -P /home/amello/models/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# 5. Copiar codigo
cp -r . /home/amello/bg-removal-service/

# 6. Instalar systemd service
sudo cp systemd/bg-removal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bg-removal

# 7. Verificar
curl http://localhost:8002/health
```

## Acesso

- Tailscale: `http://llm-gpu-linux:8002`
- Local: `http://192.168.68.193:8002`

## Logs

```bash
sudo journalctl -u bg-removal -f
```
