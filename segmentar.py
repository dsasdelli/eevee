"""Recorte com clique: o SAM (facebook/sam-vit-base) acha a regiao clicada na arte
COLORIDA e devolve o contorno dela como poligono, no mesmo formato do laco manual.

Por que na arte colorida e nao no contorno: no contorno preto e branco a orelha e a
cabeca sao o mesmo branco, separados so por uma linha; na arte elas tem cor, sombra e
borda proprias, que e o que o SAM usa para separar regioes.

Por que poligono e nao mascara: o usuario ve o laco proposto, corrige se quiser e salva
pelo mesmo caminho do recorte manual (pecas.salvar_peca), sem um segundo formato de peca.
"""
import threading
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image

import pecas

MODELO = 'facebook/sam-vit-base'
# A mascara do SAM para na borda da COR; o traco preto do contorno fica logo fora dela.
# Alargar alguns pixels traz a linha de borda para dentro da peca.
ALARGAR = 4

_trava = threading.Lock()
_sam = None                 # (modelo, processador, dispositivo), carregado no 1o uso
_cache = OrderedDict()      # (geracao, nome) -> embedding da imagem; o caro e isto
CACHE_MAX = 4


def _carregar():
    global _sam
    if _sam is None:
        from transformers import SamModel, SamProcessor
        disp = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'  carregando {MODELO} ({disp})...', flush=True)
        modelo = SamModel.from_pretrained(MODELO).to(disp).eval()
        _sam = (modelo, SamProcessor.from_pretrained(MODELO), disp)
    return _sam


def poligono(geracao, nome, positivos, negativos=()):
    """Pontos em pixels da imagem de 512. Devolve (poligono [x0, y0, ...], score)."""
    arq = pecas.caminho_imagem('arte', geracao, nome)
    if arq is None:
        raise ValueError(f'arte nao encontrada: {geracao}/{nome}')
    try:
        pts = [[float(x), float(y)] for x, y in positivos] + [[float(x), float(y)] for x, y in negativos]
    except (TypeError, ValueError):
        raise ValueError('pontos invalidos')
    if not positivos:
        raise ValueError('clique pelo menos uma vez dentro da peca')
    rotulos = [1] * len(positivos) + [0] * len(negativos)

    with _trava, torch.inference_mode():
        modelo, proc, disp = _carregar()
        img = Image.open(arq).convert('RGB')
        chave = (geracao, nome)
        entrada = proc(img, input_points=[[pts]], input_labels=[[rotulos]], return_tensors='pt').to(disp)
        if chave not in _cache:
            _cache[chave] = modelo.get_image_embeddings(entrada['pixel_values'])
            while len(_cache) > CACHE_MAX:
                _cache.popitem(last=False)
        _cache.move_to_end(chave)
        saida = modelo(image_embeddings=_cache[chave], input_points=entrada['input_points'],
                       input_labels=entrada['input_labels'], multimask_output=True)
        mascaras = proc.image_processor.post_process_masks(
            saida.pred_masks.cpu(), entrada['original_sizes'].cpu(),
            entrada['reshaped_input_sizes'].cpu())[0][0]          # (3, H, W)
        scores = saida.iou_scores[0, 0].float().cpu().numpy()

    # Das tres propostas do SAM (parte pequena, media, objeto inteiro), a de maior
    # confianca. Com um clique so, ele costuma preferir o bicho todo; os cliques
    # negativos (fora da peca) e que o empurram para a parte.
    i = int(scores.argmax())
    m = mascaras[i].numpy().astype(np.uint8)
    if ALARGAR:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ALARGAR + 1,) * 2))
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise ValueError('o SAM nao achou nada ali')
    # So a regiao que contem o clique (ou a maior, se o clique caiu na borda).
    x0, y0 = positivos[0]
    dentro = [c for c in contornos if cv2.pointPolygonTest(c, (float(x0), float(y0)), False) >= 0]
    c = max(dentro or contornos, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)       # menos pontos, mesma forma (1 px de tolerancia)
    return [float(v) for v in c.reshape(-1)], float(scores[i])
