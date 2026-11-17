"""Augmentacao on-the-fly das artes oficiais para o treino da LoRA de estilo.

Fica em modulo proprio, e nao dentro do lora_train.py, para que a folha de inspecao
(aug_folha.py) possa usar exatamente a mesma funcao sem carregar o diffusers.

AMPLITUDE PEQUENA DE PROPOSITO. A LoRA aprende tambem ENQUADRAMENTO. No experimento de
margem (predictions/difusao, folha 17), reduzir o bicho dentro do quadro quebrou a
fidelidade ao esboco -- com 20% de margem a serpente virou uma foca, porque o ControlNet
guia proporcionalmente ao quanto do quadro o traco cobre. Por isso o zoom-out para em 5%
e a rotacao em 7 graus.

SEM MEXER NA COR alem de um brilho/contraste minimo: a paleta oficial e parte do
acabamento que a LoRA tem que aprender. Matiz e saturacao ficam intactos.

FUNDO. Rotacao e zoom-out abrem cantos. Arte com fundo branco recebe branco. Arte com
fundo de cenario recebe REFLEXO da borda: preencher de branco criaria cunhas brancas sobre
o verde/cinza, e a LoRA aprenderia a pinta-las.
"""
import math

import numpy as np
from PIL import Image, ImageEnhance

AUG_FLIP = 0.5               # espelhamento horizontal; nunca vertical, nunca 90 graus
AUG_AFIM_P = 0.7             # fracao das amostras que recebe rotacao/escala/translacao
AUG_ROT = 7.0                # graus, +-
AUG_ESCALA = (0.95, 1.10)    # < 1 encolhe o bicho: no maximo 5%, ver folha 17
AUG_DESLOC = 0.03            # fracao do lado, +-
AUG_COR_P = 0.3
AUG_COR = 0.05               # brilho e contraste, +-
PAD_REFLEXO = 0.16           # folga do reflexo; cobre o pior caso de rotacao+zoom-out+desloc

BRANCO = 235                 # canal minimo acima disto conta como branco
BORDA_BRANCA_FRAC = 0.8


def borda_branca(im):
    """True se a borda da imagem e (quase toda) branca, isto e, fundo de ficha."""
    a = np.asarray(im.convert('RGB'))
    borda = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return float((borda.min(1) > BRANCO).mean()) >= BORDA_BRANCA_FRAC


def _afim(im, ang, esc, tx, ty, fundo_branco):
    """Uma unica reamostragem BICUBIC: rotacao `ang` graus, escala `esc`, translacao em px.

    O PIL pede o mapeamento INVERSO (saida -> entrada): entrada = R(-ang)/esc * (saida -
    centro - t) + centro_entrada.
    """
    w, h = im.size
    if fundo_branco:
        fonte, pad = im, 0
    else:
        pad = int(round(PAD_REFLEXO * max(w, h)))
        fonte = Image.fromarray(np.pad(np.asarray(im), ((pad, pad), (pad, pad), (0, 0)),
                                       mode='reflect'))
    cx, cy = w / 2, h / 2
    r = math.radians(ang)
    co, si = math.cos(r) / esc, math.sin(r) / esc
    # inversa de [cos -sin; sin cos]*esc e [cos sin; -sin cos]/esc
    a, b = co, si
    d, e = -si, co
    ox, oy = cx + tx, cy + ty
    c = cx + pad - a * ox - b * oy
    f = cy + pad - d * ox - e * oy
    return fonte.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                           resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def augmenta(im, rng, fundo_branco):
    """Devolve uma variacao de `im` (PIL RGB). `rng` e um numpy Generator."""
    if rng.random() < AUG_FLIP:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < AUG_AFIM_P:
        lado = max(im.size)
        im = _afim(im,
                   ang=rng.uniform(-AUG_ROT, AUG_ROT),
                   esc=rng.uniform(*AUG_ESCALA),
                   tx=rng.uniform(-AUG_DESLOC, AUG_DESLOC) * lado,
                   ty=rng.uniform(-AUG_DESLOC, AUG_DESLOC) * lado,
                   fundo_branco=fundo_branco)
    if rng.random() < AUG_COR_P:
        # O fundo branco e RESTAURADO depois do ajuste. Sem isso, -5% de brilho leva o
        # 255 a ~242 e o fundo sai cinza visivel (folha 22, primeira versao) -- a LoRA
        # aprenderia fundo nao-branco, que e justamente o defeito que se quer tirar.
        fundo = np.asarray(im).min(2) > BRANCO
        im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-AUG_COR, AUG_COR))
        im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-AUG_COR, AUG_COR))
        a = np.array(im)
        a[fundo] = 255
        im = Image.fromarray(a)
    return im


AUG_PIX_DESLOC = 8           # px de SPRITE (256), +-; deslocamento sempre inteiro


def augmenta_pixel(im, rng):
    """Variacao de um SPRITE de pixel art, aplicada em 256 px ANTES do resize NEAREST.

    Sem rotacao nem escala: qualquer interpolacao criaria tons intermediarios e ensinaria
    a LoRA a borrar a borda dura, que e a assinatura do estilo (ver lora_train.py). So
    espelho, deslocamento inteiro com fundo branco e brilho/contraste com branco restaurado.
    """
    if rng.random() < AUG_FLIP:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    dx, dy = (int(v) for v in rng.integers(-AUG_PIX_DESLOC, AUG_PIX_DESLOC + 1, size=2))
    if dx or dy:
        tela = Image.new('RGB', im.size, (255, 255, 255))
        tela.paste(im, (dx, dy))
        im = tela
    if rng.random() < AUG_COR_P:
        fundo = np.asarray(im).min(2) > BRANCO
        im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-AUG_COR, AUG_COR))
        im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-AUG_COR, AUG_COR))
        a = np.array(im)
        a[fundo] = 255
        im = Image.fromarray(a)
    return im
