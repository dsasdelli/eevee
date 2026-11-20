"""Folha 28: todos os desenhos humanos (pokemonN.bmp) nas LoRAs de arte retreinadas no split por especie
(19/11/2026), contra o SD puro.

Usa o MESMO pipeline do leaderboard v4 (leaderboard.Difusao: prompt, negativo e
para_controlnet do app; ControlNet 0,7, 28 passos, cfg 7,5), semente 42, saida CRUA sem
tirar_fundo. Imprime a fracao de saidas com fundo (mesmo criterio do leaderboard).
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import leaderboard as L  # noqa: E402
from esboco import fit_sketch  # noqa: E402

SAIDA = RAIZ / 'predictions/difusao/28-especies-desenhos.png'
SEMENTE = 42
COLUNAS = [('sugimori 12k', 'lora-sugimori-aug/lora-step012000.safetensors'),
           ('sug+outros 8k', 'lora-sugimori-outros-aug/lora-step008000.safetensors'),
           ('anime 8k', 'lora-anime-aug/lora-step008000.safetensors'),
           ('sug+outros+anime 12k', 'lora-sugimori-outros-anime-aug/lora-step012000.safetensors'),
           ('SD puro', 'puro')]


def desenhos():
    """Todos os sprites-human-drawn/pokemonN.bmp, em ordem de N (copias e outros nomes ficam de fora)."""
    ps = [p for p in (RAIZ / 'sprites-human-drawn').glob('pokemon*.bmp') if p.stem[7:].isdigit()]
    ps.sort(key=lambda p: int(p.stem[7:]))
    return ps


def main():
    ps = desenhos()
    print('desenhos:', ', '.join(p.stem for p in ps), flush=True)
    esb = [fit_sketch(Image.open(p), L.LADO) for p in ps]
    dif = L.Difusao()
    colunas = []
    for rot, fonte in COLUNAS:
        dif.carregar(fonte)
        saidas = dif.gerar(esb, [SEMENTE])
        print(f'{rot:22s} com fundo: {sum(L.com_fundo(s) for s in saidas)}/{len(saidas)}', flush=True)
        colunas.append((rot, saidas))

    W, H, M = 256, 26, 6
    cols = len(colunas) + 1
    folha = Image.new('RGB', (M + cols * (W + M), H + len(esb) * (W + M) + M), (215, 215, 215))
    d = ImageDraw.Draw(folha)
    d.text((M + 8, 8), 'desenho', fill=(0, 0, 0))
    for c, (rot, _) in enumerate(colunas, start=1):
        d.text((M + c * (W + M) + 8, 8), rot, fill=(0, 0, 0))
    for r, e in enumerate(esb):
        y = H + r * (W + M)
        folha.paste(e.convert('RGB').resize((W, W)), (M, y))
        for c, (_, s) in enumerate(colunas, start=1):
            folha.paste(s[r].resize((W, W), Image.LANCZOS), (M + c * (W + M), y))
    folha.save(SAIDA)
    print('->', SAIDA)


if __name__ == '__main__':
    main()
