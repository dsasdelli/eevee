"""Folha de inspecao da augmentacao, para conferir A OLHO antes de gastar horas de GPU.

12 artes fixas (6 com fundo branco, 6 com cenario), a original e 6 variacoes de cada.
Conferir: sem cunha branca no cenario, sem borda dura, bicho nao encolhe.

    python aug_folha.py
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from aug_artwork import augmenta, borda_branca
from helper import DIR_ARTWORK_PROCESSED

SAIDA = Path('predictions/difusao/22-augmentacao-amostras.png')
LADO = 192
VARIACOES = 6


def folha_sprites():
    """12 sprites (gen1 e gen9), original + 6 variacoes, ampliados em NEAREST como no treino."""
    from aug_artwork import augmenta_pixel
    from esboco import list_pairs
    _, reais = list_pairs()
    rnd = random.Random(7)
    escolha = (rnd.sample([p for p in reais if Path(p).parent.name == 'gen1'], 6)
               + rnd.sample([p for p in reais if Path(p).parent.name == 'gen9'], 6))
    rng = np.random.default_rng(0)
    folha = Image.new('RGB', (LADO * (VARIACOES + 1), LADO * len(escolha)), (128, 128, 128))
    for li, p in enumerate(escolha):
        im = Image.open(p).convert('RGB')
        celulas = [im] + [augmenta_pixel(im, rng) for _ in range(VARIACOES)]
        for ci, c in enumerate(celulas):
            folha.paste(c.resize((LADO, LADO), Image.NEAREST), (ci * LADO, li * LADO))
    saida = SAIDA.with_name('25-augmentacao-sprites.png')
    folha.save(saida)
    print(f'salvo {saida}')


def main():
    import sys
    if '--sprites' in sys.argv:
        return folha_sprites()
    todos = sorted(Path(DIR_ARTWORK_PROCESSED).glob('gen*/*.png'))
    random.Random(7).shuffle(todos)
    brancas, cenario = [], []
    for p in todos:
        alvo = brancas if borda_branca(Image.open(p)) else cenario
        if len(alvo) < 6:
            alvo.append(p)
        if len(brancas) == 6 and len(cenario) == 6:
            break
    linhas = [(p, True) for p in brancas] + [(p, False) for p in cenario]

    rng = np.random.default_rng(0)
    folha = Image.new('RGB', (LADO * (VARIACOES + 1), LADO * len(linhas)), (128, 128, 128))
    for li, (p, branca) in enumerate(linhas):
        im = Image.open(p).convert('RGB')
        celulas = [im] + [augmenta(im, rng, branca) for _ in range(VARIACOES)]
        for ci, c in enumerate(celulas):
            folha.paste(c.resize((LADO, LADO), Image.LANCZOS), (ci * LADO, li * LADO))
        ImageDraw.Draw(folha).text((4, li * LADO + 4),
                                   f'{p.stem} [{"branco" if branca else "reflexo"}]',
                                   fill=(255, 0, 0))
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    folha.save(SAIDA)
    print(f'salvo {SAIDA}  ({len(brancas)} brancas, {len(cenario)} com cenario)')


if __name__ == '__main__':
    main()
