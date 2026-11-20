"""Gera splits/teste-especies.txt: o teste das artes, 10% das ESPECIES, comum aos 3 grupos.

POR QUE POR ESPECIE
O teste anterior (100 artes por grupo) tinha a mesma especie no treino em outra arte para a
maioria das artes de teste (58 de 92 especies no sugimori), toda especie de "outros" existe
no sugimori, e a variante shiny tem o MESMO traco da normal. Com o contorno de teste como
entrada do leaderboard, o modelo podia so repetir o que decorou. Aqui uma especie de teste
sai de TODOS os grupos, em todas as formas.

POR QUE 90/10 E NAO 80/20
As bases tem 1,1-1,7 mil artes; 10% ja da ~110-170 de teste por grupo. O ruido do KID cai
com o numero de ENTRADAS do leaderboard, nao com mais teste.

REPRESENTATIVO
KID e distancia entre conjuntos (nao ha KID de uma imagem so). Sorteia N splits por especie,
descarta os que deixam algum grupo fora de 8-12% das artes, e fica com o de menor PIOR
KID(teste_g, treino_g) entre os grupos, em features DINOv2 (as mesmas do leaderboard). Usa
so os dados, nenhum modelo: a escolha nao vaza nada para a avaliacao.

    python splits/gera_split.py              (da raiz do projeto)
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from helper import GRUPOS, especies  # noqa: E402

SAIDA = RAIZ / 'splits' / 'teste-especies.txt'
FRACAO = 0.10
FAIXA = (0.08, 0.12)


def features(caminhos, cache):
    if cache.exists():
        f = np.load(cache)
        if len(f) == len(caminhos):
            return f.astype(np.float32)
    from PIL import Image
    from leaderboard import Dino
    dino = Dino()
    f = np.concatenate([dino([Image.open(p) for p in caminhos[i:i + 256]])
                        for i in range(0, len(caminhos), 256)])
    np.save(cache, f.astype(np.float16))
    return f.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sorteios', type=int, default=500)
    p.add_argument('--cache', default=str(RAIZ / 'splits' / 'features-artes.npy'),
                   help='features DINOv2 de todas as artes (nao versionar)')
    args = p.parse_args()

    from leaderboard import kid
    mapas = {g: especies(g) for g in GRUPOS}
    caminhos = [c for g in GRUPOS for c in mapas[g]]
    f = features(caminhos, Path(args.cache))
    idx, i = {}, 0
    for g in GRUPOS:
        idx[g] = np.arange(i, i + len(mapas[g]))
        i += len(mapas[g])
    esp = {g: np.array(list(mapas[g].values())) for g in GRUPOS}
    todas = sorted(set().union(*(set(v) for v in esp.values())))
    k = round(FRACAO * len(todas))
    print(f'{len(caminhos)} artes, {len(todas)} especies, {k} de teste por sorteio')

    resultados = []
    for semente in range(args.sorteios):
        teste = set(random.Random(semente).sample(todas, k))
        fr, kids = {}, {}
        for g in GRUPOS:
            m = np.isin(esp[g], list(teste))
            fr[g] = m.mean()
            if not FAIXA[0] <= fr[g] <= FAIXA[1]:
                break
            kids[g] = kid(f[idx[g][m]], f[idx[g][~m]]) * 1000
        else:
            resultados.append((max(kids.values()), semente, kids, fr))
    if not resultados:
        raise SystemExit('nenhum sorteio dentro da faixa')
    resultados.sort(key=lambda r: r[0])
    pior, semente, kids, fr = resultados[0]
    mediana = float(np.median([r[0] for r in resultados]))
    print(f'{len(resultados)} de {args.sorteios} sorteios dentro de {FAIXA}')
    print(f'escolhido: semente {semente}, pior KID {pior:.1f} (mediana dos validos {mediana:.1f})')
    for g in GRUPOS:
        print(f'  {g}: {fr[g]:.1%} das artes, KID(teste, treino) {kids[g]:.1f}')

    teste = sorted(random.Random(semente).sample(todas, k))
    cab = [f'# Teste por especie, gerado por splits/gera_split.py. NAO editar a mao.',
           f'# {k} de {len(todas)} especies ({FRACAO:.0%}); semente {semente} de {args.sorteios} sorteios,',
           f'# escolhida pelo menor pior KID(teste, treino) entre os grupos: {pior:.1f} '
           f'(mediana dos {len(resultados)} validos: {mediana:.1f}).',
           '# ' + ', '.join(f'{g} {fr[g]:.1%} / KID {kids[g]:.1f}' for g in GRUPOS)]
    SAIDA.write_text('\n'.join(cab + teste) + '\n', encoding='utf-8')
    print(f'-> {SAIDA}')


if __name__ == '__main__':
    main()
