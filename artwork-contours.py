"""Extrai o contorno das artes processadas, como `sprite-contours.py` faz com sprites.

Mesmo encadeamento, pelos mesmos motivos: desfoque gaussiano para o limiar nao reagir
a ruido de compressao, limiar ADAPTATIVO (e nao global, porque arte oficial tem
iluminacao desigual e um limiar unico perderia o lado escuro do bicho), e
`drawContours` preenchido para fechar o traco em vez de deixa-lo picotado.

DENSIDADE DE TINTA, QUE AQUI NUNCA FOI DETALHE

A lacuna entre o traco do dataset e o de um desenho humano foi o que quebrou o gerador
antes de ser corrigida, entao a densidade e medida, nao suposta. Medido em 09/11/2026,
com os MESMOS parametros do script de sprites:

    sprites-processed-countours (256 px)   13,4% de tinta
    contornos de artwork (512 px)           7,3% de tinta  (sugimori + outros juntos)
    desenho humano de verdade               4% a 8%

Eu esperava o contrario -- a 512 a janela do limiar cobre metade da area relativa, e
supus que o traco sairia mais denso. Sai MENOS: arte oficial e lisa, com gradiente
suave, enquanto pixel art tem borda dura em todo lugar que o limiar adaptativo le como
detalhe. O efeito pratico e bom: o contorno de artwork ja nasce na faixa de um desenho
humano, que era exatamente o problema que a augmentacao de traco existia para remendar.

Os parametros ficam na linha de comando para poder calibrar isso. `--amostra N` mede
sem gravar nada.

Por padrao gera os contornos de sugimori e outros (artwork-<grupo>-processed-countours),
que sao os que o estudio usa para recortar pecas; --grupo anime gera os do anime.

    python artwork-contours.py                      padrao, igual ao de sprites
    python artwork-contours.py --bloco 9 --c 3      traco mais limpo
    python artwork-contours.py --amostra 40         so mede a tinta, nao grava
"""
import argparse
from os import listdir, makedirs, path

import cv2 as cv
import numpy as np

from helper import GRUPOS, dir_contornos, dir_processadas


def contorno(caminho, desfoque, bloco, c):
    im = cv.imread(caminho, 0)
    if im is None:
        return None
    out = np.zeros_like(im)
    im = cv.GaussianBlur(im, (desfoque, desfoque), 0)
    im = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, bloco, c)
    contornos, _ = cv.findContours(im, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(out, contornos, -1, 255, thickness=cv.FILLED)
    return out


def tinta(out):
    """Fracao de pixels de TRACO.

    O traco e ESCURO sobre fundo claro -- conferido em
    `sprites-processed-countours`: 13,4% de pixels escuros contra 86,6% claros, lendo o
    traco com `< 128`. O `drawContours` preenchido pinta o FUNDO
    de branco e o que sobra escuro sao as linhas; medir o que esta aceso daria a
    fracao de fundo, que e o complemento (foi o erro da primeira versao: 92,7%).
    """
    return float((out < 128).mean())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--desfoque', type=int, default=9, help='lado do kernel gaussiano (impar)')
    p.add_argument('--bloco', type=int, default=5, help='janela do limiar adaptativo (impar)')
    p.add_argument('--c', type=int, default=2, help='constante subtraida do limiar')
    p.add_argument('--grupo', choices=GRUPOS, action='append',
                   help='grupo a processar; repita para mais de um (padrao: sugimori e outros)')
    p.add_argument('--amostra', type=int,
                   help='so MEDE a tinta em N imagens e nao grava nada')
    return p.parse_args()


def main():
    args = parse_args()
    if args.desfoque % 2 == 0 or args.bloco % 2 == 0:
        raise SystemExit('--desfoque e --bloco precisam ser impares')

    grupos = args.grupo or ['sugimori', 'outros']
    # (grupo, geracao) de cada pasta de artes processadas.
    pastas = [(g, d) for g in grupos for d in sorted(listdir(dir_processadas(g)))
              if path.isdir(path.join(dir_processadas(g), d))]
    tintas = []
    total = 0

    if args.amostra:
        import random
        random.seed(0)
        todos = [path.join(dir_processadas(g), gen, f)
                 for g, gen in pastas for f in listdir(path.join(dir_processadas(g), gen))]
        for caminho in random.sample(todos, min(args.amostra, len(todos))):
            out = contorno(caminho, args.desfoque, args.bloco, args.c)
            if out is not None:
                tintas.append(tinta(out))
        t = np.array(tintas)
        print(f'desfoque {args.desfoque} bloco {args.bloco} C {args.c} -> '
              f'tinta media {t.mean() * 100:.1f}%  (p10 {np.percentile(t, 10) * 100:.1f} '
              f'p90 {np.percentile(t, 90) * 100:.1f}) em {len(t)} imagens')
        return

    for g, gen in pastas:
        origem = path.join(dir_processadas(g), gen)
        destino = path.join(dir_contornos(g), gen)
        makedirs(destino, exist_ok=True)
        arquivos = listdir(origem)
        for f in arquivos:
            out = contorno(path.join(origem, f), args.desfoque, args.bloco, args.c)
            if out is None:
                print(f'  {gen}/{f}: nao abriu, pulada')
                continue
            cv.imwrite(path.join(destino, f), out)
            tintas.append(tinta(out))
            total += 1
        print(f'{gen}: {len(arquivos)} artes -> {destino}')

    t = np.array(tintas)
    print(f'{chr(10)}{total} contornos | tinta media {t.mean() * 100:.1f}% '
          f'(p10 {np.percentile(t, 10) * 100:.1f} p90 {np.percentile(t, 90) * 100:.1f})')


if __name__ == '__main__':
    main()
