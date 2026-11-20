"""Inventa um nome para cada Pokemon gerado, misturando os nomes dos reais mais parecidos.

    python nomes.py          teste: sanidade dos vizinhos + 20 nomes inventados

COMO FUNCIONA

A imagem gerada (crua, antes de reduzir) vai para o espaco do DINOv2 -- o mesmo extrator
do leaderboard -- e e comparada por cosseno com uma base unica: as artes de
sugimori e outros (artwork-<grupo>-processed) MAIS os sprites de sprites-processed. As duas juntas para o nome sair
igual bem com modelo de artwork e de sprite. Varias imagens da mesma especie (megas,
formas, sprite e arte) contam como UMA especie, pela melhor parecida.

Dos vizinhos sorteia-se 2, 3 ou 4 (pela semente, entao a mesma semente da o mesmo
nome) e as silabas sao misturadas: o mais parecido da o comeco, o ultimo da o final, os
do meio uma silaba cada.

Os embeddings da base (~5.500 imagens) sao calculados UMA vez e guardados em
nomes-dino.npz; depois a subida so le o arquivo.
"""
import csv
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).parent
CACHE = RAIZ / 'nomes-dino.npz'
VOGAIS = 'aeiouy'
# Anime fica de fora: e um estilo so, e a base serve para reconhecer a ESPECIE.
GRUPOS_BASE = ('sugimori', 'outros')


def _branco(im):
    """Sprite com transparencia vira preto no RGB; achata sobre branco antes."""
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        fundo = Image.new('RGBA', im.size, 'WHITE')
        fundo.paste(im, (0, 0), im)
        return fundo.convert('RGB')
    return im.convert('RGB')


def _base():
    """Lista (caminho, especie) de artes + sprites."""
    itens, especies = [], set()
    from helper import dir_artes, dir_processadas
    for g in GRUPOS_BASE:
        with open(RAIZ / dir_artes(g) / 'indice.csv', encoding='utf-8') as f:
            for lin in csv.DictReader(f):
                stem = Path(lin['arquivo'].replace('\\', '/')).stem
                p = RAIZ / dir_processadas(g) / lin['geracao'] / f'{stem}.png'
                if p.exists():
                    itens.append((p, lin['pokemon']))
                    especies.add(lin['pokemon'])
    for p in sorted((RAIZ / 'sprites-processed').glob('gen*/*.png')):
        stem = p.stem
        # Forma de sprite ("deoxys-attack") cai na especie se o prefixo for conhecido.
        esp = stem if stem in especies else stem.split('-')[0]
        itens.append((p, esp if esp in especies else stem))
    return itens


def silabas(nome):
    """charizard -> cha|ri|zard; consoantes do fim grudam na ultima silaba."""
    s = re.sub(r'[^a-z]', '', nome.lower())
    partes = re.findall(rf'[^{VOGAIS}]*[{VOGAIS}]+', s)
    if not partes:
        return [s] if s else []
    partes[-1] += s[len(''.join(partes)):]
    # Encontro de consoantes entre silabas: a 1a fica na silaba de tras
    # (archen -> ar|chen, nao a|rchen), mas digrafo de inicio (ch, sh, th...) nao se parte.
    for i in range(1, len(partes)):
        while True:
            m = re.match(rf'[^{VOGAIS}]*', partes[i]).group()
            if len(m) < 2 or m[:2] in INICIOS:
                break
            partes[i - 1] += m[0]
            partes[i] = partes[i][1:]
    return partes


# Encontros que podem abrir silaba; os outros se partem (bellsprout -> bell|sprout).
INICIOS = {'bl', 'br', 'ch', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'kr', 'ph', 'pl',
           'pr', 'qu', 'sc', 'sh', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw', 'th', 'tr',
           'tw', 'wh'}


def misturar(nomes, rng, proibidos=()):
    """Comeco do 1o, uma silaba de cada um do meio, final do ultimo. 2 a 4 silabas."""
    proibidos = {re.sub(r'[^a-z]', '', p.lower()) for p in proibidos}
    sil = [silabas(n) for n in nomes]
    sil = [s for s in sil if s]
    if not sil:
        return None
    for _ in range(30):
        s1 = sil[0]
        comeco = s1[:rng.choice([1, 2])] if len(s1) > 2 else s1[:1]
        meio = []
        for s in sil[1:-1]:
            internas = s[1:-1] or s
            meio.append(rng.choice(internas))
        final = sil[-1][-1:] if len(sil) > 1 else s1[-1:]
        partes = comeco + meio + final
        # Tira silaba do meio ate caber em 4 silabas e 12 letras.
        while len(partes) > 2 and (len(partes) > 4 or len(''.join(partes)) > 12):
            partes.pop(rng.randrange(1, len(partes) - 1))
        nome = ''.join(partes)
        nome = re.sub(r'(.)\1\1+', r'\1\1', nome)    # "aaa" -> "aa"
        if 3 <= len(nome) <= 12 and nome not in proibidos:
            return nome.capitalize()
    return None          # so da nome real: quem chama tenta com mais vizinhos


class Nomeador:
    def __init__(self):
        from leaderboard import Dino
        self.dino = Dino()
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=False)
            self.feats, self.especies = z['feats'], list(z['especies'])
        else:
            itens = _base()
            print(f'nomes: calculando DINOv2 de {len(itens)} imagens (so na 1a vez)...',
                  flush=True)
            feats = []
            for i in range(0, len(itens), 256):
                ims = [_branco(Image.open(p)) for p, _ in itens[i:i + 256]]
                feats.append(self.dino(ims))
            self.feats = np.concatenate(feats).astype(np.float32)
            self.especies = [e for _, e in itens]
            np.savez(CACHE, feats=self.feats, especies=np.array(self.especies))
        self.todas = set(self.especies)

    def vizinhos(self, img, k):
        f = self.dino([_branco(img)])[0]
        sim = self.feats @ f
        vistos, saida = set(), []
        for i in np.argsort(-sim):
            e = self.especies[i]
            if e not in vistos:
                vistos.add(e)
                saida.append(e)
                if len(saida) == k:
                    break
        return saida

    def nomear(self, img, semente):
        rng = random.Random(semente)
        k = rng.choice([2, 3, 4])
        viz = self.vizinhos(img, 6)
        # Se a mistura so consegue reproduzir um nome real (archen + archeops), entra
        # mais um vizinho.
        for kk in range(k, 7):
            nome = misturar(viz[:kk], rng, self.todas)
            if nome:
                return {'nome': nome, 'inspiracao': viz[:kk]}
        return {'nome': viz[0].capitalize() + 'mon', 'inspiracao': viz[:1]}


if __name__ == '__main__':
    n = Nomeador()
    print(f'base: {len(n.especies)} imagens, {len(n.todas)} especies')
    for esp in ['pikachu', 'charizard', 'bulbasaur', 'gengar', 'lapras']:
        p = next(RAIZ.glob(f'artwork-sugimori-processed/gen*/{esp}.png'))
        print(f'  {esp:10s} -> {n.vizinhos(Image.open(p), 4)}')
    rng = random.Random(0)
    amostra = sorted(p for g in GRUPOS_BASE for p in RAIZ.glob(f'artwork-{g}-processed/gen*/*.png'))
    for semente, p in enumerate(rng.sample(amostra, 20)):
        r = n.nomear(Image.open(p), semente)
        print(f'  {p.stem:22s} {r["nome"]:14s} <- {", ".join(r["inspiracao"])}')
