"""Biblioteca de pecas (olhos, orelhas, caudas...) recortadas dos contornos de artwork.

Uma peca e um PNG RGBA tirado de `artwork-processed-countours`: dentro do laco fica o
contorno como esta (traco preto, interior BRANCO OPACO), fora fica transparente. O
branco opaco faz as pecas se cobrirem como papel recortado: uma cabeca posta sobre o
corpo esconde as linhas do corpo por baixo dela.

O recorte e feito aqui, e nao no navegador, para sair do PNG original de 512 px sem
nenhuma reamostragem: o traco da peca e exatamente o traco que o modelo viu no treino.
"""
import json
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).parent
CONTORNOS = RAIZ / 'artwork-processed-countours'
ARTES = RAIZ / 'artwork-processed'
PECAS = RAIZ / 'pecas'
INDICE = PECAS / 'indice.json'

CATEGORIAS = ['olho', 'orelha', 'boca', 'nariz', 'cauda', 'pata', 'asa', 'chifre',
              'cabeca', 'corpo', 'outro']
MARGEM = 4

_NOME = re.compile(r'^[a-z0-9][a-z0-9-]*$')
_GERACAO = re.compile(r'^gen[1-9]$')
_trava = threading.Lock()


def listar_pokemons():
    """[{nome, geracao}] de todo contorno que tem a arte colorida correspondente."""
    saida = []
    for g in sorted(CONTORNOS.iterdir()):
        if not (g.is_dir() and _GERACAO.match(g.name)):
            continue
        for f in sorted(g.glob('*.png')):
            if (ARTES / g.name / f.name).exists():
                saida.append({'nome': f.stem, 'geracao': g.name})
    return saida


def caminho_imagem(tipo, geracao, nome):
    """Arquivo do contorno ou da arte, ou None se o pedido nao for valido. Valida por
    expressao regular ANTES de montar o caminho: nada vindo da URL vira '..'."""
    if tipo not in ('contorno', 'arte') or not _GERACAO.match(geracao or '') \
            or not _NOME.match(nome or ''):
        return None
    f = (CONTORNOS if tipo == 'contorno' else ARTES) / geracao / f'{nome}.png'
    return f if f.exists() else None


def ler_indice():
    if not INDICE.exists():
        return []
    return json.loads(INDICE.read_text(encoding='utf-8'))


def _gravar_indice(itens):
    PECAS.mkdir(exist_ok=True)
    tmp = INDICE.with_suffix('.tmp')
    tmp.write_text(json.dumps(itens, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(INDICE)          # troca atomica: um erro no meio nao corrompe o indice


def salvar_peca(geracao, nome, categoria, poligono):
    """Recorta `poligono` ([x0, y0, x1, y1, ...] em pixels do contorno de 512 px) e
    devolve a entrada nova do indice. Levanta ValueError com mensagem para o usuario."""
    if categoria not in CATEGORIAS:
        raise ValueError(f'categoria desconhecida: {categoria}')
    origem = caminho_imagem('contorno', geracao, nome)
    if origem is None:
        raise ValueError(f'contorno nao encontrado: {geracao}/{nome}')
    try:
        pts = [(float(poligono[i]), float(poligono[i + 1]))
               for i in range(0, len(poligono) - 1, 2)]
    except (TypeError, ValueError):
        raise ValueError('poligono invalido')
    if len(pts) < 3:
        raise ValueError('o laco precisa de pelo menos 3 pontos')

    contorno = Image.open(origem).convert('L')
    mascara = Image.new('L', contorno.size, 0)
    ImageDraw.Draw(mascara).polygon(pts, fill=255)
    caixa = mascara.getbbox()
    if caixa is None:
        raise ValueError('o laco nao cobre nenhum pixel')
    x0, y0, x1, y1 = caixa
    w, h = contorno.size
    caixa = (max(0, x0 - MARGEM), max(0, y0 - MARGEM), min(w, x1 + MARGEM), min(h, y1 + MARGEM))

    rgba = Image.merge('RGBA', (contorno, contorno, contorno, mascara)).crop(caixa)
    # Peca sem nenhuma tinta e quase sempre um laco errado (caiu no fundo branco).
    # Tinta e ESCURO (< 128), como o pix2pix_data.py le o traco.
    a = np.asarray(rgba)
    if not ((a[..., 0] < 128) & (a[..., 3] > 0)).any():
        raise ValueError('o laco nao pegou nenhum traco')

    with _trava:
        itens = ler_indice()
        usados = {i['id'] for i in itens}
        n = 1
        while f'{nome}-{categoria}-{n}' in usados:
            n += 1
        ident = f'{nome}-{categoria}-{n}'
        pasta = PECAS / categoria
        pasta.mkdir(parents=True, exist_ok=True)
        rgba.save(pasta / f'{ident}.png', optimize=True)
        item = {'id': ident, 'categoria': categoria, 'pokemon': nome, 'geracao': geracao,
                'arquivo': f'{categoria}/{ident}.png', 'largura': rgba.width,
                'altura': rgba.height, 'caixa': list(caixa)}
        itens.append(item)
        _gravar_indice(itens)
    return item


def apagar_peca(ident):
    """Apaga UMA peca pelo id exato: o arquivo e a entrada do indice."""
    with _trava:
        itens = ler_indice()
        alvo = next((i for i in itens if i['id'] == ident), None)
        if alvo is None:
            raise KeyError(ident)
        arquivo = PECAS / alvo['arquivo']
        if arquivo.exists():
            arquivo.unlink()
        _gravar_indice([i for i in itens if i['id'] != ident])
