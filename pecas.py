"""Biblioteca de pecas (olhos, orelhas, caudas...) recortadas dos contornos de artwork.

Uma peca e um PNG RGBA tirado dos contornos de artwork (`artwork-sugimori-processed-countours`
e `artwork-outros-processed-countours`) que guarda SO A TINTA:
o traco preto dentro do laco e opaco, todo o resto (o branco de dentro inclusive) e
transparente. Assim uma peca posta sobre outra, ou sobre o lapis, nao apaga o que esta
por baixo: os tracos se somam, como num desenho a mao. (A primeira versao guardava o
interior branco opaco, como papel recortado, e isso escondia o desenho; ver so_tinta.)

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
# Grupos de onde saem as pecas. O nome (geracao/pokemon.png) e unico entre eles, entao o
# indice das pecas guarda so pokemon e geracao, sem o grupo.
GRUPOS = ('sugimori', 'outros')
CONTORNOS = [RAIZ / f'artwork-{g}-processed-countours' for g in GRUPOS]
ARTES = [RAIZ / f'artwork-{g}-processed' for g in GRUPOS]
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
    for contornos, artes in zip(CONTORNOS, ARTES):
        for g in sorted(contornos.iterdir()):
            if not (g.is_dir() and _GERACAO.match(g.name)):
                continue
            for f in sorted(g.glob('*.png')):
                if (artes / g.name / f.name).exists():
                    saida.append({'nome': f.stem, 'geracao': g.name})
    return sorted(saida, key=lambda d: (d['geracao'], d['nome']))


def caminho_imagem(tipo, geracao, nome):
    """Arquivo do contorno ou da arte, ou None se o pedido nao for valido. Valida por
    expressao regular ANTES de montar o caminho: nada vindo da URL vira '..'."""
    if tipo not in ('contorno', 'arte') or not _GERACAO.match(geracao or '') \
            or not _NOME.match(nome or ''):
        return None
    for raiz in (CONTORNOS if tipo == 'contorno' else ARTES):
        f = raiz / geracao / f'{nome}.png'
        if f.exists():
            return f
    return None


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

    rgba = so_tinta(Image.merge('RGBA', (contorno, contorno, contorno, mascara)).crop(caixa))
    # Peca sem nenhuma tinta e quase sempre um laco errado (caiu no fundo branco).
    if not (np.asarray(rgba)[..., 3] > 0).any():
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


def so_tinta(rgba):
    """Deixa so o traco: cor preta, e opacidade = quao escuro o pixel era (vezes a
    opacidade que ja tinha). O branco vira transparente. Os contornos sao binarios,
    entao na pratica o alfa sai 0 ou 255; a formula continua certa se um dia nao forem."""
    a = np.asarray(rgba.convert('RGBA')).astype(np.uint16)
    cinza = a[..., :3].mean(axis=2)
    alfa = ((255 - cinza) * a[..., 3] / 255).round().astype(np.uint8)
    saida = np.zeros(a.shape, np.uint8)
    saida[..., 3] = alfa
    return Image.fromarray(saida, 'RGBA')


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
