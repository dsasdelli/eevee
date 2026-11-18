"""Regras de nome e de serie da arte de anime, sem rede.

Separadas do anime-downloader.py para poderem ser usadas sem rede (processamento,
conferencia do indice).
"""
import re

NUMERO = re.compile(r'^File:(\d{3,4})(.+)\.(png|jpe?g|webp)$', re.I)

# Siglas das series do anime, como a wiki usa nos nomes de arquivo.
SERIES = {'OS', 'AG', 'DP', 'BW', 'XY', 'SM', 'JN', 'HZ',  # series principais
          'PJ', 'PH', 'SS'}                                # especiais e curtas
SIGLA = re.compile(r'\b(' + '|'.join(sorted(SERIES)) + r')\b')
FILME = re.compile(r'\bM(\d{1,2})\b')
# Nao e arte do anime, apesar de estar na categoria: Mystery Dungeon, Smash Bros.
OUTRO = re.compile(r'Mystery Dungeon|SSBB', re.I)


def nome_arquivo(numero, resto, ext):
    """(29, 'Nidoran♀ AG anime', 'png') -> '0029-nidoran-ag-anime.png'

    O numero da Pokedex entra no nome porque so o texto nao basta: Nidoran♀ e
    Nidoran♂ perdem o simbolo na normalizacao e colidiam no mesmo arquivo.
    """
    base = re.sub(r'[^a-z0-9]+', '-', resto.lower()).strip('-')
    return f'{numero:04d}-{base}.{ext.lower().replace("jpeg", "jpg")}'


def nomes_unicos(analisados):
    """{titulo: nome de arquivo}, sem colisao.

    `analisados` e {titulo: (numero, resto, ext)}. A wiki tem titulos que so diferem na
    caixa (`0052Meowth AG Anime 2` e `... anime 2`), e o Windows nao distingue caixa
    no nome de arquivo. Nesses grupos TODOS ganham um sufixo do hash do titulo: a regra
    nao depende da ordem e nao muda o nome de mais ninguem.
    """
    import hashlib
    from collections import defaultdict
    grupos = defaultdict(list)
    for titulo, a in analisados.items():
        grupos[nome_arquivo(*a)].append(titulo)
    saida = {}
    for nome, titulos in grupos.items():
        for t in titulos:
            if len(titulos) == 1:
                saida[t] = nome
            else:
                base, ext = nome.rsplit('.', 1)
                saida[t] = f'{base}-{hashlib.md5(t.encode()).hexdigest()[:6]}.{ext}'
    return saida


def categoria(resto):
    """'Zorua anime 3' -> 'Anime'; 'Florges anime XY' -> 'Anime XY'; 'Xerneas M17' -> 'Filme M17'."""
    if OUTRO.search(resto):
        return 'Outro'
    m = SIGLA.search(resto)
    if m:
        return f'Anime {m.group(1)}'
    m = FILME.search(resto)
    if m:
        return f'Filme M{m.group(1)}'
    return 'Anime'


def analisa(titulo):
    """'File:0029Nidoran♀ AG anime.png' -> (29, 'Nidoran♀ AG anime', 'png'), ou None."""
    m = NUMERO.match(titulo)
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip(), m.group(3)
