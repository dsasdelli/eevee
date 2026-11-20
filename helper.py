GENERATIONS = {
    "red-blue": "gen1",
    "silver": "gen2",
    "ruby-sapphire": "gen3",
    "diamond-pearl": "gen4",
    "black-white": "gen5",
    "x-y": "gen6",
    "sun-moon": "gen7",
    "sword-shield": "gen8",
    "scarlet-violet": "gen9"
}

DIR_BASE = 'sprites'
DIR_PROCESSED = 'sprites-processed'
DIR_PROCESSED_COUNTOURS = 'sprites-processed-countours'
# A arte se divide em tres GRUPOS, cada um com as proprias pastas:
#   sugimori  arte oficial de Ken Sugimori, incluindo a Early (anos 90)   -- PokemonDB
#   outros    Global Link e Other (marketing, eventos)                     -- PokemonDB
#   anime     arte de anime da Pokemon Wiki do Fandom (anime-downloader.py)
# O teste e uma lista fixa de ESPECIES em splits/teste-especies.txt, comum aos tres grupos
# (ver lora_train.py e splits/gera_split.py).
GRUPOS = ('sugimori', 'outros', 'anime')


def dir_artes(grupo):
    """Arte original baixada, com o indice.csv do downloader."""
    return f'artwork-{grupo}'


def dir_processadas(grupo):
    """Arte recortada e reduzida para ARTWORK_SIZE (artwork-processer.py)."""
    return f'artwork-{grupo}-processed'


def dir_contornos(grupo):
    """Contorno de cada arte processada (artwork-contours.py)."""
    return f'artwork-{grupo}-processed-countours'


def especies(grupo):
    """{caminho da arte processada: especie} de um grupo, pelo indice.csv do downloader.

    O processado guarda o nome (stem) do original e a geracao; a especie vem do indice.
    Arte processada sem linha no indice e erro: o split por especie nao saberia onde po-la.
    """
    import csv
    from pathlib import Path
    por_chave = {}
    with open(Path(dir_artes(grupo)) / 'indice.csv', encoding='utf-8') as f:
        for lin in csv.DictReader(f):
            stem = Path(lin['arquivo'].replace(chr(92), '/')).stem
            por_chave[(lin['geracao'], stem)] = lin['pokemon']
    saida = {}
    for p in sorted(Path(dir_processadas(grupo)).glob('gen*/*.png')):
        e = por_chave.get((p.parent.name, p.stem))
        if e is None:
            raise SystemExit(f'{p} nao esta no indice.csv de {grupo}')
        saida[p] = e
    return saida


# As artes tem lado maior de 731 px em media; 512 aproveita esse detalhe e casa com a
# resolucao em que o Stable Diffusion gera. Os sprites continuam em 256.
ARTWORK_SIZE = 512