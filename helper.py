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
DIR_ARTWORK = 'artwork'
DIR_ARTWORK_PROCESSED = 'artwork-processed'
DIR_ARTWORK_PROCESSED_COUNTOURS = 'artwork-processed-countours'
# Arte de anime da Pokemon Wiki do Fandom (anime-downloader.py). Fica SEPARADA de
# artwork/ para nao mudar o split de teste das artes, sorteado sobre artwork-processed.
DIR_ARTWORK_ANIME = 'artwork-anime'
DIR_ARTWORK_ANIME_PROCESSED = 'artwork-anime-processed'
# As artes tem lado maior de 731 px em media; 512 aproveita esse detalhe e casa com a
# resolucao em que o Stable Diffusion gera. Os sprites continuam em 256.
ARTWORK_SIZE = 512