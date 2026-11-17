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
# As artes tem lado maior de 731 px em media; 512 aproveita esse detalhe e casa com a
# resolucao em que o Stable Diffusion gera. Os sprites continuam em 256.
ARTWORK_SIZE = 512