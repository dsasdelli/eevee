import requests
import yaml
from os import path, makedirs
from helper import GENERATIONS, DIR_BASE

def genUrl(generation, pokemon):
    return f"https://img.pokemondb.net/sprites/{generation}/normal/{pokemon}.png"

pokemons = []
with open("pokemons.yaml") as f:
    pokemons = yaml.safe_load(f)

for pokemon in pokemons:
    count = 0
    for generation in GENERATIONS.items():
        if(not path.exists(f'{DIR_BASE}\\{generation[1]}')):
            makedirs(f'{DIR_BASE}\\{generation[1]}')
            print(f"Criando diretório de destino: {DIR_BASE}\\{generation[1]}")
        print(f"Baixando Pokemon {pokemon} na geração {generation[0]}")
        response = requests.get(genUrl(generation[0], pokemon))
        if response.status_code == 404:
            print(f"Erro 404 para o Pokemon {pokemon} na geração {generation[0]}")
        else:
            with open(f'{DIR_BASE}\\{generation[1]}\\{pokemon}.png', 'wb') as f:
                f.write(response.content)
                count+=1
    if count == 0:
        print(f"Nenhum Pokemon {pokemon} encontrado")