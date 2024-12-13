import requests
from bs4 import BeautifulSoup
import json

response = requests.get("https://pokemondb.net/evolution")
soup = BeautifulSoup(response.content, 'html.parser')

family_divs = soup.find_all("div", {"class": "infocard-filter-block"})

# Div do tipo infocard-list-evo
def extractPokemons(div, last_pokemon=None):
    pokemons = {}
    pokemon_divs = div.find_all("div", {"class": "infocard"}, recursive=False)
    for pokemon_div in pokemon_divs:
        link = pokemon_div.find("a", {"class": "ent-name"})
        current_pokemon = link['href'].split('/')[2]
        if last_pokemon is not None:
            pokemons[current_pokemon] = last_pokemon
        last_pokemon = current_pokemon
    return (pokemons, last_pokemon)

pokemons_complete = {}
for family_div in family_divs:
    evolution_divs = family_div.find_all("div", {"class": "infocard-list-evo"}, recursive=False)
    for evolution_div in evolution_divs:
        (pokemons, last_pokemon) = extractPokemons(evolution_div)
        evolution_split_divs = evolution_div.find_all("span", {"class": "infocard-evo-split"}, recursive=False)
        for evolution_split_div in evolution_split_divs:
            evolution_divs_splitted = evolution_split_div.find_all("div", {"class": "infocard-list-evo"}, recursive=False)
            for evolution_div_splitted in evolution_divs_splitted:
                pokemons_complete.update(extractPokemons(evolution_div_splitted, last_pokemon)[0])

        pokemons_complete.update(pokemons)

with open('evolutions.json', 'w') as fp:
    json.dump(pokemons_complete, fp, indent=4)
