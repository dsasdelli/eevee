"""Baixa a arte oficial de ANIME dos Pokemon, da Pokemon Wiki do Fandom.

POR QUE O FANDOM E NAO O BULBAGARDEN

O Bulbagarden (archives e Bulbapedia) responde 403 com desafio do Cloudflare a qualquer
cliente automatizado -- API, paginas, ate o robots.txt (medido em 17/11/2026). Isso e
uma decisao do site, e nao e contornada aqui. A Pokemon Wiki do Fandom roda o mesmo
MediaWiki com a API aberta, e a `Category:Anime artwork` dela tem 2.141 arquivos, dos
quais 2.103 de Pokemon, de 843 especies, quase todos com ~4000 px.

COMO OS ARQUIVOS SAO RECONHECIDOS

O nome segue o padrao `<numero da Pokedex><Nome> <serie> anime <n>.png`, por exemplo
`0001Bulbasaur AG anime 2.png` ou `0026Raichu Alola SM anime 2.png`. O numero leva ao
`pokemons.yaml` (campo `national`), que da o slug e a geracao. Arquivo sem numero
(Ash, Serena, Rotom Dex) nao e de Pokemon e fica de fora.

POR QUE A MINIATURA DE 1024 E NAO O ORIGINAL

O original tem ~4000 px e somaria dezenas de GB. O artwork-processer.py reduz tudo
para 512 de qualquer jeito; 1024 ainda e o dobro disso.

PASTA SEPARADA, DE PROPOSITO

Vai para artwork-anime/, o grupo "anime" (ver helper.py), separado de sugimori e
outros. Anime e outro estilo (traco de cel, sombra chapada) e deve entrar no treino so
quando pedido; o teste e o das especies de splits/teste-especies.txt.

    python anime-downloader.py                     tudo
    python anime-downloader.py --limite 20         so os vinte primeiros arquivos
    python anime-downloader.py --pokemon pikachu,eevee
"""
import argparse
import csv
import re
import time
from os import makedirs, path

from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException
import yaml

from anime_nomes import analisa, categoria, nomes_unicos
from helper import dir_artes

API = 'https://pokemon.fandom.com/api.php'
CATEGORIA = 'Category:Anime artwork'
DIR_ANIME = dir_artes('anime')
INDICE = f'{DIR_ANIME}/indice.csv'
LARGURA = 1024
PAUSA = 0.4
TENTATIVAS = 3
ESPERA_ERRO = 60.0
LOTE = 50            # limite de titulos por consulta de imageinfo da API

CABECALHO = ['pokemon', 'geracao', 'categoria', 'variacao', 'url', 'arquivo']


def busca(sessao, url, params=None, binario=False):
    """GET com tentativas. Devolve None em 404, levanta no resto depois de tentar."""
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            r = sessao.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binario else r.json()
        except (RequestException, ValueError) as e:
            if tentativa == TENTATIVAS:
                raise
            print(f'    tentativa {tentativa} falhou ({type(e).__name__}), repetindo')
            time.sleep(ESPERA_ERRO * tentativa)
    return None


def titulos_da_categoria(sessao):
    titulos, cont = [], {}
    while True:
        r = busca(sessao, API, {'action': 'query', 'format': 'json', 'list': 'categorymembers',
                                'cmtitle': CATEGORIA, 'cmtype': 'file', 'cmlimit': 500, **cont})
        titulos += [m['title'] for m in r['query']['categorymembers']]
        if 'continue' not in r:
            return titulos
        cont = {'cmcontinue': r['continue']['cmcontinue']}


def urls(sessao, titulos):
    """{titulo: (url da miniatura, largura original, altura original)}"""
    saida = {}
    for i in range(0, len(titulos), LOTE):
        r = busca(sessao, API, {'action': 'query', 'format': 'json', 'prop': 'imageinfo',
                                'iiprop': 'url|size', 'iiurlwidth': LARGURA,
                                'titles': '|'.join(titulos[i:i + LOTE])})
        for p in r['query']['pages'].values():
            if 'imageinfo' not in p:
                continue
            ii = p['imageinfo'][0]
            # Se o original for mais estreito que LARGURA, a API devolve o proprio original.
            url = ii.get('thumburl') or ii['url']
            # Sem isto o CDN converte para WebP quando o cliente aceita WebP.
            url += ('&' if '?' in url else '?') + 'format=original'
            saida[p['title']] = (url, ii['width'], ii['height'])
        time.sleep(PAUSA)
    return saida


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--limite', type=int, help='processa so os N primeiros arquivos')
    p.add_argument('--pokemon', help='lista de slugs separada por virgula, para testar')
    p.add_argument('--pausa', type=float, default=PAUSA, help='segundos entre downloads')
    return p.parse_args()


def main():
    args = parse_args()
    with open('pokemons.yaml', encoding='utf-8') as f:
        pokemons = yaml.safe_load(f)
    por_numero = {v['national']: k for k, v in pokemons.items()}

    sessao = requests.Session(impersonate="chrome")
    titulos = titulos_da_categoria(sessao)
    print(f'{CATEGORIA}: {len(titulos)} arquivos')

    alvos, fora_do_yaml, sem_numero = [], [], 0
    for t in titulos:
        a = analisa(t)
        if a is None:
            sem_numero += 1
            continue
        numero, resto, ext = a
        slug = por_numero.get(numero)
        if slug is None:
            fora_do_yaml.append(t)
            continue
        alvos.append((t, slug, numero, resto, ext))
    # Calculado sobre a lista INTEIRA, antes de --pokemon/--limite, para o nome de um
    # arquivo nao depender de quais outros entraram na rodada.
    nomes = nomes_unicos({a[0]: a[2:] for a in alvos})
    print(f'{len(alvos)} de Pokemon | {sem_numero} sem numero (personagens) | '
          f'{len(fora_do_yaml)} com numero fora do pokemons.yaml')

    if args.pokemon:
        quero = {n.strip() for n in args.pokemon.split(',') if n.strip()}
        alvos = [a for a in alvos if a[1] in quero]
    if args.limite:
        alvos = alvos[:args.limite]

    makedirs(DIR_ANIME, exist_ok=True)
    novo_indice = not path.exists(INDICE) or path.getsize(INDICE) == 0
    ja_no_indice = set()
    if not novo_indice:
        with open(INDICE, newline='', encoding='utf-8') as fh:
            linhas = list(csv.reader(fh))
        # Sem cabecalho, o artwork-processer.py nao acha a coluna categoria (--excluir).
        if not linhas or linhas[0] != CABECALHO:
            raise SystemExit(f'{INDICE} sem cabecalho; apague-o e rode de novo para recria-lo')
        ja_no_indice = {(l[0], l[5]) for l in linhas[1:] if len(l) >= 6}

    # So consulta a URL de quem ainda falta baixar: numa retomada isso poupa a maior
    # parte das chamadas a API.
    def destino_de(titulo, slug):
        gen = f"gen{pokemons[slug]['gen']}"
        return gen, path.join(DIR_ANIME, gen, nomes[titulo])

    faltam = [a for a in alvos if not path.exists(destino_de(a[0], a[1])[1])]
    enderecos = urls(sessao, [a[0] for a in faltam]) if faltam else {}

    baixadas = existentes = 0
    falhas = []
    with open(INDICE, 'a', newline='', encoding='utf-8') as fh:
        escritor = csv.writer(fh)
        if novo_indice:
            escritor.writerow(CABECALHO)
        for i, (titulo, slug, numero, resto, ext) in enumerate(alvos, 1):
            gen, caminho = destino_de(titulo, slug)
            cat = categoria(resto)
            variacao = titulo[len('File:'):]
            if path.exists(caminho):
                existentes += 1
                if (slug, caminho) not in ja_no_indice:
                    escritor.writerow([slug, gen, cat, variacao, '', caminho])
                    ja_no_indice.add((slug, caminho))
                continue
            if titulo not in enderecos:
                print(f'[{i}/{len(alvos)}] {titulo}: sem imageinfo na API')
                falhas.append(titulo)
                continue
            url, larg, alt = enderecos[titulo]
            try:
                dados = busca(sessao, url, binario=True)
            except (RequestException) as e:
                print(f'[{i}/{len(alvos)}] {titulo}: FALHOU ({e})')
                falhas.append(titulo)
                continue
            if dados is None:
                print(f'[{i}/{len(alvos)}] {titulo}: 404 na imagem')
                falhas.append(titulo)
                continue
            makedirs(path.dirname(caminho), exist_ok=True)
            with open(caminho, 'wb') as img:
                img.write(dados)
            if (slug, caminho) not in ja_no_indice:
                escritor.writerow([slug, gen, cat, variacao, url, caminho])
                ja_no_indice.add((slug, caminho))
            baixadas += 1
            fh.flush()          # retomada depois de Ctrl+C nao perde o indice
            print(f'[{i}/{len(alvos)}] {slug} ({gen}) {cat}: {larg}x{alt} -> '
                  f'{path.basename(caminho)} ({len(dados) // 1024} KB)')
            time.sleep(args.pausa)

    print(f'\nbaixadas {baixadas} | ja existentes {existentes} | falhas {len(falhas)}')
    if falhas:
        print('falhas:', ', '.join(falhas[:20]) + (' ...' if len(falhas) > 20 else ''))
    if fora_do_yaml:
        print('numero fora do pokemons.yaml:', ', '.join(fora_do_yaml[:10]))
    print(f'indice em {INDICE}')


if __name__ == '__main__':
    main()
