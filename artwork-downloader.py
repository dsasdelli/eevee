"""Baixa a arte oficial de todos os Pokemon de pokemondb.net, por geracao.

POR QUE NAO DA PARA FAZER COMO O sprite-downloader.py

O `sprite-downloader.py` monta a URL direto a partir do nome e da geracao do jogo,
porque sprite tem endereco previsivel: um por Pokemon por jogo. Artwork nao tem. Cada
Pokemon pode ter de 1 a 30 imagens (medido: baxcalibur 1, tyranitar 5, charizard 10,
pikachu 30), em quatro categorias com padroes de URL diferentes:

    artwork/large/charizard.jpg              Sugimori, a arte oficial moderna
    artwork/original/charizard-gen1.jpg      Early Sugimori, dos anos 90
    artwork/vector/large/charizard.png       Global Link, vetor chapado
    artwork/alt/charizard-gigantamax-alt.jpg marketing

Nao ha como enumerar isso sem ler a pagina. Dai este programa existir.

DOIS GRUPOS, DUAS PASTAS

Cada arte vai para o grupo da sua categoria (ver helper.py): categoria com "Sugimori"
(inclusive a Early) vai para artwork-sugimori/, o resto (Global Link, Other) para
artwork-outros/. Cada pasta tem o proprio indice.csv.

A GERACAO E A DO POKEMON, NAO A DO JOGO

O sprite-downloader percorre GENERATIONS porque o mesmo bicho tem um sprite por jogo.
Artwork nao e por jogo, entao a pasta sai do campo `gen` do proprio pokemons.yaml.
Consequencia aceita: o Gigantamax do Charizard, que e conteudo de gen8, fica em gen1
junto do resto da arte dele -- a alternativa espalharia as variacoes de um mesmo
Pokemon por varias pastas.

    python artwork-downloader.py                     tudo
    python artwork-downloader.py --limite 10         so os dez primeiros
    python artwork-downloader.py --pokemon charizard,pikachu
"""
import argparse
import csv
import re
import time
from os import makedirs, path

import requests
import yaml

from helper import dir_artes

PAGINA = 'https://pokemondb.net/artwork/{}'
GRUPOS_PDB = ('sugimori', 'outros')
CABECALHO = ['pokemon', 'geracao', 'categoria', 'variacao', 'url', 'arquivo']


def grupo_da_categoria(categoria):
    """'Early Sugimori artwork - Gold/Silver' -> 'sugimori'; 'Global Link artwork' -> 'outros'."""
    return 'sugimori' if 'Sugimori' in categoria else 'outros'


def indice(grupo):
    return f'{dir_artes(grupo)}/indice.csv'
# Identificar-se e o minimo ao varrer mais de mil paginas de um site de terceiros.
CABECALHOS = {'User-Agent': 'eevee-artwork-downloader/1.0 (projeto pessoal de dataset)'}
PAUSA = 0.4          # segundos entre requisicoes
TENTATIVAS = 3
ESPERA_ERRO = 2.0    # segundos antes de tentar de novo

# Cada arte e um bloco grid-col com a categoria em <strong>, o rotulo da variacao em
# <span class="text-muted">, a versao GRANDE no href e a miniatura no src. Regex
# resolve aqui e dispensa o BeautifulSoup, que nao esta instalado -- mesma politica de
# nao acrescentar dependencia que nao resolve problema.
BLOCO = re.compile(r'<div class="grid-col[^"]*">(.*?)</div>', re.S)
CATEGORIA = re.compile(r'<strong[^>]*>(.*?)</strong>', re.S)
VARIACAO = re.compile(r'<span class="text-muted">(.*?)</span>', re.S)
GRANDE = re.compile(r'<a href="(https://img\.pokemondb\.net/artwork/[^"]+)"')
MINIATURA = re.compile(r'<img src="(https://img\.pokemondb\.net/artwork/[^"]+)"')
TAGS = re.compile(r'<[^>]+>')


def limpa(html):
    return TAGS.sub('', html).replace('&amp;', '&').strip()


def nome_arquivo(url):
    """URL da imagem -> nome de arquivo, com a categoria no prefixo.

    O prefixo nao e enfeite: sem ele, `vector/tyranitar-mega.png` e
    `tyranitar-mega.jpg` so nao colidem por causa da extensao, o que e fragil demais.
    Com prefixo, a origem fica legivel no nome e da para filtrar por estilo com glob.

    O segmento `large` e descartado por ser tamanho, nao categoria.
    """
    caminho = url.split('/artwork/', 1)[-1]
    partes = [p for p in caminho.split('/') if p != 'large']
    return '-'.join(partes)


def artes_da_pagina(html):
    """Extrai [(categoria, variacao, url)] de uma pagina de artwork."""
    achados = []
    for bloco in BLOCO.findall(html):
        if 'img.pokemondb.net/artwork' not in bloco:
            continue
        # Prefere a versao grande; a propria pagina avisa que so ALGUMAS tem (medido:
        # charizard 4 de 10 sem, pikachu 5 de 30).
        alvo = GRANDE.search(bloco) or MINIATURA.search(bloco)
        if not alvo:
            continue
        cat = CATEGORIA.search(bloco)
        var = VARIACAO.search(bloco)
        achados.append((limpa(cat.group(1)) if cat else '',
                        limpa(var.group(1)) if var else '',
                        alvo.group(1)))
    return achados


def busca(sessao, url, binario=False):
    """GET com tentativas. Devolve None em 404, levanta no resto depois de tentar."""
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            r = sessao.get(url, headers=CABECALHOS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binario else r.text
        except requests.RequestException as e:
            if tentativa == TENTATIVAS:
                raise
            print(f'    tentativa {tentativa} falhou ({type(e).__name__}), repetindo')
            time.sleep(ESPERA_ERRO * tentativa)
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--limite', type=int, help='processa so os N primeiros Pokemon')
    p.add_argument('--pokemon', help='lista separada por virgula, para testar casos')
    p.add_argument('--pausa', type=float, default=PAUSA, help='segundos entre requisicoes')
    return p.parse_args()


def main():
    args = parse_args()
    with open('pokemons.yaml', encoding='utf-8') as f:
        pokemons = yaml.safe_load(f)

    if args.pokemon:
        escolhidos = [n.strip() for n in args.pokemon.split(',') if n.strip()]
        faltam = [n for n in escolhidos if n not in pokemons]
        if faltam:
            raise SystemExit(f'nao estao em pokemons.yaml: {faltam}')
    else:
        escolhidos = list(pokemons)
        if args.limite:
            escolhidos = escolhidos[:args.limite]

    # Abre o indice em modo append e so escreve o cabecalho se o arquivo e novo: o
    # programa e retomavel, e truncar aqui apagaria o registro das rodadas anteriores.
    #
    # Mas append sozinho acumula duplicata: numa rodada de retomada, cada arte ja
    # baixada seria registrada de novo. Por isso as linhas ja presentes sao carregadas
    # antes e usadas para nao reescrever -- sem isso o indice ficava inutil depois de
    # poucas rodadas (medido: 84 linhas para 69 imagens).
    ja_no_indice = set()
    arquivos, escritores = {}, {}
    for g in GRUPOS_PDB:
        makedirs(dir_artes(g), exist_ok=True)
        novo = not path.exists(indice(g))
        if not novo:
            with open(indice(g), newline='', encoding='utf-8') as fh:
                for linha in csv.reader(fh):
                    if len(linha) >= 6:
                        ja_no_indice.add((linha[0], linha[5]))
        arquivos[g] = open(indice(g), 'a', newline='', encoding='utf-8')
        escritores[g] = csv.writer(arquivos[g])
        if novo:
            escritores[g].writerow(CABECALHO)
    sessao = requests.Session()
    baixadas = existentes = sem_arte = 0
    falhas = []

    try:
        for i, nome in enumerate(escolhidos, 1):
            gen = f"gen{pokemons[nome]['gen']}"

            try:
                html = busca(sessao, PAGINA.format(nome))
            except requests.RequestException as e:
                print(f'[{i}/{len(escolhidos)}] {nome}: FALHOU a pagina ({e})')
                falhas.append(nome)
                continue
            if html is None:
                print(f'[{i}/{len(escolhidos)}] {nome}: sem pagina de artwork (404)')
                sem_arte += 1
                continue

            artes = artes_da_pagina(html)
            if not artes:
                print(f'[{i}/{len(escolhidos)}] {nome}: pagina sem arte nenhuma')
                sem_arte += 1
                continue

            novas = 0
            for categoria, variacao, url in artes:
                grupo = grupo_da_categoria(categoria)
                destino = path.join(dir_artes(grupo), gen)
                makedirs(destino, exist_ok=True)
                arquivo = nome_arquivo(url)
                caminho = path.join(destino, arquivo)
                if (nome, caminho) not in ja_no_indice:
                    escritores[grupo].writerow([nome, gen, categoria, variacao, url, caminho])
                    ja_no_indice.add((nome, caminho))
                if path.exists(caminho):
                    existentes += 1
                    continue
                try:
                    dados = busca(sessao, url, binario=True)
                except requests.RequestException as e:
                    print(f'    {arquivo}: FALHOU ({e})')
                    falhas.append(f'{nome}/{arquivo}')
                    continue
                if dados is None:
                    print(f'    {arquivo}: 404 na imagem')
                    falhas.append(f'{nome}/{arquivo}')
                    continue
                with open(caminho, 'wb') as img:
                    img.write(dados)
                baixadas += 1
                novas += 1
                time.sleep(args.pausa)

            print(f'[{i}/{len(escolhidos)}] {nome} ({gen}): {len(artes)} artes, '
                  f'{novas} baixadas, {len(artes) - novas} ja existentes')
            for fh in arquivos.values():
                fh.flush()      # retomada depois de Ctrl+C nao perde o indice
            time.sleep(args.pausa)
    finally:
        for fh in arquivos.values():
            fh.close()

    print(f'\nbaixadas {baixadas} | ja existentes {existentes} | '
          f'sem arte {sem_arte} | falhas {len(falhas)}')
    if falhas:
        print('falhas:', ', '.join(falhas[:20]) + (' ...' if len(falhas) > 20 else ''))
    print('indices em', ', '.join(indice(g) for g in GRUPOS_PDB))


if __name__ == '__main__':
    main()
