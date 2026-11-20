"""Enquadra as artes baixadas: fundo branco, recortadas no conteudo, 512x512.

E o equivalente do `sprite-processer.py` para a arte oficial, e segue os mesmos tres
passos, pelos mesmos motivos: transparencia vira branco (senao um PNG com alfa leria
como preto), recorte na caixa do conteudo (para o bicho ocupar a moldura em vez de
flutuar nela) e escala pelo MAIOR lado, centralizada, preservando a proporcao.

DUAS DIFERENCAS EM RELACAO AO DE SPRITES

Tamanho 512, nao 256. As artes tem lado maior de 731 px em media (faixa 270 a 1000),
entao 256 jogaria fora o detalhe que motivou baixa-las; 512 ainda e reducao para a
maioria, e e a resolucao em que o Stable Diffusion gera.

Reamostragem LANCZOS em vez de BOX. BOX e media simples de celula, boa para reduzir
pixel art sem inventar tom intermediario. Arte oficial tem gradiente e linha fina
antisserrilhada -- ali LANCZOS preserva nitidez de borda que a media borraria. E para
as poucas artes menores que 512, que precisam ser AMPLIADAS, BOX degrada bem mais.

As subpastas de geracao sao descobertas no disco, nao em GENERATIONS: a pasta de
artwork e organizada pela geracao do POKEMON, nao pela do jogo.

Processa por GRUPO (sugimori, outros, anime; ver helper.py), de artwork-<grupo>/ para
artwork-<grupo>-processed/. O anime tem filtro proprio por padrao (PADROES): arte com
conteudo menor que 450 px e a categoria 'Outro' (Mystery Dungeon, Smash) ficam de fora.

    python artwork-processer.py                       os tres grupos
    python artwork-processer.py --grupo anime
"""
import argparse
import csv
from os import listdir, makedirs, path

from PIL import Image, ImageOps

from helper import ARTWORK_SIZE, GRUPOS, dir_artes, dir_processadas

SUFIXOS = {'.png', '.jpg', '.jpeg'}
# (min_lado, categorias excluidas) de cada grupo, quando a linha de comando nao diz.
PADROES = {'sugimori': (0, ''), 'outros': (0, ''), 'anime': (450, 'Outro')}


def alfa_para_branco(im):
    """Achata a transparencia sobre branco. Metade das artes e RGBA."""
    if im.mode not in ('RGBA', 'LA', 'P'):
        return im.convert('RGB')
    im = im.convert('RGBA')
    fundo = Image.new('RGBA', im.size, 'WHITE')
    fundo.paste(im, (0, 0), im)
    return fundo.convert('RGB')


def recorta(im):
    """Corta na caixa do conteudo. getbbox() acha o retangulo do que NAO e preto,
    entao a imagem e invertida primeiro -- aqui o conteudo e escuro sobre claro."""
    caixa = ImageOps.invert(im).getbbox()
    return im.crop(caixa) if caixa else im


def enquadra(im, tamanho=ARTWORK_SIZE):
    folha = Image.new('RGB', (tamanho, tamanho), 'WHITE')
    escala = tamanho / max(im.size)
    novo = im.resize((max(1, round(im.width * escala)), max(1, round(im.height * escala))),
                     resample=Image.LANCZOS)
    folha.paste(novo, ((tamanho - novo.width) // 2, (tamanho - novo.height) // 2))
    return folha


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grupo', choices=GRUPOS, action='append',
                   help='grupo a processar; repita para mais de um (padrao: todos)')
    p.add_argument('--min-lado', type=int, default=None,
                   help='descarta a arte cujo conteudo recortado tem lado maior abaixo '
                        'disto. Ampliar 275 px para 512 borra, e resolucao misturada e a '
                        'diluicao que estragou a LoRA das nove geracoes')
    p.add_argument('--excluir', default=None,
                   help='categorias do indice.csv do grupo a descartar, separadas por '
                        'virgula (ex. Outro)')
    return p.parse_args()


def main():
    args = parse_args()
    for grupo in args.grupo or GRUPOS:
        min_lado, excluir = PADROES[grupo]
        if args.min_lado is not None:
            min_lado = args.min_lado
        if args.excluir is not None:
            excluir = args.excluir
        print(f'== {grupo} (min-lado {min_lado}, excluir {excluir or "nada"})')
        processa(dir_artes(grupo), dir_processadas(grupo), min_lado, excluir)


def processa(origem_raiz, destino_raiz, min_lado, excluir):
    excluidos = set()
    if excluir:
        cats = {c.strip() for c in excluir.split(',') if c.strip()}
        with open(path.join(origem_raiz, 'indice.csv'), newline='', encoding='utf-8') as fh:
            excluidos = {path.basename(l['arquivo'].replace(chr(92), '/'))
                         for l in csv.DictReader(fh) if l['categoria'] in cats}
        print(f'excluindo {len(excluidos)} artes das categorias {sorted(cats)}')

    geracoes = sorted(d for d in listdir(origem_raiz)
                      if path.isdir(path.join(origem_raiz, d)))
    total = vazias = pequenas = fora = 0
    for gen in geracoes:
        origem = path.join(origem_raiz, gen)
        destino = path.join(destino_raiz, gen)
        makedirs(destino, exist_ok=True)
        arquivos = [f for f in listdir(origem) if path.splitext(f)[1].lower() in SUFIXOS]
        feitas = 0
        for f in arquivos:
            if f in excluidos:
                fora += 1
                continue
            im = alfa_para_branco(Image.open(path.join(origem, f)))
            im = recorta(im)
            # Arte inteiramente branca nao tem o que recortar nem enquadrar, e viraria
            # uma folha em branco no conjunto de treino.
            if min(im.size) < 2:
                print(f'  {gen}/{f}: imagem vazia, pulada')
                vazias += 1
                continue
            if max(im.size) < min_lado:
                pequenas += 1
                continue
            # Sempre .png na saida: o recorte e o enquadramento nao devem passar por
            # recompressao JPEG, que acrescentaria artefato de bloco justamente nas
            # bordas duras que o extrator de contorno vai ler depois.
            saida = path.splitext(f)[0] + '.png'
            enquadra(im).save(path.join(destino, saida))
            total += 1
            feitas += 1
        print(f'{gen}: {feitas} de {len(arquivos)} artes -> {destino}')
    print(f'{chr(10)}{total} processadas em {ARTWORK_SIZE}x{ARTWORK_SIZE} | vazias {vazias} | '
          f'menores que {min_lado} px {pequenas} | excluidas por categoria {fora}')


if __name__ == '__main__':
    main()
