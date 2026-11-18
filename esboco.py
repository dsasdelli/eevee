"""Utilitarios de esboco e de dataset compartilhados pelo app e pelas LoRAs.

Vieram do pix2pix (aposentado em 17/11/2026) e foram copiados sem mudar a logica:
o enquadramento do esboco e o split fixo dos sprites continuam exatamente os mesmos,
porque as LoRAs de sprite foram treinadas sobre esse split.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

PATH_INPUT = 'sprites-processed-countours'  # o contorno de cada sprite
PATH_REAL = 'sprites-processed'             # o sprite colorido
TEST_SIZE = 1000
SPLIT_SEED = 1337


def fit_sketch(im, size):
    """Enquadra um esboco do jeito que sprite-processer.py preparou o dataset.

    O ControlNet guia proporcionalmente ao quanto do quadro o traco cobre: um desenho
    feito num canto da tela, pequeno, sai mal obedecido. Por isso o esboco e posto no
    mesmo enquadramento do dataset -- fundo branco, recortado no conteudo, escalado
    para encostar nas bordas e centralizado.

    Os tres passos reproduzem sprite-processer.py:
      1. transparencia vira branco (um PNG com fundo transparente leria como preto);
      2. recorta na bounding box do conteudo;
      3. escala pelo maior lado e centraliza, preservando a proporcao.
    """
    # getbbox() acha o retangulo do que NAO e preto; como o conteudo aqui e escuro
    # sobre claro, inverte-se a imagem primeiro para achar o retangulo do desenho.
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        fundo = Image.new('RGBA', im.size, 'WHITE')
        fundo.paste(im, (0, 0), im)  # o terceiro argumento usa o alfa como mascara
        im = fundo
    im = im.convert('L')

    bbox = ImageOps.invert(im).getbbox()
    if bbox:  # None quando a imagem e inteiramente branca
        im = im.crop(bbox)

    # Escala pelo MAIOR lado para a imagem caber inteira no quadrado, sem distorcer.
    escala = size / max(im.size)
    novo = Image.new('L', (size, size), 255)  # 255 = branco, o fundo do dataset
    red = im.resize((max(1, int(im.width * escala)), max(1, int(im.height * escala))),
                    Image.BOX)
    novo.paste(red, ((size - red.width) // 2, (size - red.height) // 2))
    return novo


def list_pairs(path_input=PATH_INPUT, path_real=PATH_REAL):
    """Lista ordenada dos dois lados do par (contorno, sprite), com o pareamento conferido.

    O pareamento depende de os dois diretorios terem a mesma estrutura
    (<geracao>/<pokemon>.png) e de sorted() produzir a mesma ordem nos dois. Se um
    arquivo faltar de um lado, todos os pares a partir dali ficam deslocados sem erro
    nenhum aparecer. Dai a conferencia explicita abaixo.
    """
    input_paths = sorted(Path(path_input).glob('*/*.png'))
    real_paths = sorted(Path(path_real).glob('*/*.png'))

    if not input_paths:
        raise FileNotFoundError(f'nenhum png encontrado em {path_input}/*/*.png')
    if len(input_paths) != len(real_paths):
        raise ValueError(f'{len(input_paths)} entradas para {len(real_paths)} alvos')
    for p_in, p_re in zip(input_paths, real_paths):
        # Compara geracao e nome do arquivo; o diretorio raiz e que difere.
        if (p_in.parent.name, p_in.name) != (p_re.parent.name, p_re.name):
            raise ValueError(f'par desalinhado: {p_in} != {p_re}')

    return input_paths, real_paths


def fixed_split(n, test_size=TEST_SIZE, seed=SPLIT_SEED):
    """Split fixo e reproduzivel. Devolve (train_idx, test_idx), disjuntos.

    default_rng(seed) cria um gerador proprio, isolado do estado global do numpy:
    o resultado nao muda se outro trecho do programa sortear alguma coisa antes.

    ATENCAO -- limitacao conhecida: o split e por IMAGEM, nao por pokemon. O mesmo
    pokemon aparece em ate 9 geracoes com arte parecida, entao 91,7% das imagens de
    teste tem um "primo" no treino. Corrigir exige agrupar por p.name antes de permutar.
    """
    perm = np.random.default_rng(seed).permutation(n)
    return perm[test_size:], perm[:test_size]
