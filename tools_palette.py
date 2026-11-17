"""Extrai a paleta dominante de cada sprite e guarda em cache.

POR QUE EM CACHE, E NAO NO __getitem__

O k-means roda sobre ~30 000 pixels por sprite. No DataLoader isso sairia caro e,
pior, seria recalculado a cada epoca para a mesma imagem -- a paleta de um sprite nao
muda. Calcular uma vez e guardar num .npz deixa o treino na mesma velocidade.

ORDEM CANONICA

As K cores saem ordenadas por luminancia, da mais escura para a mais clara. A ordem
precisa ser estavel: se a mesma cor aparecesse ora na posicao 1 ora na 3, cada posicao
deixaria de significar alguma coisa e a rede nao teria o que aprender.
"""
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2

K = 4
CACHE = 'palettes.npz'


def palette(path, k=K, seed=0):
    """K cores dominantes do sprite, ordenadas por luminancia. Devolve (k, 3) em 0..1.

    So os pixels do conteudo entram: o fundo branco domina a imagem em area e, se
    entrasse, roubaria uma das K posicoes para dizer "branco" -- que e justamente a
    cor que nao carrega informacao nenhuma sobre o pokemon.
    """
    a = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    px = a[lum < 0.98].reshape(-1, 3)
    if len(px) < k:  # sprite quase vazio: devolve cinza neutro
        return np.full((k, 3), 0.5, np.float32)

    # minit='++' (k-means++) escolhe sementes espalhadas; com 'random' um sprite de
    # poucas cores costuma devolver clusters vazios e cores NaN.
    centro, _ = kmeans2(px, k, minit='++', seed=seed, iter=25)
    centro = np.nan_to_num(centro, nan=0.5)
    ordem = np.argsort(centro @ np.array([0.299, 0.587, 0.114], np.float32))
    return centro[ordem].astype(np.float32)


def main():
    from pix2pix_data import PATH_INPUT, PATH_REAL, list_pairs
    _, reais = list_pairs(PATH_INPUT, PATH_REAL)
    chaves, cores = [], []
    for i, p in enumerate(reais):
        chaves.append(f'{p.parent.name}/{p.name}')
        cores.append(palette(p))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(reais)}', flush=True)
    np.savez_compressed(CACHE, keys=np.array(chaves), palettes=np.stack(cores))
    print(f'{len(chaves)} paletas em {CACHE}')


if __name__ == '__main__':
    main()
