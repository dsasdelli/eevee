"""Testa se o gerador aguenta um contorno parecido com o que um usuario desenharia.

Compara tres entradas para o MESMO pokemon do conjunto de teste:
  A  contorno do dataset         -- o que a rede viu no treino (controle)
  B  contorno afinado            -- mesmas linhas, erodidas: menos tinta
  C  silhueta limpa              -- so o contorno externo do sprite, sem interior

C e a aproximacao do desenho de um usuario: traco fino, sem a textura interna que o
adaptiveThreshold de sprite-contours.py produz.
"""
import sys
from pathlib import Path

from scipy import ndimage
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, 'D:/Projetos/Github/eevee')
from pix2pix_data import PATH_INPUT, PATH_REAL, fixed_split, list_pairs
from pix2pix_model import Generator, up_mode_from_state

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    'sprites-processed-countours_torch_training_checkpoints/ckpt-050000.pt'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = Path('C:/Users/Lomir/AppData/Local/Temp/claude/D--Projetos-Github-eevee/'
           'f2186ad1-96f6-4c2d-8f6c-9f5c43f874a6/scratchpad/domain_gap_resize.png')


def thin(contour_arr):
    """Afina o traco: erode a mascara escura, devolvendo linhas mais finas."""
    dark = contour_arr < 128
    eroded = ndimage.binary_erosion(dark, np.ones((3, 3), bool))
    return np.where(eroded, 0, 255).astype(np.uint8)


def silhouette(sprite_arr):
    """So o contorno externo do sprite, traco fino -- sem nada do interior."""
    gray = sprite_arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    mask = gray < 250
    # Preenche buracos para ficar com a silhueta cheia; a borda dela e o gradiente
    # morfologico (dilatacao menos erosao), que da um traco de 1-2 pixels.
    filled = ndimage.binary_fill_holes(mask)
    k = np.ones((3, 3), bool)
    borda = ndimage.binary_dilation(filled, k) & ~ndimage.binary_erosion(filled, k)
    return np.where(borda, 0, 255).astype(np.uint8)


def to_t(arr2d, device):
    t = torch.from_numpy(arr2d).float().div_(127.5).sub_(1.0)
    return t.unsqueeze(0).unsqueeze(0).to(device)


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(CKPT, map_location=device)
    gen = Generator(img_size=256, up_mode=up_mode_from_state(state)).to(device)
    gen.load_state_dict(state['generator'])
    gen.train()  # modo padrao do pix2pix_predict.py

    inputs, reals = list_pairs(PATH_INPUT, PATH_REAL)
    _, test_idx = fixed_split(len(inputs))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(N, 7, figsize=(19, 2.8 * N))
    titles = ['alvo real', 'A entrada dataset', 'A saida',
              'B entrada afinada', 'B saida', 'C silhueta limpa', 'C saida']

    for row in range(N):
        i = test_idx[row * 7]
        cont = np.array(Image.open(inputs[i]).convert('L').resize((256, 256), Image.NEAREST))
        spr = np.array(Image.open(reals[i]).convert('RGB').resize((256, 256), Image.NEAREST))

        variants = [cont, thin(cont), silhouette(spr)]
        cells = [spr]
        for v in variants:
            pred = gen(to_t(v, device))[0].cpu().mul(0.5).add(0.5).clamp(0, 1)
            cells.append(v)
            cells.append(pred.permute(1, 2, 0).numpy())

        inks = [float((v < 128).mean()) for v in variants]
        for col, (ax, img) in enumerate(zip(axes[row], cells)):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=255 if img.ndim == 2 else None)
            ax.axis('off')
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        axes[row][0].set_ylabel(Path(inputs[i]).stem, fontsize=8)
        print(f'{Path(inputs[i]).stem:16} tinta: A {inks[0]:.3f} | B {inks[1]:.3f} | C {inks[2]:.3f}')

    fig.suptitle(f'Lacuna de dominio -- {Path(CKPT).name} (passo {state["step"]})', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=85)
    print(f'\nsalvo: {OUT}')


main()
