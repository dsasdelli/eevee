"""Mede, com entrada de SILHUETA LIMPA, se a saida tem estrutura ou e um borrao.

Duas medidas, sobre os pixels de dentro da silhueta:
  saturacao   -- a cor e viva ou lavada
  estrutura   -- gradiente espacial medio da luminancia. Um sprite tem bordas
                 internas (membros, olhos, sombreado); um borrao liso nao tem.
O alvo real entra como referencia do que esses numeros valem num sprite de verdade.
"""
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, 'D:/Projetos/Github/eevee')
from pix2pix_data import PATH_INPUT, PATH_REAL, fixed_split, list_pairs
from pix2pix_model import Generator

N = 120

def silhouette(y):
    lum = np.array(y.convert('L'), np.uint8)
    filled = ndimage.binary_fill_holes(lum < 250)
    k = np.ones((3, 3), bool)
    borda = ndimage.binary_dilation(filled, k) & ~ndimage.binary_erosion(filled, k)
    return np.where(borda, 0, 255).astype(np.uint8), filled

def medidas(rgb01, dentro):
    mx, mn = rgb01.max(0), rgb01.min(0)
    sat = ((mx - mn) / np.clip(mx, 1e-6, None))[dentro].mean()
    lum = rgb01[0]*0.299 + rgb01[1]*0.587 + rgb01[2]*0.114
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)[dentro].mean()
    return float(sat), float(grad)

@torch.no_grad()
def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ip, rp = list_pairs(PATH_INPUT, PATH_REAL)
    _, test_idx = fixed_split(len(ip))
    idxs = test_idx[:N]

    sils, dentros, alvos = [], [], []
    for i in idxs:
        y = Image.open(rp[i]).convert('RGB').resize((256, 256), Image.NEAREST)
        s, d = silhouette(y)
        sils.append(s); dentros.append(d)
        alvos.append(np.asarray(y, np.float32).transpose(2, 0, 1) / 255)

    ref = np.array([medidas(a, d) for a, d in zip(alvos, dentros)])
    print(f'{N} pokemon do conjunto de teste, entrada = silhueta limpa\n')
    print(f"{'modelo':34} {'saturacao':>10} {'estrutura':>10}")
    print(f"{'alvo real (referencia)':34} {ref[:,0].mean():10.4f} {ref[:,1].mean():10.4f}")

    for nome, ck in [('antes  lam15-step050000', 'lam15-step050000.pt'),
                     ('depois ckpt-040000 (aug traco)', 'ckpt-040000.pt')]:
        st = torch.load(f'sprites-processed-countours_torch_training_checkpoints/{ck}', map_location=dev)
        g = Generator(img_size=256).to(dev); g.load_state_dict(st['generator']); g.train()
        out = []
        for s, d in zip(sils, dentros):
            t = torch.from_numpy(s).float().div_(127.5).sub_(1.0)[None, None].to(dev)
            p = g(t)[0].cpu().mul(0.5).add(0.5).clamp(0, 1).numpy()
            out.append(medidas(p, d))
        o = np.array(out)
        print(f'{nome:34} {o[:,0].mean():10.4f} {o[:,1].mean():10.4f}')

main()
