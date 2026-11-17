"""Mede L1, saturacao e a separacao luminancia/crominancia no conjunto de teste.

Roda o MESMO checkpoint nos dois modos do gerador (train/eval), porque o
pix2pix amostra com dropout e BatchNorm em modo treino e os numeros diferem.
Compara varios checkpoints com uma convencao unica, para que a comparacao
entre rodadas seja valida mesmo que nao reproduza exatamente uma medicao antiga.
"""
import argparse
from pathlib import Path

import torch

from pix2pix_data import PATH_INPUT, PATH_REAL, PokemonPairs, fixed_split, list_pairs
from pix2pix_model import Generator


def saturation(rgb01):
    """Saturacao HSV: (max - min) / max, por pixel. Media sobre a imagem inteira.

    Inclui o fundo branco, onde max == min e a saturacao e 0. Isso puxa a media
    para baixo, mas da mesma forma nos dois lados da comparacao -- e o fundo e
    identico entre predicao e alvo.
    """
    mx = rgb01.amax(dim=1)
    mn = rgb01.amin(dim=1)
    return ((mx - mn) / mx.clamp(min=1e-6)).mean(dim=(1, 2))


def luma_chroma(rgb01):
    """Separa em Y (Rec. 601) e os dois canais de crominancia Cb/Cr."""
    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, (b - y), (r - y)


@torch.no_grad()
def measure(ckpt_path, mode, device, img_size, batch_size):
    input_paths, real_paths = list_pairs(PATH_INPUT, PATH_REAL)
    _, test_idx = fixed_split(len(input_paths))
    ds = PokemonPairs(input_paths, real_paths, test_idx, train=False, img_size=img_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    state = torch.load(ckpt_path, map_location=device)
    gen = Generator(img_size=img_size).to(device)
    gen.load_state_dict(state['generator'])
    gen.train() if mode == 'train' else gen.eval()

    acc = dict(l1=0.0, sat_pred=0.0, sat_real=0.0, l1_luma=0.0, l1_chroma=0.0)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = gen(x)

        # L1 na mesma faixa [-1, 1] que pix2pix_predict.py --eval, para que o
        # numero seja diretamente comparavel ao 0,180 ja registrado.
        acc['l1'] += torch.abs(y - pred).mean(dim=(1, 2, 3)).sum().item()

        p01 = pred.mul(0.5).add(0.5).clamp(0, 1)
        y01 = y.mul(0.5).add(0.5).clamp(0, 1)
        acc['sat_pred'] += saturation(p01).sum().item()
        acc['sat_real'] += saturation(y01).sum().item()

        py, pcb, pcr = luma_chroma(p01)
        yy, ycb, ycr = luma_chroma(y01)
        acc['l1_luma'] += (py - yy).abs().mean(dim=(1, 2)).sum().item()
        acc['l1_chroma'] += (((pcb - ycb).abs() + (pcr - ycr).abs()) / 2).mean(dim=(1, 2)).sum().item()
        n += x.size(0)

    return {k: v / n for k, v in acc.items()}, n, state['step']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    rows = []
    for ckpt in args.checkpoints:
        for mode in ('train', 'eval'):
            m, n, step = measure(Path(ckpt), mode, device, args.img_size, args.batch_size)
            rows.append((Path(ckpt).name, step, mode, m, n))
            print(f'{Path(ckpt).name} passo {step} modo {mode}: '
                  f"L1 {m['l1']:.4f} | sat pred {m['sat_pred']:.4f} "
                  f"alvo {m['sat_real']:.4f} | L1 luma {m['l1_luma']:.4f} "
                  f"croma {m['l1_chroma']:.4f}", flush=True)

    print(f"\n{'checkpoint':34} {'modo':6} {'L1':>7} {'sat pred':>9} {'sat alvo':>9} {'luma':>7} {'croma':>7}")
    for name, step, mode, m, n in rows:
        print(f"{name:34} {mode:6} {m['l1']:7.4f} {m['sat_pred']:9.4f} "
              f"{m['sat_real']:9.4f} {m['l1_luma']:7.4f} {m['l1_chroma']:7.4f}")
    print(f'\n{rows[0][4]} pares de teste, split seed 1337')


if __name__ == '__main__':
    main()
