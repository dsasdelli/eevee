"""Folha 23: os 10 desenhos humanos em lora-art-tudo (4000) contra lora-art-tudo-aug.

Mesmo pipeline da varre_ckpt.py (ControlNet scribble, forca 0,7, 28 passos, cfg 7,5,
semente 42), saida CRUA sem tirar_fundo, porque o fundo e um dos criterios.
Imprime tambem a fracao de borda nao-branca por coluna (alarme grosseiro de fundo).
"""
import sys

sys.path.insert(0, r'D:\Projetos\Github\eevee')

from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw

from lora_train import LEGENDA
from pix2pix_predict import fit_sketch

RAIZ = Path(r'D:\Projetos\Github\eevee')
SAIDA = RAIZ / 'predictions/difusao/23-aug-contra-sem-aug.png'
COLUNAS = [('art-tudo 4000', 'lora-art-tudo', 4000)] + \
          [(f'aug {p}', 'lora-art-tudo-aug', p) for p in (4000, 8000, 12000, 16000)]
NEG = 'photo, realistic, text, watermark, logo, blurry, grayscale, human, background scenery'


def ctrl(im):
    a = np.array(im.convert('L'), dtype=np.uint8)
    return Image.fromarray(255 - a).convert('RGB').resize((512, 512), Image.NEAREST)


def borda_suja(im):
    a = np.asarray(im.convert('RGB'))
    b = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3),
                        a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    return float((b.min(1) < 235).mean())


def main():
    cn = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_scribble',
                                         dtype=torch.float16, variant='fp16')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5', controlnet=cn, variant='fp16',
        dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)

    alvos = list(range(1, 11))
    esb = {n: fit_sketch(Image.open(RAIZ / f'sprites-human-drawn/pokemon{n}.bmp'), 512)
           for n in alvos}
    colunas = []
    for rot, pasta, passo in COLUNAS:
        pipe.unload_lora_weights()
        pipe.load_lora_weights(str(RAIZ / pasta), weight_name=f'lora-step{passo:06d}.safetensors')
        saidas = [pipe(LEGENDA, image=ctrl(esb[n]), negative_prompt=NEG, num_inference_steps=28,
                       guidance_scale=7.5, controlnet_conditioning_scale=0.7,
                       generator=torch.Generator('cuda').manual_seed(42)).images[0]
                  for n in alvos]
        sujas = [borda_suja(s) for s in saidas]
        print(f'{rot:14s} borda nao-branca media {np.mean(sujas):.2f}  '
              f'com fundo (>30%): {sum(x > 0.3 for x in sujas)}/10', flush=True)
        colunas.append((rot, saidas))

    W, H = 256, 24
    cols = len(colunas) + 1
    folha = Image.new('RGB', (W * cols + 6 * (cols + 1), H + (W + 6) * len(alvos) + 6),
                      (215, 215, 215))
    d = ImageDraw.Draw(folha)
    d.text((10, 6), 'desenho', fill=(0, 0, 0))
    for c, (rot, _) in enumerate(colunas, start=1):
        d.text((6 + c * (W + 6) + 10, 6), rot, fill=(0, 0, 0))
    for r, n in enumerate(alvos):
        folha.paste(esb[n].convert('RGB').resize((W, W)), (6, H + r * (W + 6)))
        for c, (_, s) in enumerate(colunas, start=1):
            folha.paste(s[r].resize((W, W), Image.LANCZOS), (6 + c * (W + 6), H + r * (W + 6)))
    folha.save(SAIDA)
    print('->', SAIDA)


if __name__ == '__main__':
    main()
