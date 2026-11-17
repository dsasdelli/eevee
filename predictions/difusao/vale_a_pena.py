"""A LoRA valeu a pena, ou prompt bom no SD puro basta?

A comparacao anterior era enviesada: o 'SD cru' recebia o prompt da LoRA, com o token
`pkmnsprite` (que nao significa nada para o modelo base) e 'pixel art' (que faz o SD
desenhar uma grade literal por cima). Aqui o SD puro recebe um prompt feito PARA ELE,
sem 'pixel art', e a aparencia de sprite vem da pixelizacao -- que e pos-processamento
e nao depende de treino nenhum.

Se o SD puro + prompt bom + pixelizacao empatar com a LoRA, as 4,3 h de treino nao se
pagaram e e melhor saber disso.
"""
import sys

sys.path.insert(0, r'D:\Projetos\Github\eevee')

from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw

from lora_predict import NEGATIVO, PROMPT, para_controlnet, pixelizar, tirar_fundo
from pix2pix_predict import fit_sketch

S = Path(r'C:\Users\Lomir\AppData\Local\Temp\claude\D--Projetos-Github-eevee\6116fa34-1652-486d-b8ec-f1811ba28098\scratchpad')
ALVOS = [1, 3, 5, 7]
SEMENTES = [42, 101, 777]

# Prompt pensado para o SD base: sem 'pixel art' (que vira grade) e sem o token da
# LoRA. Pede o que se quer de verdade -- cor chapada, contorno marcado, ficha de bicho.
PROMPT_JUSTO = ('official pokemon creature artwork, original monster design, '
                'flat vibrant colors, clean cel shading, bold dark outline, '
                'plain white background, full body, centered, game asset')


def nitidez(im):
    a = np.asarray(im.convert('RGB'), np.float32) / 255
    dv = np.abs(a[1:, :] - a[:-1, :]).mean()
    dh = np.abs(a[:, 1:] - a[:, :-1]).mean()
    return float((dv + dh) / 2)


def satur_bicho(im):
    a = np.asarray(im.convert('RGB'), np.float32) / 255
    mx, mn = a.max(axis=2), a.min(axis=2)
    corpo = mn < 0.85
    return float((mx - mn)[corpo].mean()) if corpo.any() else 0.0


def main():
    cn = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_scribble',
                                         dtype=torch.float16, variant='fp16')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5', controlnet=cn, variant='fp16',
        dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)

    esb = {n: fit_sketch(Image.open(rf'D:\Projetos\Github\eevee\sprites-human-drawn\pokemon{n}.bmp'), 512)
           for n in ALVOS}
    ctrl = {n: para_controlnet(esb[n]) for n in ALVOS}

    config = [
        ('SD puro, prompt justo', None, PROMPT_JUSTO),
        ('SD puro, prompt da LoRA', None, PROMPT),
        ('LoRA 8000', 'lora-step008000.safetensors', PROMPT),
    ]
    resultado, mostrar = {}, {}
    for rot, lora, prompt in config:
        pipe.unload_lora_weights()
        if lora:
            pipe.load_lora_weights(r'D:\Projetos\Github\eevee\lora-pkmnsprite', weight_name=lora)
        metr, ims = [], []
        for n in ALVOS:
            for sem in SEMENTES:
                img = pipe(prompt, image=ctrl[n], negative_prompt=NEGATIVO,
                           num_inference_steps=28, guidance_scale=7.5,
                           controlnet_conditioning_scale=0.7,
                           generator=torch.Generator('cuda').manual_seed(sem)).images[0]
                lim = tirar_fundo(img)
                pix = pixelizar(lim, 96, 24)
                metr.append((nitidez(lim), satur_bicho(lim), nitidez(pix)))
                if sem == 42:
                    ims.append(pix)
        resultado[rot] = np.array(metr)
        mostrar[rot] = ims
        print(rot, 'ok', flush=True)

    # Referencia: os sprites reais, no mesmo tamanho de 96 px.
    from pix2pix_data import fixed_split, list_pairs
    _, reais = list_pairs()
    _, te = fixed_split(len(reais))
    alvo = [Image.open(reais[i]).resize((96, 96), Image.BOX) for i in te[:60]]
    print(f'{chr(10)}referencia (sprites reais a 96 px): nitidez '
          f'{np.mean([nitidez(i) for i in alvo]):.4f}  saturacao '
          f'{np.mean([satur_bicho(i) for i in alvo]):.4f}{chr(10)}')
    print(f'{"":26s} {"nitidez 512":>12s} {"satur":>8s} {"nitidez 96px":>13s}')
    for rot, m in resultado.items():
        print(f'{rot:26s} {m[:, 0].mean():12.4f} {m[:, 1].mean():8.4f} {m[:, 2].mean():13.4f}')

    W, H = 384, 28
    ncol = len(ALVOS) + 1
    nlin = len(mostrar)
    f = Image.new('RGB', (W * ncol + 6 * (ncol + 1), H + (W + 6) * nlin + 6), (215, 215, 215))
    d = ImageDraw.Draw(f)
    for c, n in enumerate(ALVOS, start=1):
        d.text((6 + c * (W + 6) + 10, 7), f'desenho {n}', fill=(0, 0, 0))
    for r, rot in enumerate(mostrar):
        d.text((10, H + r * (W + 6) + 6), rot, fill=(0, 0, 0))
        for c, im in enumerate(mostrar[rot], start=1):
            f.paste(im.resize((W, W), Image.NEAREST), (6 + c * (W + 6), H + r * (W + 6)))
    f.save(S / 'vale_a_pena.png')
    print(f'{chr(10)}-> vale_a_pena.png  (todas pixelizadas a 96 px)')


if __name__ == '__main__':
    main()
