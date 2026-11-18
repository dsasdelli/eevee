"""Varredura dos checkpoints da LoRA em poucos desenhos, para escolher o ponto.

LoRA de estilo costuma passar do ponto: o estilo gruda cedo e, dali em diante, o
adaptador comeca a comer a capacidade do prior -- as criaturas ficam genericas, a
composicao colapsa, as cores saturam. O ultimo passo raramente e o melhor, entao se
escolhe olhando, nao pegando o final.
"""
import sys

sys.path.insert(0, r'D:\Projetos\Github\eevee')

from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw

from esboco import fit_sketch

S = Path(r'C:\Users\Lomir\AppData\Local\Temp\claude\D--Projetos-Github-eevee\6116fa34-1652-486d-b8ec-f1811ba28098\scratchpad')
LORA = Path(r'D:\Projetos\Github\eevee\lora-pkmnsprite')
ALVOS = [1, 3, 7]
PASSOS = [None, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
PROMPT = ('pkmnsprite, a pokemon creature sprite, flat shaded pixel art, '
          'limited palette, plain white background, full body, centered')
NEG = 'photo, realistic, text, watermark, logo, blurry, grayscale, human, background scenery'


def ctrl(im):
    a = np.array(im.convert('L'), dtype=np.uint8)
    return Image.fromarray(255 - a).convert('RGB').resize((512, 512), Image.NEAREST)


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
    colunas = []
    for passo in PASSOS:
        pipe.unload_lora_weights()
        if passo is None:
            rot = 'SD cru'
        else:
            pipe.load_lora_weights(str(LORA), weight_name=f'lora-step{passo:06d}.safetensors')
            rot = str(passo)
        saidas = [pipe(PROMPT, image=ctrl(esb[n]), negative_prompt=NEG, num_inference_steps=28,
                       guidance_scale=7.5, controlnet_conditioning_scale=0.7,
                       generator=torch.Generator('cuda').manual_seed(42)).images[0]
                  for n in ALVOS]
        colunas.append((rot, saidas))
        print(rot, 'ok', flush=True)

    W, H = 384, 30
    cols = len(colunas) + 1
    folha = Image.new('RGB', (W * cols + 6 * (cols + 1), H + (W + 6) * len(ALVOS) + 6),
                      (215, 215, 215))
    d = ImageDraw.Draw(folha)
    d.text((10, 8), 'desenho', fill=(0, 0, 0))
    for c, (rot, _) in enumerate(colunas, start=1):
        d.text((6 + c * (W + 6) + 10, 8), rot, fill=(0, 0, 0))
    for r, n in enumerate(ALVOS):
        folha.paste(esb[n].convert('RGB').resize((W, W)), (6, H + r * (W + 6)))
        for c, (_, s) in enumerate(colunas, start=1):
            folha.paste(s[r].resize((W, W), Image.LANCZOS), (6 + c * (W + 6), H + r * (W + 6)))
    folha.save(S / 'varredura.png')
    print('->', S / 'varredura.png')


if __name__ == '__main__':
    main()
