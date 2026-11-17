"""Compara a LoRA de estilo contra o SD cru, nos sete desenhos humanos.

Varre checkpoints intermediarios de proposito: LoRA de estilo costuma passar do ponto,
e o ultimo passo nem sempre e o melhor.
"""
import sys

sys.path.insert(0, r'D:\Projetos\Github\eevee')

from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

from pix2pix_predict import fit_sketch

S = Path(r'C:\Users\Lomir\AppData\Local\Temp\claude\D--Projetos-Github-eevee\6116fa34-1652-486d-b8ec-f1811ba28098\scratchpad')
LORA = Path(r'D:\Projetos\Github\eevee\lora-pkmnsprite')
ESCALA = 0.7          # a que melhor equilibrou forma e liberdade na sondagem crua
PASSOS = [None, 2000, 6000, 11000]

# A legenda tem de conter o token-gatilho treinado, senao a LoRA nao e acionada.
PROMPT = ('pkmnsprite, a pokemon creature sprite, flat shaded pixel art, '
          'limited palette, plain white background, full body, centered')
NEG = 'photo, realistic, text, watermark, logo, blurry, grayscale, human, background scenery'


def para_controlnet(im):
    """O ControlNet scribble espera traco BRANCO em fundo PRETO -- inverso do desenho."""
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

    esbocos = {}
    for n in range(1, 8):
        esb = fit_sketch(Image.open(rf'D:\Projetos\Github\eevee\sprites-human-drawn\pokemon{n}.bmp'), 512)
        esbocos[n] = (esb.convert('RGB').resize((512, 512)), para_controlnet(esb))

    colunas = {}
    for passo in PASSOS:
        if passo is None:
            pipe.unload_lora_weights()
            rotulo = 'SD cru'
        else:
            pipe.unload_lora_weights()
            pipe.load_lora_weights(str(LORA), weight_name=f'lora-step{passo:06d}.safetensors')
            rotulo = f'LoRA {passo}'
        saidas = []
        for n in range(1, 8):
            img = pipe(PROMPT, image=esbocos[n][1], negative_prompt=NEG,
                       num_inference_steps=28, guidance_scale=7.5,
                       controlnet_conditioning_scale=ESCALA,
                       generator=torch.Generator('cuda').manual_seed(42)).images[0]
            saidas.append(img)
        colunas[rotulo] = saidas
        print(f'{rotulo} ok', flush=True)

    rotulos = ['desenho'] + list(colunas)
    cols = len(rotulos)
    W = 512
    folha = Image.new('RGB', (W * cols + 8 * (cols + 1), W * 7 + 8 * 8), (210, 210, 210))
    for n in range(1, 8):
        r = n - 1
        folha.paste(esbocos[n][0], (8, 8 + r * (W + 8)))
        for c, rot in enumerate(colunas, start=1):
            folha.paste(colunas[rot][r], (8 + c * (W + 8), 8 + r * (W + 8)))
    folha.resize((folha.width // 2, folha.height // 2), Image.LANCZOS).save(S / 'aval_lora.png')
    print('colunas:', ' | '.join(rotulos))
    print('->', S / 'aval_lora.png')


if __name__ == '__main__':
    main()
