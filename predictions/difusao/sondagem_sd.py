"""Sondagem: SD 1.5 + ControlNet scribble pronto, SEM treino nenhum.

Pergunta unica: o prior pre-treinado produz criatura plausivel respeitando a forma
de longe? Se sim, o caminho se valida e a LoRA so acerta o estilo.

O botao que importa e controlnet_conditioning_scale: 1.0 segue o traco de perto,
0.5 usa como sugestao. E exatamente a liberdade que o pix2pix nao tem.
"""
import sys, torch, numpy as np
sys.path.insert(0, r'D:\Projetos\Github\eevee')
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from esboco import fit_sketch

S = Path(r'C:\Users\Lomir\AppData\Local\Temp\claude\D--Projetos-Github-eevee\6116fa34-1652-486d-b8ec-f1811ba28098\scratchpad')
# variant='fp16' e OBRIGATORIO aqui: so os pesos fp16 estao em cache. Sem a flag o
# diffusers procura os fp32 (que foram apagados por ocuparem 22 GB inuteis) e falha.
cn = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_scribble',
                                     torch_dtype=torch.float16, variant='fp16')
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5', controlnet=cn, variant='fp16',
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')
pipe.set_progress_bar_config(disable=True)

PROMPT = ('pokemon official artwork, cute creature, vibrant flat colors, '
          'clean cel shading, white background, nintendo game freak style')
NEG = 'photo, realistic, text, watermark, blurry, grayscale, human, sketch lines'

# O ControlNet scribble espera traco BRANCO em fundo PRETO -- o inverso do desenho.
def para_controlnet(im):
    a = np.array(im.convert('L'), dtype=np.uint8)
    return Image.fromarray(255 - a).convert('RGB').resize((512,512), Image.NEAREST)

escalas = [1.0, 0.7, 0.4]
linhas = []
for n in range(1,8):
    esb = fit_sketch(Image.open(f'sprites-human-drawn/pokemon{n}.bmp'), 512)
    ctrl = para_controlnet(esb)
    row = [esb.convert('RGB').resize((512,512))]
    for esc in escalas:
        img = pipe(PROMPT, image=ctrl, negative_prompt=NEG, num_inference_steps=28,
                   guidance_scale=7.5, controlnet_conditioning_scale=esc,
                   generator=torch.Generator('cuda').manual_seed(42)).images[0]
        row.append(img)
    linhas.append(row); print(f'pokemon{n} ok', flush=True)

W=512; cols=len(linhas[0])
f=Image.new('RGB',(W*cols+8*(cols+1), W*7+8*8),(210,210,210))
for r,row in enumerate(linhas):
    for c,im in enumerate(row): f.paste(im,(8+c*(W+8), 8+r*(W+8)))
f.resize((f.width//2, f.height//2), Image.LANCZOS).save(S/'sondagem_sd.png')
print('-> sondagem_sd.png  (desenho | escala 1.0 | 0.7 | 0.4)')
