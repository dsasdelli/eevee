"""Gera sprites a partir de esbocos pelo caminho de difusao: SD 1.5 + ControlNet + LoRA.

E o equivalente do pix2pix_predict.py para a segunda frente do projeto. A diferenca que
importa esta no --forca: o pix2pix TRACA o esboco (medido, a deformacao da entrada chega
a saida com 108% a 127% da intensidade), enquanto aqui o esboco e orientacao e a forca
do condicionamento e um dial continuo. 1,0 segue o traco de perto; 0,4 usa como
sugestao e redesenha. 0,7 equilibra.

    # um rabisco seu
    python lora_predict.py --input meu_rabisco.png

    # a pasta inteira, com o sprite pixelizado ao lado
    python lora_predict.py --input sprites-human-drawn/ --pixelizar

    # mais liberdade para o modelo corrigir uma forma torta
    python lora_predict.py --input rabisco.png --forca 0.5 --amostras 4

    # com refino de detalhe (dobra o tempo, ganho sutil)
    python lora_predict.py --input rabisco.png --hires --pixelizar
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionControlNetPipeline, UniPCMultistepScheduler)
from PIL import Image

from pix2pix_predict import SUFFIXES, fit_sketch

BASE = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CONTROLNET = 'lllyasviel/control_v11p_sd15_scribble'
# LoRA treinada SO na gen9 (551 sprites), escolhida pelo usuario em 08/11/2026 por
# entregar o acabamento de sprite moderno que ele queria. O passo 3000 bate a nitidez
# dos sprites gen9 reais quase exatamente (0,0528 contra 0,0525, medido sobre o corpo
# e nao sobre a imagem inteira -- com fundo incluido a metrica mede enquadramento).
#
# Alternativas, trocaveis com --lora:
#   lora-gen9/lora-step001000.safetensors   mais vistosa, cor mais viva
#   lora-gen9/lora-step002000.safetensors   idem
#   lora-pkmnsprite/lora-step008000.safetensors   as nove geracoes juntas, mais blocada
LORA_PADRAO = 'lora-gen9/lora-step003000.safetensors'

# O token-gatilho 'pkmnsprite' TEM de estar no prompt: e nele que o estilo foi treinado
# (ver lora_train.py). Sem ele a LoRA carrega mas quase nao atua.
PROMPT = ('pkmnsprite, a pokemon creature sprite, flat shaded pixel art, '
          'limited palette, plain white background, full body, centered')
NEGATIVO = ('photo, realistic, text, watermark, logo, blurry, grayscale, human, '
            'background scenery')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='arquivo ou pasta de esbocos')
    p.add_argument('--output-dir', default='predictions/difusao/saidas')
    p.add_argument('--lora', default=LORA_PADRAO,
                   help=f'arquivo .safetensors da LoRA, ou "nenhuma" para o SD cru '
                        f'(default: {LORA_PADRAO})')
    p.add_argument('--forca', type=float, default=0.7,
                   help='forca do ControlNet, 0 a 1. 1,0 segue o traco, 0,4 redesenha '
                        '(default: 0.7)')
    p.add_argument('--passos', type=int, default=28, help='passos de difusao')
    p.add_argument('--cfg', type=float, default=7.5, help='peso do prompt')
    p.add_argument('--amostras', type=int, default=1,
                   help='quantas variacoes por esboco. Ao contrario do pix2pix, que e '
                        'quase deterministico, aqui cada semente da um bicho diferente')
    p.add_argument('--semente', type=int, default=42)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--negativo', default=NEGATIVO)
    p.add_argument('--hires', type=float, nargs='?', const=0.35, default=None,
                   metavar='FORCA',
                   help='segunda passada de refino em 768 (hires fix). Acrescenta '
                        'detalhe pequeno -- boca, garra, separacao -- sem a composicao '
                        'amolecer, porque a forma ja foi decidida em 512. Faixa util '
                        '0,25 a 0,35 (default 0.35); em 0,5 comeca a perder a cara de '
                        'sprite. Dobra o tempo: 5,5s -> 10,6s por imagem')
    p.add_argument('--hires-lado', type=int, default=768,
                   help='resolucao da segunda passada (default 768)')
    p.add_argument('--fundo', choices=['manter', 'branco'], default='branco',
                   help='o difusor costuma inventar fundo colorido mesmo com o prompt '
                        'pedindo branco, e resolver isso no prompt negativo lava a '
                        'criatura junto (medido). "branco" troca o fundo depois, por '
                        'componente conexo que encosta na borda (default)')
    p.add_argument('--pixelizar', action='store_true',
                   help='salva tambem a versao reduzida a sprite, com paleta curta')
    p.add_argument('--lado', type=int, default=256,
                   help='lado do sprite com --pixelizar (default 256, que e o tamanho '
                        'dos sprites do dataset). Medido em 08/11/2026: a 96 px a '
                        'nitidez fica em 0,0758 contra 0,0253 dos sprites reais -- '
                        'blocao demais; a 256 cai para 0,0325, perto do alvo')
    p.add_argument('--cores', type=int, default=64,
                   help='cores com --pixelizar (default 64). Os sprites reais ficam '
                        'entre ~16 e ~124 cores; com 24 o corpo faixeia em 256 px')
    return p.parse_args()


# A GERACAO fica em 512 mesmo quando a saida e 256. Testado em 08/11/2026: gerar em
# 640 ou 768 para ter mais detalhe antes da reducao PIORA -- o SD 1.5 foi treinado em
# 512 e acima disso as criaturas amolecem e fogem do esboco (em 768 a serpente perdeu
# a forma por inteiro), com 2,5x mais tempo. Detalhe vem de reduzir menos, nao de
# gerar maior: 512 -> 256 e so 2x, contra 5,3x do antigo 512 -> 96.
GERACAO = 512


def para_controlnet(im, size=GERACAO):
    """O ControlNet scribble espera traco BRANCO em fundo PRETO -- o inverso do desenho.

    Sem a inversao ele le a folha inteira como traco e a saida ignora o esboco.
    """
    a = np.array(im.convert('L'), dtype=np.uint8)
    return Image.fromarray(255 - a).convert('RGB').resize((size, size), Image.NEAREST)


def tirar_fundo(im, tolerancia=0.12, cor=(255, 255, 255)):
    """Troca o fundo gerado por branco (ou o que for `cor`), sem tocar na criatura.

    POR QUE NAO RESOLVER ISSO NO PROMPT

    Tentado e medido em 08/11/2026: por o nome das cores indesejadas no prompt negativo
    ('green background', 'yellow background'...) de fato limpa mais fundos -- a fracao
    de amostras com fundo limpo vai de 35% para 60% -- mas LAVA A CRIATURA junto. O CFG
    nao localiza: banir uma cor do fundo bane da imagem inteira. O numero melhora e o
    resultado piora. Por isso o conserto e geometrico, aqui, e nao textual.

    COMO

    A cor do fundo e a MEDIANA da borda, nao um branco fixo: o fundo gerado costuma ser
    chapado mas de cor arbitraria. Depois so se apagam os componentes conexos que
    ENCOSTAM na borda -- uma regiao da mesma cor dentro do bicho (uma barriga clara,
    por exemplo) nao toca a moldura e sobrevive.
    """
    from scipy import ndimage

    a = np.asarray(im.convert('RGB'), np.float32) / 255
    h, w, _ = a.shape
    moldura = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    fundo = np.median(moldura, axis=0)

    perto = np.abs(a - fundo).max(axis=2) < tolerancia
    rotulos, n = ndimage.label(perto)
    if n == 0:
        return im.convert('RGB')
    da_borda = set(np.concatenate([rotulos[0, :], rotulos[-1, :],
                                   rotulos[:, 0], rotulos[:, -1]]).tolist()) - {0}
    if not da_borda:
        return im.convert('RGB')
    mascara = np.isin(rotulos, list(da_borda))

    saida = (a * 255).astype(np.uint8)
    saida[mascara] = np.array(cor, np.uint8)
    return Image.fromarray(saida)


def pixelizar(im, lado, cores):
    """512 px continuo -> sprite de `lado` px com paleta curta.

    BOX na reducao (media da celula) e nao amostragem: amostrar pegaria um pixel
    arbitrario de cada bloco e herdaria o ruido do difusor. Quantizacao SEM dithering,
    porque dither espalha pontinhos -- o oposto de pixel art chapada.
    """
    p = im.resize((lado, lado), Image.BOX)
    return p.quantize(colors=cores, method=Image.MEDIANCUT, dither=Image.NONE).convert('RGB')


def main():
    args = parse_args()
    origem = Path(args.input)
    arquivos = ([origem] if origem.is_file()
                else sorted(p for p in origem.iterdir() if p.suffix.lower() in SUFFIXES))
    if not arquivos:
        raise FileNotFoundError(f'nenhuma imagem em {origem}')

    cn = ControlNetModel.from_pretrained(CONTROLNET, dtype=torch.float16, variant='fp16')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=cn, variant='fp16', dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
    pipe.set_progress_bar_config(disable=True)

    if args.lora != 'nenhuma':
        caminho = Path(args.lora)
        if not caminho.exists():
            raise SystemExit(f'LoRA nao encontrada: {caminho}')
        pipe.load_lora_weights(str(caminho.parent), weight_name=caminho.name)
        print(f'LoRA: {caminho.name}')
    else:
        print('LoRA: nenhuma (SD cru)')

    # O pipeline de refino reusa os MESMOS componentes ja na GPU; instanciar outro do
    # zero duplicaria 3,4 GB de pesos sem necessidade.
    refino = StableDiffusionControlNetImg2ImgPipeline(**pipe.components) if args.hires else None
    if refino is not None:
        refino.set_progress_bar_config(disable=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for arq in arquivos:
        esboco = fit_sketch(Image.open(arq), GERACAO)
        controle = para_controlnet(esboco)
        controle_hires = para_controlnet(esboco, args.hires_lado) if args.hires else None
        for k in range(args.amostras):
            semente = args.semente + k
            img = pipe(args.prompt, image=controle, negative_prompt=args.negativo,
                       num_inference_steps=args.passos, guidance_scale=args.cfg,
                       controlnet_conditioning_scale=args.forca,
                       generator=torch.Generator(pipe.device.type).manual_seed(semente)
                       ).images[0]
            if refino is not None:
                # LANCZOS na ampliacao: a segunda passada parte de uma imagem ja
                # composta, e um reamostrador duro introduziria serrilhado que o
                # refino depois trataria como detalhe a preservar.
                grande = img.resize((args.hires_lado, args.hires_lado), Image.LANCZOS)
                img = refino(args.prompt, image=grande, control_image=controle_hires,
                             negative_prompt=args.negativo, strength=args.hires,
                             num_inference_steps=args.passos, guidance_scale=args.cfg,
                             controlnet_conditioning_scale=args.forca,
                             generator=torch.Generator(pipe.device.type).manual_seed(semente)
                             ).images[0]
            if args.fundo == 'branco':
                img = tirar_fundo(img)
            sufixo = '' if args.amostras == 1 else f'-{k + 1}'
            destino = out / f'{arq.stem}-gerado{sufixo}.png'
            img.save(destino)
            print(f'  {arq.name} -> {destino}')
            if args.pixelizar:
                pix = out / f'{arq.stem}-sprite{sufixo}.png'
                pixelizar(img, args.lado, args.cores).save(pix)
                print(f'  {arq.name} -> {pix}  ({args.lado}px, {args.cores} cores)')


if __name__ == '__main__':
    main()
