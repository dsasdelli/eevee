"""Treina uma LoRA de ESTILO de sprite de Pokemon sobre o Stable Diffusion 1.5.

POR QUE LORA, E NAO AJUSTE COMPLETO

O SD 1.5 ja sabe o que e uma criatura: anatomia, rosto, volume, sombreado. Isso veio
de centenas de milhoes de imagens e e exatamente o que os 4.289 sprites deste projeto
nunca poderiam ensinar. O que falta nele e o ESTILO -- sprite chapado, paleta curta,
fundo branco, enquadramento de ficha. A LoRA injeta matrizes de baixo posto nas camadas
de atencao e treina so elas: alguns MB em vez de 3,2 GB, e sem risco de apagar o
conhecimento que torna o caminho da difusao interessante em primeiro lugar.

DECISOES QUE IMPORTAM

Legenda unica com token-gatilho. Nao se usa o nome do pokemon: com 'bulbasaur' na
legenda a rede aprende a associar nome a aparencia, e o objetivo aqui e justamente o
contrario -- criatura inedita no estilo certo. Todas as amostras levam a mesma frase,
entao o unico sinal que a LoRA pode extrair e o estilo comum a elas.

So o SPLIT DE TREINO entra. A divisao e a de esboco.fixed_split, com semente fixa,
para que os 1.000 pares de teste continuem nunca vistos.

Escala NEAREST para 512. Os sprites sao 256x256 de pixel art; interpolacao suave
inventaria tons intermediarios e ensinaria a LoRA a borrar justamente a borda dura
que e a assinatura do estilo.

    python lora_train.py --steps 6000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from aug_artwork import augmenta, augmenta_pixel, borda_branca
from helper import GRUPOS, dir_processadas, especies
from esboco import fixed_split, list_pairs

BASE = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
# Token raro de proposito: 'pkmnsprite' nao e palavra comum no vocabulario do CLIP,
# entao o estilo gruda nele em vez de contaminar termos uteis como 'pokemon'.
LEGENDA = ('pkmnsprite, a pokemon creature sprite, flat shaded pixel art, '
           'limited palette, plain white background, full body, centered')


class Sprites(Dataset):
    """So a imagem: a legenda e a mesma para todas e fica fora do laco de dados."""

    def __init__(self, paths, size, aug=False, suave=False):
        self.paths, self.size, self.aug = paths, size, aug
        # NEAREST e para pixel art (ver o docstring); arte oficial tem gradiente e linha
        # antisserrilhada, e la LANCZOS e o certo -- o mesmo do artwork-processer.py.
        self.resample = Image.LANCZOS if suave else Image.NEAREST
        self._rng = None
        self.branca = None
        if aug and suave:
            # Decidido uma vez por arquivo: preencher cantos de branco ou por reflexo.
            self.branca = [borda_branca(Image.open(p)) for p in paths]
            print(f'augmentacao ligada: {sum(self.branca)} de {len(paths)} com fundo '
                  f'branco, {len(paths) - sum(self.branca)} preenchidas por reflexo')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert('RGB')
        if self.aug and self._rng is None:
            # Criado DENTRO do worker e uma vez so: com persistent_workers o estado
            # avanca entre epocas, entao a mesma imagem nao repete a mesma variacao,
            # e cada worker tem semente propria (torch.initial_seed difere por worker).
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        if self.aug and self.branca is None:
            # Pixel art: augmenta no tamanho NATIVO, antes do NEAREST, para todo
            # deslocamento ser em pixel inteiro de sprite (ver aug_artwork.augmenta_pixel).
            im = augmenta_pixel(im, self._rng)
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size), self.resample)
        if self.aug and self.branca is not None:
            im = augmenta(im, self._rng, self.branca[i])
        a = torch.from_numpy(np.array(im, np.uint8)).permute(2, 0, 1).float()
        return a.div_(127.5).sub_(1.0)      # [-1, 1], a faixa que o VAE espera


SPLITS = Path(__file__).parent / 'splits'


def especies_de_teste():
    """As especies de teste, de splits/teste-especies.txt (linhas com # sao comentario)."""
    lista = SPLITS / 'teste-especies.txt'
    if not lista.exists():
        raise SystemExit(f'{lista} nao existe -- rode splits/gera_split.py')
    return {l.strip() for l in lista.read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')}


def teste_do_grupo(grupo):
    """Caminhos das artes de TESTE de um grupo: toda arte cuja especie esta em
    splits/teste-especies.txt. A lista e fixa e versionada; o treino e o grupo menos ela.

    Historia: ate 17/11/2026 o teste era sorteado na hora por arte, e o antigo --categoria
    sorteava OUTRO split no subconjunto (as LoRAs Sugimori de entao treinaram em 112 das 300
    artes de teste). Em 18/11 virou 100 artes fixas por grupo. Em 19/11 virou 10% das
    ESPECIES, comum aos 3 grupos: antes a mesma especie (ate como shiny, com o mesmo traco)
    ficava no treino em outra arte. Escolha do sorteio em splits/gera_split.py.
    """
    teste_esp = especies_de_teste()
    return [p for p, e in especies(grupo).items() if e in teste_esp]


def caminhos_grupo(grupo):
    """Caminhos de TREINO de um grupo: todas as artes processadas menos as de teste."""
    raiz = Path(dir_processadas(grupo))
    if not raiz.exists():
        raise SystemExit(f'{raiz} nao existe -- rode artwork-processer.py --grupo {grupo}')
    todos = sorted(raiz.glob('gen*/*.png'))
    teste = set(teste_do_grupo(grupo))
    treino = [p for p in todos if p not in teste]
    print(f'{grupo}: {len(todos)} imagens, {len(treino)} de treino, {len(teste)} de teste')
    return treino


def grupos_de(dados):
    """'sugimori+outros' -> ['sugimori', 'outros']; erro de argumento se houver grupo invalido."""
    gs = dados.split('+')
    ruins = [g for g in gs if g not in GRUPOS]
    if ruins or len(set(gs)) != len(gs):
        raise argparse.ArgumentTypeError(
            f'{dados!r}: use sprites ou grupos de {"/".join(GRUPOS)} unidos por +')
    return dados


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=4,
                   help='4 e o otimo medido na 3060 de 12 GB: 4,1 imagens/s em 6,1 GB. '
                        'Lote 8 nao acelera nada e sobe para 10,1 GB')
    p.add_argument('--accum', type=int, default=2, help='lotes acumulados antes do update')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--rank', type=int, default=16, help='posto das matrizes LoRA')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--out', default='lora-pkmnsprite')
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dados', type=lambda d: d if d == 'sprites' else grupos_de(d),
                   default='sprites',
                   help='de qual conjunto treinar. "sprites" sao os 256x256 de jogo '
                        '(default, usa o fixed_split de esboco). Para arte, um ou mais grupos '
                        'unidos por +: sugimori, outros, anime -- ex. "sugimori", '
                        '"sugimori+outros", "sugimori+outros+anime". Cada grupo tira do '
                        'treino as artes das especies de splits/teste-especies.txt')
    p.add_argument('--gen', metavar='GERACOES',
                   help='treina so nas geracoes dadas, separadas por virgula '
                        '(ex. gen9 ou gen9,gen7). O dataset inteiro e a media de nove '
                        'geracoes, e as antigas sao sprites de 56 a 120 px ampliados -- '
                        'misturadas, a LoRA sai com pixel art dura embutida. Medido em '
                        '08/11/2026, so gen7 e gen9 sao 256x256 nativos. O filtro e '
                        'aplicado DEPOIS do fixed_split, entao o conjunto de teste '
                        'continua intacto e as medicoes seguem comparaveis')
    p.add_argument('--resume', metavar='ARQUIVO',
                   help='continua de uma LoRA ja treinada (.safetensors). So os PESOS '
                        'sao retomados -- o estado do Adam nao e salvo, entao ha um '
                        'solavanco pequeno nos primeiros passos. Para LoRA isso e '
                        'aceitavel e evita refazer o que ja foi treinado. A numeracao '
                        'CONTINUA do passo do arquivo (lora-step016000 -> o proximo e '
                        '016000 + --save-every) e a perda.csv recebe as linhas novas, '
                        'entao use o mesmo --out da rodada original: nada de pasta -cont. '
                        '--steps conta so os passos NOVOS')
    p.add_argument('--aug', action='store_true',
                   help='augmentacao on-the-fly (espelho, rotacao/escala/translacao leves, '
                        'brilho/contraste minimo). Amplitudes em aug_artwork.py. Desligado '
                        'por default para as rodadas antigas continuarem reproduziveis')
    p.add_argument('--grad-checkpointing', action='store_true',
                   help='troca velocidade por memoria. Desligado por default: medido, '
                        'custa 17%% de velocidade e a 3060 tem folga de sobra sem ele')
    return p.parse_args()


def main():
    args = parse_args()
    dev = 'cuda'

    tok = CLIPTokenizer.from_pretrained(BASE, subfolder='tokenizer')
    txt = CLIPTextModel.from_pretrained(BASE, subfolder='text_encoder',
                                        variant='fp16', dtype=torch.float16).to(dev)
    vae = AutoencoderKL.from_pretrained(BASE, subfolder='vae',
                                        variant='fp16', dtype=torch.float16).to(dev)
    # A UNet treina em fp32. O VAE e o CLIP podem ficar em fp16 porque estao
    # congelados e so fazem passagem direta, mas rodar backward sobre pesos fp16 sem
    # escalador de gradiente faz as atualizacoes estourarem ou zerarem por falta de
    # faixa dinamica -- medido em 07/11/2026: a perda SUBIU de 0,058 para 0,21 e a LoRA
    # saiu ruido puro. A conta roda em fp16 via autocast; os pesos ficam em fp32.
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder='unet',
                                                variant='fp16', dtype=torch.float32).to(dev)
    sched = DDPMScheduler.from_pretrained(BASE, subfolder='scheduler')

    # Tudo congelado; so os adaptadores treinam.
    for m in (txt, vae, unet):
        m.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']))

    if args.resume:
        from safetensors.torch import load_file
        bruto = load_file(args.resume)
        # save_lora_weights grava no estilo diffusers, com prefixo 'unet.'; tirar o
        # prefixo basta para casar com as chaves do peft. Conferido em 08/11/2026:
        # NAO acrescentar '.default.' -- get_peft_model_state_dict devolve sem ele.
        from peft import set_peft_model_state_dict
        limpo = {k.removeprefix('unet.'): v for k, v in bruto.items()}
        set_peft_model_state_dict(unet, limpo)
        # Conferencia de valor, nao de contagem: um resume silenciosamente parcial
        # recomecaria de um estado quebrado sem dar erro nenhum.
        agora = get_peft_model_state_dict(unet)
        iguais = sum(1 for k, v in limpo.items()
                     if k in agora and torch.equal(agora[k].cpu().float(), v.cpu().float()))
        if iguais != len(limpo):
            raise SystemExit(f'resume incompleto: {iguais} de {len(limpo)} tensores '
                             f'conferem com o arquivo. Nao treine a partir disto.')
        print(f'retomado de {Path(args.resume).name}: {iguais}/{len(limpo)} tensores conferidos')

    treinaveis = [q for q in unet.parameters() if q.requires_grad]
    print(f'treinaveis: {sum(q.numel() for q in treinaveis) / 1e6:.2f} M de '
          f'{sum(q.numel() for q in unet.parameters()) / 1e6:.0f} M parametros')

    opt = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=1e-2)
    # O escalador multiplica a perda antes do backward para os gradientes pequenos nao
    # sumirem na faixa do fp16, e divide de volta antes do passo do otimizador.
    escalador = torch.amp.GradScaler('cuda')
    if args.grad_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.dados != 'sprites':
        paths = [p for g in args.dados.split('+') for p in caminhos_grupo(g)]
    else:
        _, reais = list_pairs()
        treino_idx, _ = fixed_split(len(reais))
        paths = [reais[i] for i in treino_idx]
    if args.gen:
        antes = len(paths)
        querida = {g.strip() for g in args.gen.split(',') if g.strip()}
        disponiveis = {p.parent.name for p in paths}
        if not querida <= disponiveis:
            raise SystemExit(f'--gen: geracao desconhecida {sorted(querida - disponiveis)}. '
                             f'Disponiveis: {", ".join(sorted(disponiveis))}')
        paths = [p for p in paths if p.parent.name in querida]
        if not paths:
            raise SystemExit(f'--gen {args.gen}: nada no split de treino')
        quebra = {g: sum(1 for p in paths if p.parent.name == g) for g in sorted(querida)}
        print(f'filtrado para {"+".join(sorted(querida))}: {len(paths)} de {antes} '
              f'sprites de treino  {quebra}')
    print(f'{len(paths)} imagens de treino (o conjunto de teste fica de fora)')
    ld = DataLoader(Sprites(paths, args.size, aug=args.aug,
                            suave=args.dados != 'sprites'), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=True)

    # A legenda e constante, entao o texto passa pelo CLIP UMA vez e nao a cada passo.
    with torch.no_grad():
        ids = tok([LEGENDA], padding='max_length', max_length=tok.model_max_length,
                  truncation=True, return_tensors='pt').input_ids.to(dev)
        ctx = txt(ids)[0].float().repeat(args.batch_size, 1, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Log estruturado, alem do stdout: o grafico de perda le daqui em vez de
    # parsear texto, e cada rodada fica com seu proprio arquivo.
    csv = out / 'perda.csv'
    # Com --resume a contagem continua do passo do arquivo retomado, na mesma pasta.
    inicio = 0
    if args.resume:
        import re
        m = re.search(r'lora-step(\d+)', Path(args.resume).name)
        if not m:
            raise SystemExit(f'--resume {args.resume}: nome sem lora-stepNNNNNN, nao sei o passo')
        inicio = int(m.group(1))
    finais = [inicio + k for k in range(args.save_every, args.steps + 1, args.save_every)]
    finais += [inicio + args.steps]
    ja_existem = [n for n in finais if (out / f'lora-step{n:06d}.safetensors').exists()]
    if ja_existem:
        raise SystemExit(f'{out} ja tem lora-step{ja_existem[0]:06d}: esta rodada sobrescreveria')
    if inicio and csv.exists():
        # Corta o que a rodada original registrou depois do ponto retomado.
        linhas = csv.read_text(encoding='utf-8').splitlines()
        linhas = [l for l in linhas[1:] if int(l.split(',')[0]) <= inicio]
        csv.write_text(chr(10).join(['passo,perda'] + linhas) + chr(10), encoding='utf-8')
    else:
        csv.write_text('passo,perda' + chr(10), encoding='utf-8')
    passo, t0, perdas = 0, time.time(), []
    while passo < args.steps:
        for x in ld:
            if passo >= args.steps:
                break
            x = x.to(dev, dtype=torch.float16)
            with torch.no_grad():
                lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            lat = lat.float()                       # o VAE e fp16; a UNet treina em fp32
            ruido = torch.randn_like(lat)
            t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
            sujo = sched.add_noise(lat, ruido, t)
            with torch.autocast('cuda', dtype=torch.float16):
                pred = unet(sujo, t, encoder_hidden_states=ctx).sample
                # O SD 1.5 preve o RUIDO (epsilon), nao a imagem: o alvo e o ruido.
                perda = F.mse_loss(pred.float(), ruido)
            escalador.scale(perda / args.accum).backward()
            perdas.append(perda.item())

            if (passo + 1) % args.accum == 0:
                # unscale antes do clip: cortar norma sobre gradiente escalado cortaria
                # no limiar errado.
                escalador.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
                escalador.step(opt)
                escalador.update()
                opt.zero_grad(set_to_none=True)

            passo += 1
            if passo % 20 == 0:
                with csv.open('a', encoding='utf-8') as fh:
                    fh.write(f'{inicio + passo},{np.mean(perdas[-20:]):.6f}' + chr(10))
            if passo % 200 == 0:
                dt = time.time() - t0
                print(f'passo {passo}/{args.steps} | perda {np.mean(perdas[-200:]):.4f} | '
                      f'{dt / passo:.3f} s/passo | faltam '
                      f'{(args.steps - passo) * dt / passo / 60:.0f} min', flush=True)
            if passo % args.save_every == 0 or passo == args.steps:
                StableDiffusionPipeline.save_lora_weights(
                    out, get_peft_model_state_dict(unet),
                    weight_name=f'lora-step{inicio + passo:06d}.safetensors')
                print(f'  salvo: {out}/lora-step{inicio + passo:06d}.safetensors', flush=True)

    print(f'concluido: {args.steps} passos em {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    main()
