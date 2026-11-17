"""Leaderboard dos modelos: a mesma bancada para todos, de agora em diante.

    python leaderboard.py              mede o que ainda nao foi medido e reescreve o HTML
    python leaderboard.py --refazer X  mede de novo o modelo X (ou 'todos')
    python leaderboard.py --so-html    so reescreve leaderboard/index.html

Para entrar um modelo novo, acrescente uma linha em MODELOS e rode de novo: os ja
medidos ficam no cache (leaderboard/resultados.json) e nao sao refeitos.

A BANCADA

Entrada: os 10 desenhos humanos de sprites-human-drawn, cada um com 4 sementes = 40
saidas por modelo. A saida e medida CRUA: sem tirar_fundo e sem pixelizar, porque o
fundo e uma das coisas medidas e o pos-processamento e opcional no app.

Tres metricas, todas "menor e melhor":

1. FIDELIDADE ao esboco (px). Extrai o contorno da saida com o MESMO limiar adaptativo
   do artwork-contours.py e mede, para cada pixel de traco do esboco, a distancia ate a
   borda mais proxima da saida. Media em px num quadro de 512. Mede "a forma obedece ao
   desenho". Nao penaliza detalhe interno que a saida acrescente -- arte oficial tem.

2. ESTILO: KID x 1000 contra as 300 artes oficiais de TESTE (nenhuma LoRA as viu), no
   espaco de caracteristicas do DINOv2. KID e a versao nao-viesada do FID e funciona com
   poucas amostras. Mede "parece arte oficial". A linha de referencia (artes reais do
   treino contra as de teste) mostra o piso e o ruido da medida.

3. FUNDO: fracao das saidas cuja borda (faixa de 8 px) tem mais de 30% de pixels
   nao-brancos.

A posicao e a media dos postos nas tres metricas.
"""
import argparse
import base64
import io
import json
import random
import time
from datetime import date
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

RAIZ = Path(__file__).parent
SAIDA = RAIZ / 'leaderboard'
CACHE = SAIDA / 'resultados.json'
VERSAO = 2                        # muda se a bancada mudar: invalida o cache
DESENHOS = list(range(1, 11))
SEMENTES = [0, 1, 2, 3]
LADO = 512

P2P = RAIZ / 'sprites-processed-countours_torch_training_checkpoints'

# (id, rotulo, familia, fonte, nota). fonte: chave do app.MODELOS (difusao) ou .pt (pix2pix)
MODELOS = [
    ('art-aug-8k',  'Artwork oficial (aug 8000)',   'LoRA artwork', 'art-aug-8k',  'todas as categorias, augmentada'),
    ('art-aug-16k', 'Artwork oficial (aug 16000)',  'LoRA artwork', 'art-aug-16k', 'idem, treino longo'),
    ('sug-aug-8k',  'Artwork Sugimori (aug 8000)',  'LoRA artwork', 'sug-aug-8k',  'so Sugimori, augmentada'),
    ('art-tudo',    'Artwork oficial (tudo)',       'LoRA artwork', 'art-tudo',    'sem augmentacao, passo 4000'),
    ('art-sugimori','Artwork oficial (Sugimori)',   'LoRA artwork', 'art-sugimori','sem augmentacao, passo 2000'),
    ('gen9',        'Gen 9',                        'LoRA sprite',  'gen9',        '551 sprites gen9, passo 3000'),
    ('gen9-vivo',   'Gen 9 (cor mais viva)',        'LoRA sprite',  'gen9-vivo',   'passo 1000'),
    ('g97',         'Gen 9+7',                      'LoRA sprite',  'g97',         '773 sprites, passo 1000'),
    ('g97-contido', 'Gen 9+7 (contido)',            'LoRA sprite',  'g97-contido', 'passo 4000'),
    ('sprite',      'Sprite classico (9 geracoes)', 'LoRA sprite',  'sprite',      '3.289 sprites, passo 8000'),
    ('puro',        'Sem LoRA (SD 1.5 puro)',       'Base',         'puro',        'so o modelo base, prompt proprio'),
    ('sug-aug-12k', 'Artwork Sugimori (aug 12000)', 'LoRA artwork', 'lora-art-sugimori-aug-cont/lora-step004000.safetensors', 'continuacao com --resume do 8000'),
    ('sug-aug-16k', 'Artwork Sugimori (aug 16000)', 'LoRA artwork', 'lora-art-sugimori-aug-cont/lora-step008000.safetensors', 'continuacao com --resume do 8000'),
    ('art-aug-24k', 'Artwork oficial (aug 24000)',  'LoRA artwork', 'lora-art-tudo-aug-cont/lora-step008000.safetensors', 'continuacao com --resume do 16000'),
    ('art-aug-32k', 'Artwork oficial (aug 32000)',  'LoRA artwork', 'lora-art-tudo-aug-cont/lora-step016000.safetensors', 'continuacao com --resume do 16000'),
    ('sprite-aug-4k', 'Sprite 9 geracoes (aug 4000)', 'LoRA sprite', 'lora-sprite-aug/lora-step004000.safetensors', 'augmentacao de pixel art: espelho + deslocamento'),
    ('sprite-aug-8k', 'Sprite 9 geracoes (aug 8000)', 'LoRA sprite', 'lora-sprite-aug/lora-step008000.safetensors', 'augmentacao de pixel art'),
    ('gen9-aug-2k', 'Gen 9 (aug 2000)',  'LoRA sprite', 'lora-gen9-aug/lora-step002000.safetensors', 'augmentacao de pixel art'),
    ('gen9-aug-4k', 'Gen 9 (aug 4000)',  'LoRA sprite', 'lora-gen9-aug/lora-step004000.safetensors', 'augmentacao de pixel art'),
    ('p2p-scribble',  'pix2pix paleta+rabisco',     'pix2pix', 'scribble-lam100-step040000.pt',  'melhor pix2pix; aqui sem rabisco, paleta sorteada'),
    ('p2p-palette',   'pix2pix paleta',             'pix2pix', 'palette-lam100-step040000.pt',   'paleta sorteada'),
    ('p2p-infantil',  'pix2pix infantil',           'pix2pix', 'infantil-lam100-step040000.pt',  'traco deformado no treino'),
    ('p2p-skip2',     'pix2pix skip_drop',          'pix2pix', 'skip2-lam100-step040000.pt',     'beco sem saida'),
    ('p2p-stroke',    'pix2pix aug de traco',       'pix2pix', 'stroke-lam15-transpose-step040000.pt', 'lambda 15'),
    ('p2p-baseline',  'pix2pix baseline',           'pix2pix', 'baseline-lam100-step040000.pt',  'lambda 100, sem augmentacao'),
    ('p2p-lam15',     'pix2pix lambda 15',          'pix2pix', 'lam15-step050000.pt',            'o primeiro'),
    ('p2p-resize',    'pix2pix resize (colapsou)',  'pix2pix', 'resize-lam15-step040000-colapsou.pt', 'decoder resize, colapso de modo'),
]


# ------------------------------------------------------------------ metricas
def tinta_da_saida(im):
    """Contorno da saida com os parametros do artwork-contours.py; True = traco."""
    g = np.asarray(im.convert('L'))
    out = np.zeros_like(g)
    g = cv.GaussianBlur(g, (9, 9), 0)
    g = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    cont, _ = cv.findContours(g, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(out, cont, -1, 255, thickness=cv.FILLED)
    return out < 128          # o FUNDO e pintado de branco; o traco e o que fica escuro


def fidelidade(esboco, saida):
    traco = np.asarray(esboco.convert('L').resize((LADO, LADO), Image.NEAREST)) < 128
    borda = tinta_da_saida(saida.resize((LADO, LADO), Image.LANCZOS))
    if not traco.any() or not borda.any():
        return float(LADO)
    dist = distance_transform_edt(~borda)
    return float(dist[traco].mean())


def com_fundo(im):
    a = np.asarray(im.convert('RGB').resize((LADO, LADO)))
    b = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3),
                        a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    return float((b.min(1) < 235).mean()) > 0.3


class Dino:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.net = AutoModel.from_pretrained('facebook/dinov2-small').cuda().eval()

    @torch.no_grad()
    def __call__(self, ims):
        feats = []
        for i in range(0, len(ims), 32):
            x = self.proc(images=[im.convert('RGB') for im in ims[i:i + 32]],
                          return_tensors='pt')['pixel_values'].cuda()
            feats.append(self.net(pixel_values=x).pooler_output.float().cpu().numpy())
        f = np.concatenate(feats)
        # Normalizado e reescalado para x.y/d virar o COSSENO: as features cruas do DINOv2
        # tem norma grande e o nucleo cubico explodia (v1: KID na casa de 20.000, ruido
        # maior que o sinal).
        return f / np.linalg.norm(f, axis=1, keepdims=True) * np.sqrt(f.shape[1])


def kid(x, y):
    """MMD^2 nao-viesado com nucleo polinomial cubico (Binkowski et al. 2018)."""
    d = x.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    m, n = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


# ------------------------------------------------------------------ geracao
def esbocos():
    from pix2pix_predict import fit_sketch
    return {n: fit_sketch(Image.open(RAIZ / f'sprites-human-drawn/pokemon{n}.bmp'), LADO)
            for n in DESENHOS}


class Difusao:
    def __init__(self):
        import app
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)
        self.app = app
        cn = ControlNetModel.from_pretrained(app.CONTROLNET if hasattr(app, 'CONTROLNET')
                                             else 'lllyasviel/control_v11p_sd15_scribble',
                                             dtype=torch.float16, variant='fp16')
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-v1-5', controlnet=cn, variant='fp16',
            dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to('cuda')
        self.pipe.set_progress_bar_config(disable=True)

    def gerar(self, chave, esb):
        # fonte = chave do app.MODELOS, ou caminho direto de um .safetensors (checkpoint
        # que nao esta no app)
        if chave.endswith('.safetensors'):
            m = {'arquivo': RAIZ / chave}
        else:
            m = self.app.MODELOS[chave]
        self.pipe.unload_lora_weights()
        if m['arquivo'] is not None:
            self.pipe.load_lora_weights(str(m['arquivo'].parent),
                                        weight_name=m['arquivo'].name)
        prompt = m.get('prompt', self.app.PROMPT)
        saidas = []
        for n in DESENHOS:
            ctrl = self.app.para_controlnet(esb[n])
            for s in SEMENTES:
                saidas.append((n, s, self.pipe(
                    prompt, image=ctrl, negative_prompt=self.app.NEGATIVO,
                    num_inference_steps=28, guidance_scale=7.5,
                    controlnet_conditioning_scale=0.7,
                    generator=torch.Generator('cuda').manual_seed(s)).images[0]))
        return saidas


@torch.no_grad()
def gerar_pix2pix(arquivo, esb):
    from pix2pix_data import IMG_SIZE, PALETTE_CACHE
    from pix2pix_model import Generator, in_channels_from_state, up_mode_from_state
    from pix2pix_predict import fit_sketch, planos_condicionantes, to_pil, to_tensor
    st = torch.load(P2P / arquivo, map_location='cuda')
    g = Generator(in_channels=in_channels_from_state(st), img_size=IMG_SIZE,
                  up_mode=up_mode_from_state(st), skip_drop=st.get('skip_drop', 0)).cuda()
    g.load_state_dict(st['generator'])
    g.train()                  # o pix2pix amostra em modo treino (ver memoria do projeto)
    pals = np.load(PALETTE_CACHE, allow_pickle=True)['palettes']
    saidas = []
    for n in DESENHOS:
        x0 = to_tensor(fit_sketch(Image.open(RAIZ / f'sprites-human-drawn/pokemon{n}.bmp'),
                                  IMG_SIZE))
        for s in SEMENTES:
            torch.manual_seed(s)
            pal = pals[random.Random(s * 100 + n).randrange(len(pals))]
            ex = planos_condicionantes(g.in_channels, pal, [], IMG_SIZE, 24)
            x = x0 if ex is None else torch.cat([x0, ex], 0)
            out = to_pil(g(x.unsqueeze(0).cuda())[0]).convert('RGB')
            saidas.append((n, s, out.resize((LADO, LADO), Image.LANCZOS)))
    del g
    torch.cuda.empty_cache()
    return saidas


# ------------------------------------------------------------------ referencia
def artes_teste_e_treino():
    from helper import DIR_ARTWORK_PROCESSED
    from lora_train import SEMENTE_ARTWORK, TESTE_ARTWORK
    todos = sorted(Path(DIR_ARTWORK_PROCESSED).glob('gen*/*.png'))
    ordem = list(range(len(todos)))
    random.Random(SEMENTE_ARTWORK).shuffle(ordem)
    teste = [todos[i] for i in ordem[:TESTE_ARTWORK]]
    treino = [todos[i] for i in ordem[TESTE_ARTWORK:]]
    return teste, treino


def miniatura(saidas):
    """Faixa com a semente 0 dos desenhos 1, 4, 7 e 10, em data URI."""
    escolha = [im for n, s, im in saidas if s == 0 and n in (1, 4, 7, 10)]
    t = 112
    faixa = Image.new('RGB', (t * len(escolha), t), (255, 255, 255))
    for i, im in enumerate(escolha):
        faixa.paste(im.resize((t, t), Image.LANCZOS), (i * t, 0))
    buf = io.BytesIO()
    faixa.save(buf, 'WEBP', quality=80)
    return 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ html
def html(res):
    linhas = [r for r in res.values() if r.get('versao') == VERSAO and not r.get('referencia')]
    ref = next((r for r in res.values() if r.get('referencia')), None)
    for chave in ('fidelidade', 'kid', 'fundo'):
        ordem = sorted(linhas, key=lambda r: r[chave])
        for i, r in enumerate(ordem):
            r['posto_' + chave] = i + 1
    for r in linhas:
        r['media'] = (r['posto_fidelidade'] + r['posto_kid'] + r['posto_fundo']) / 3
    # Ordena pelo ESTILO. A media de postos (v2) punha o pix2pix no topo: ele decalca o
    # esboco (fidelidade 0,4 px) e nunca tem fundo, mas nao parece arte oficial nenhuma.
    linhas.sort(key=lambda r: r['kid'])

    def celula(r, chave, fmt):
        melhor = r['posto_' + chave] <= 3
        return (f'<td class="num{" top" if melhor else ""}">{fmt(r[chave])}'
                f'<small>#{r["posto_" + chave]}</small></td>')

    corpo = []
    for i, r in enumerate(linhas, 1):
        corpo.append(
            f'<tr><td class="pos">{i}</td>'
            f'<td><b>{r["rotulo"]}</b><span class="fam">{r["familia"]}</span>'
            f'<div class="nota">{r["nota"]}</div></td>'
            + celula(r, 'fidelidade', lambda v: f'{v:.1f} px')
            + celula(r, 'kid', lambda v: f'{v:.1f}')
            + celula(r, 'fundo', lambda v: f'{v * 100:.0f}%')
            + f'<td><img src="{r["miniatura"]}" alt="amostras de {r["rotulo"]}"></td></tr>')
    ref_txt = ''
    if ref:
        ref_txt = (f'<p class="ref"><b>Referência — artes oficiais reais</b> (40 do treino contra '
                   f'as 300 de teste, 4 sorteios): KID {ref["kid"]:.1f} ± {ref["kid_dp"]:.1f}, '
                   f'fundo {ref["fundo"] * 100:.0f}%. É o piso do KID: abaixo disso não há '
                   f'o que ganhar, e o ± é o tamanho do ruído da medida — diferenças menores '
                   f'que ele entre dois modelos não querem dizer nada.</p>')
    return f'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leaderboard de Modelos</title>
<style>
:root {{ --fundo:#f4f3ef; --papel:#fcfcfb; --linha:#e0dfd9; --tinta:#16150f; --tinta2:#5c5a50;
  --tinta3:#8d8a7e; --acento:#2a78d6; --top:#e6f4ec; --topt:#146c46; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --fundo:#16150f; --papel:#1f1e18; --linha:#34322a; --tinta:#eceae2; --tinta2:#b3b0a4;
  --tinta3:#8d8a7e; --acento:#6aa6ec; --top:#173527; --topt:#7fd6a8; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--fundo); color:var(--tinta);
  font:14px/1.5 ui-sans-serif, system-ui, "Segoe UI", sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 16px 60px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} .sub {{ color:var(--tinta3); margin:0 0 20px; }}
.rolo {{ overflow-x:auto; background:var(--papel); border:1px solid var(--linha); border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; min-width:860px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid var(--linha); text-align:left; vertical-align:middle; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--tinta3); font-weight:600; }}
tr:last-child td {{ border-bottom:none; }}
.pos {{ font-weight:700; font-size:16px; color:var(--tinta3); width:36px; }}
.fam {{ margin-left:8px; font-size:11px; color:var(--tinta3); border:1px solid var(--linha);
  border-radius:99px; padding:1px 7px; white-space:nowrap; }}
.nota {{ font-size:12px; color:var(--tinta2); }}
.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.num small {{ display:block; font-size:11px; color:var(--tinta3); }}
.num.top {{ background:var(--top); color:var(--topt); font-weight:600; }}
td img {{ display:block; height:56px; border-radius:4px; border:1px solid var(--linha); }}
.ref {{ background:var(--papel); border:1px solid var(--linha); border-radius:10px; padding:12px 14px; }}
h2 {{ font-size:16px; margin:28px 0 8px; }} li {{ margin-bottom:6px; }}
</style></head><body><main>
<h1>Leaderboard de Modelos</h1>
<p class="sub">Atualizado em {date.today().strftime("%d/%m/%Y")} · {len(linhas)} modelos ·
10 desenhos humanos × 4 sementes = 40 saídas cada · todas as métricas: menor é melhor ·
verde = top 3 da coluna</p>
<div class="rolo"><table>
<thead><tr><th>#</th><th>Modelo</th><th>Fidelidade ao esboço</th><th>Estilo (KID×1000)</th>
<th>Com fundo</th><th>Amostras (desenhos 1, 4, 7, 10)</th></tr></thead>
<tbody>{"".join(corpo)}</tbody></table></div>
{ref_txt}
<h2>Como ler</h2>
<ul>
<li><b>Posição</b> = ordem pelo <b>estilo</b>, o objetivo do projeto. As outras duas colunas são restrições: fidelidade abaixo de ~5 px já é obedecer ao desenho. O pix2pix tem fidelidade quase perfeita porque <i>decalca</i> o traço, e por isso a média dos postos o punha injustamente no topo.</li>
<li><b>Fidelidade ao esboço</b>: distância média (px, quadro de 512) de cada traço do desenho
até a borda mais próxima na saída. Mede se a forma obedece ao desenho. Não penaliza detalhe
interno acrescentado.</li>
<li><b>Estilo</b>: KID contra as 300 artes oficiais de teste, que nenhum modelo viu, no espaço
do DINOv2. Mede “parece arte oficial”. Modelos de sprite são julgados contra artwork de
propósito: o objetivo do projeto é acabamento de arte oficial.</li>
<li><b>Com fundo</b>: fração das saídas com mais de 30% da borda não-branca.</li>
<li>Saídas cruas, sem remoção de fundo nem pixelização. pix2pix: paleta sorteada do dataset,
sem rabisco, saída de 256 px ampliada para 512.</li>
<li>O ControlNet usa força 0,7, 28 passos e CFG 7,5, os padrões do app.</li>
</ul>
</main></body></html>'''


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refazer', default=None)
    ap.add_argument('--so-html', action='store_true')
    args = ap.parse_args()

    SAIDA.mkdir(exist_ok=True)
    res = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    if args.refazer:
        for k in list(res):
            if args.refazer in ('todos', k):
                del res[k]

    if not args.so_html:
        pend = [m for m in MODELOS if res.get(m[0], {}).get('versao') != VERSAO]
        precisa_ref = res.get('__ref__', {}).get('versao') != VERSAO
        if pend or precisa_ref:
            dino = Dino()
            teste, treino = artes_teste_e_treino()
            f_teste = dino([Image.open(p) for p in teste])
            if precisa_ref:
                kids, fundos = [], []
                for i in range(4):
                    amostra = random.Random(i).sample(treino, 40)
                    ims = [Image.open(p) for p in amostra]
                    kids.append(kid(dino(ims), f_teste) * 1000)
                    fundos.append(np.mean([com_fundo(im) for im in ims]))
                res['__ref__'] = {'versao': VERSAO, 'referencia': True,
                                  'kid': float(np.mean(kids)), 'kid_dp': float(np.std(kids)),
                                  'fundo': float(np.mean(fundos))}
                CACHE.write_text(json.dumps(res, indent=1), encoding='utf-8')
            esb = esbocos()
            dif = None
            for mid, rot, fam, fonte, nota in pend:
                t0 = time.time()
                if fam == 'pix2pix':
                    saidas = gerar_pix2pix(fonte, esb)
                else:
                    dif = dif or Difusao()
                    saidas = dif.gerar(fonte, esb)
                ims = [im for _, _, im in saidas]
                r = {'versao': VERSAO, 'rotulo': rot, 'familia': fam, 'nota': nota,
                     'fidelidade': float(np.mean([fidelidade(esb[n], im) for n, _, im in saidas])),
                     'kid': kid(dino(ims), f_teste) * 1000,
                     'fundo': float(np.mean([com_fundo(im) for im in ims])),
                     'miniatura': miniatura(saidas)}
                res[mid] = r
                CACHE.write_text(json.dumps(res, indent=1), encoding='utf-8')
                print(f'{rot:32s} fid {r["fidelidade"]:5.1f}  kid {r["kid"]:6.1f}  '
                      f'fundo {r["fundo"] * 100:3.0f}%  ({time.time() - t0:.0f} s)', flush=True)

    (SAIDA / 'index.html').write_text(html(res), encoding='utf-8')
    print('->', SAIDA / 'index.html')


if __name__ == '__main__':
    main()
