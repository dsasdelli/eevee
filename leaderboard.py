"""Leaderboard dos modelos de ARTE: a mesma bancada para todos.

    python leaderboard.py              mede o que ainda nao foi medido e reescreve o HTML
    python leaderboard.py --refazer X  mede de novo o modelo X (ou 'todos')
    python leaderboard.py --so-html    so reescreve leaderboard/index.html
    python leaderboard.py --so X       mede so o modelo X (teste rapido)

Para entrar um modelo novo, acrescente uma linha em MODELOS e rode de novo: os ja
medidos ficam no cache (leaderboard/resultados.json) e nao sao refeitos.

A BANCADA (v4, 19/11/2026)

Teste: as artes das especies de splits/teste-especies.txt (10% das especies, todas as
artes delas em todos os grupos; nenhuma LoRA viu essas especies em forma nenhuma).

Entrada: o CONTORNO de cada arte de teste (artwork-<grupo>-processed-countours), uma
semente. Sao ~400 entradas independentes; com os 10 desenhos humanos da v3 o KID tinha
desvio de ~110 so pela escolha dos desenhos e nao separava as LoRAs de arte. Os 10
desenhos humanos continuam como coluna a parte (4 sementes), por serem o caso real do
app, mas marcados como ruidosos.

A saida e medida CRUA: sem tirar_fundo e sem pixelizar.

Metricas, todas "menor e melhor":

1. FIDELIDADE ao contorno de entrada (px): distancia media de cada pixel de traco da
   entrada ate a borda mais proxima da saida, extraida com o limiar do artwork-contours.py.

2. ESTILO: KID x 1000 no espaco do DINOv2, uma coluna por REFERENCIA (grupos). Na coluna
   G entram as saidas das entradas de G contra as artes de teste de G, entao as colunas
   combinadas pesam cada grupo pelo seu tamanho. O +- e o desvio do KID por subamostragem
   das entradas (metade, 200 vezes, escalado para o tamanho cheio), com os MESMOS indices
   para todos os modelos: a comparacao entre dois modelos e pareada. Um modelo fica
   "empatado com o 1o" se em 5% ou mais das subamostras ele vence o 1o.

3. FUNDO: fracao das saidas cuja borda (faixa de 8 px) tem mais de 30% de pixels
   nao-brancos.

A posicao e a ordem pelo KID sugimori+outros. A v3 (desenhos humanos como entrada, teste
de 100 artes por grupo) esta guardada em leaderboard/v3/.
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
FEATURES = SAIDA / 'features'
VERSAO = 4                        # muda se a bancada mudar: invalida o cache
DESENHOS = list(range(1, 11))
SEMENTES_DESENHOS = [0, 1, 2, 3]
SEMENTE_CONTORNOS = 0
LADO = 512
SUBAMOSTRAS = 200

# Contra o que o estilo e medido: nome da coluna -> grupos cujas entradas e artes entram.
REFERENCIAS = {'sugimori+outros+anime': ['sugimori', 'outros', 'anime'],
               'sugimori+outros': ['sugimori', 'outros'], 'sugimori': ['sugimori'],
               'outros': ['outros'], 'anime': ['anime']}
PRINCIPAL = 'sugimori+outros'             # a coluna que ordena a tabela

# (id, rotulo, familia, fonte, nota). fonte: caminho de um .safetensors, ou chave do
# app.MODELOS (so o 'puro', que tem prompt proprio). A familia e o --dados do treino.
# Todas treinadas no split por especie de 19/11/2026; meio e fim de cada treino.
MODELOS = [
    ('sugimori-outros-aug-8k', 'Sugimori + outros (aug 8000)', 'sugimori+outros',
     'lora-sugimori-outros-aug/lora-step008000.safetensors', 'meio do treino'),
    ('sugimori-outros-aug-16k', 'Sugimori + outros (aug 16000)', 'sugimori+outros',
     'lora-sugimori-outros-aug/lora-step016000.safetensors', 'fim do treino'),
    ('sugimori-aug-6k', 'Sugimori (aug 6000)', 'sugimori',
     'lora-sugimori-aug/lora-step006000.safetensors', 'meio do treino'),
    ('sugimori-aug-12k', 'Sugimori (aug 12000)', 'sugimori',
     'lora-sugimori-aug/lora-step012000.safetensors', 'fim do treino'),
    ('anime-aug-4k', 'Anime (aug 4000)', 'anime',
     'lora-anime-aug/lora-step004000.safetensors', 'meio do treino'),
    ('anime-aug-8k', 'Anime (aug 8000)', 'anime',
     'lora-anime-aug/lora-step008000.safetensors', 'fim do treino'),
    ('sugimori-outros-anime-aug-6k', 'Sugimori + outros + anime (aug 6000)', 'sugimori+outros+anime',
     'lora-sugimori-outros-anime-aug/lora-step006000.safetensors', 'meio do treino'),
    ('sugimori-outros-anime-aug-12k', 'Sugimori + outros + anime (aug 12000)', 'sugimori+outros+anime',
     'lora-sugimori-outros-anime-aug/lora-step012000.safetensors', 'fim do treino'),
    ('puro', 'Sem LoRA (SD 1.5 puro)', 'base',
     'puro', 'so o modelo base, prompt proprio'),
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



# ------------------------------------------------------------------ entradas
def entradas():
    """[(grupo, arte de teste, contorno dela)] na ordem de GRUPOS e de teste_do_grupo."""
    from helper import GRUPOS, dir_contornos
    from lora_train import teste_do_grupo
    saida = []
    for g in GRUPOS:
        for arte in teste_do_grupo(g):
            c = RAIZ / dir_contornos(g) / arte.parent.name / arte.name
            if not c.exists():
                raise SystemExit(f'{c} nao existe -- rode artwork-contours.py --grupo {g}')
            saida.append((g, arte, c))
    return saida


def esbocos_humanos():
    from esboco import fit_sketch
    return [fit_sketch(Image.open(RAIZ / f'sprites-human-drawn/pokemon{n}.bmp'), LADO)
            for n in DESENHOS]


def esbocos_contornos(ent):
    from esboco import fit_sketch
    return [fit_sketch(Image.open(c), LADO) for _, _, c in ent]


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

    def carregar(self, chave):
        # fonte = chave do app.MODELOS, ou caminho direto de um .safetensors
        m = {'arquivo': RAIZ / chave} if chave.endswith('.safetensors') else self.app.MODELOS[chave]
        self.pipe.unload_lora_weights()
        if m['arquivo'] is not None:
            self.pipe.load_lora_weights(str(m['arquivo'].parent), weight_name=m['arquivo'].name)
        self.prompt = m.get('prompt', self.app.PROMPT)

    def gerar(self, esbocos, sementes):
        """Uma saida por (esboco, semente), nesta ordem."""
        saidas = []
        for esb in esbocos:
            ctrl = self.app.para_controlnet(esb)
            for s in sementes:
                saidas.append(self.pipe(
                    self.prompt, image=ctrl, negative_prompt=self.app.NEGATIVO,
                    num_inference_steps=28, guidance_scale=7.5,
                    controlnet_conditioning_scale=0.7,
                    generator=torch.Generator('cuda').manual_seed(s)).images[0])
        return saidas


# ------------------------------------------------------------------ KID por coluna
def subindices(n, coluna):
    """SUBAMOSTRAS subconjuntos de metade de range(n), sem reposicao, fixos por coluna:
    os mesmos para todo modelo, o que torna pareada a comparacao entre dois modelos.
    (Sem reposicao porque linha repetida infla o KID nao-viesado.)"""
    rng = np.random.default_rng(1000 + list(REFERENCIAS).index(coluna))
    return [rng.choice(n, n // 2, replace=False) for _ in range(SUBAMOSTRAS)]


def kids_do_modelo(f, grupos_ent, f_teste):
    """{coluna: (kid, dp, subamostras)} das saidas f das entradas cujos grupos sao grupos_ent."""
    grupos_ent = np.array(grupos_ent)
    saida = {}
    for nome, gs in REFERENCIAS.items():
        fc = f[np.isin(grupos_ent, gs)]
        sub = [kid(fc[i], f_teste[nome]) * 1000 for i in subindices(len(fc), nome)]
        # Variancia do KID cai com 1/n: o dp da metade vezes 1/sqrt(2) aproxima o do cheio.
        saida[nome] = (kid(fc, f_teste[nome]) * 1000, float(np.std(sub)) / np.sqrt(2),
                       [round(v, 2) for v in sub])
    return saida


def aplica_kids(r, f, grupos_ent, f_teste):
    k = kids_do_modelo(f, grupos_ent, f_teste)
    r['kid'] = {n: v[0] for n, v in k.items()}
    r['kid_dp'] = {n: v[1] for n, v in k.items()}
    r['kid_sub'] = {n: v[2] for n, v in k.items()}


def mede_referencias(dino, ent):
    """{coluna: features das artes de teste} e {coluna: linha de referencia}.

    A referencia sao artes REAIS do treino, tantas quanto o teste da coluna, contra o
    teste: o piso do KID. 4 sorteios dao o ruido do piso."""
    from lora_train import caminhos_grupo
    f_arte = dino([Image.open(a) for _, a, _ in ent])
    grupos_ent = np.array([g for g, _, _ in ent])
    f_teste, refs = {}, {}
    for nome, gs in REFERENCIAS.items():
        f_teste[nome] = f_arte[np.isin(grupos_ent, gs)]
        treino = [p for g in gs for p in caminhos_grupo(g)]
        n = len(f_teste[nome])
        kids, fundos = [], []
        for i in range(4):
            ims = [Image.open(p) for p in random.Random(i).sample(treino, n)]
            kids.append(kid(dino(ims), f_teste[nome]) * 1000)
            fundos.append(np.mean([com_fundo(im) for im in ims]))
        refs[nome] = {'n_teste': n, 'kid': float(np.mean(kids)),
                      'kid_dp': float(np.std(kids)), 'fundo': float(np.mean(fundos))}
    return f_teste, refs


def miniatura(ims):
    t = 96
    faixa = Image.new('RGB', (t * len(ims), t), (255, 255, 255))
    for i, im in enumerate(ims):
        faixa.paste(im.resize((t, t), Image.LANCZOS), (i * t, 0))
    buf = io.BytesIO()
    faixa.save(buf, 'WEBP', quality=80)
    return 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode()


def indices_miniatura(ent):
    """As 2 primeiras entradas de cada grupo."""
    vistos, saida = {}, []
    for i, (g, _, _) in enumerate(ent):
        if vistos.get(g, 0) < 2:
            saida.append(i)
            vistos[g] = vistos.get(g, 0) + 1
    return saida


# ------------------------------------------------------------------ html
def html(res):
    linhas = [r for k, r in res.items() if k != '__ref__' and r.get('versao') == VERSAO]
    ref = res.get('__ref__') if res.get('__ref__', {}).get('versao') == VERSAO else None
    # Empate com o 1o, por coluna, pela comparacao pareada nas subamostras.
    for n in REFERENCIAS:
        if not linhas:
            break
        lider = min(linhas, key=lambda r: r['kid'][n])
        a = np.array(lider['kid_sub'][n])
        for r in linhas:
            p = float(np.mean(np.array(r['kid_sub'][n]) <= a)) if r is not lider else 1.0
            r.setdefault('empate', {})[n] = p >= 0.05
            r.setdefault('p_vence', {})[n] = p
    linhas.sort(key=lambda r: r['kid'][PRINCIPAL])

    def kid_celula(r, n):
        e = r['empate'][n]
        posto = 1 + sum(o['kid'][n] < r['kid'][n] for o in linhas)
        return (f'<td class="num{" top" if e else ""}">{r["kid"][n]:.1f}'
                f'<span class="dp"> ±{r["kid_dp"][n]:.0f}</span><small>#{posto}'
                f'{" · empata c/ 1º" if e and posto > 1 else ""}</small></td>')

    corpo = []
    for i, r in enumerate(linhas, 1):
        corpo.append(
            f'<tr><td class="pos">{i}</td>'
            f'<td><b>{r["rotulo"]}</b><span class="fam">{r["familia"]}</span>'
            f'<div class="nota">{r["nota"]}</div></td>'
            f'<td class="num">{r["fidelidade"]:.1f} px</td>'
            + ''.join(kid_celula(r, n) for n in REFERENCIAS)
            + f'<td class="num">{r["fundo"] * 100:.0f}%</td>'
            f'<td class="num ruido">{r["kid_desenhos"]:.0f}<small>fid {r["fid_desenhos"]:.1f} px</small></td>'
            f'<td><img src="{r["miniatura"]}" alt="amostras de {r["rotulo"]}"></td></tr>')
    ref_txt = ''
    if ref:
        itens = ''.join(
            f'<li><b>{n}</b> ({c["n_teste"]} artes de teste): KID {c["kid"]:.1f} ± {c["kid_dp"]:.1f}, '
            f'fundo {c["fundo"] * 100:.0f}%</li>' for n, c in ref['colunas'].items())
        ref_txt = (f'<div class="ref"><b>Referência — artes reais</b> (do treino, tantas quanto o '
                   f'teste da coluna, contra o teste; 4 sorteios). É o piso do KID: nenhum gerador '
                   f'deve ficar abaixo disso.<ul>{itens}</ul></div>')
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
table {{ border-collapse:collapse; width:100%; min-width:1300px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid var(--linha); text-align:left; vertical-align:middle; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--tinta3); font-weight:600; }}
tr:last-child td {{ border-bottom:none; }}
.pos {{ font-weight:700; font-size:16px; color:var(--tinta3); width:36px; }}
.fam {{ margin-left:8px; font-size:11px; color:var(--tinta3); border:1px solid var(--linha);
  border-radius:99px; padding:1px 7px; white-space:nowrap; }}
.nota {{ font-size:12px; color:var(--tinta2); }}
.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.num small {{ display:block; font-size:11px; color:var(--tinta3); }}
.dp {{ font-size:11px; color:var(--tinta3); }}
.num.top {{ background:var(--top); color:var(--topt); font-weight:600; }}
.ruido {{ color:var(--tinta3); }}
td img {{ display:block; height:48px; border-radius:4px; border:1px solid var(--linha); }}
.ref {{ background:var(--papel); border:1px solid var(--linha); border-radius:10px; padding:12px 14px; margin-top:16px; }}
h2 {{ font-size:16px; margin:28px 0 8px; }} li {{ margin-bottom:6px; }}
</style></head><body><main>
<h1>Leaderboard de Modelos</h1>
<p class="sub">Atualizado em {date.today().strftime("%d/%m/%Y")} · {len(linhas)} modelos ·
v4: entrada = contorno de cada arte de teste (espécies que nenhum modelo viu) · menor é melhor ·
verde = 1º ou empatado com ele</p>
<div class="rolo"><table>
<thead><tr><th>#</th><th>Modelo</th><th>Fidelidade</th>
{"".join(f"<th>KID {n}</th>" for n in REFERENCIAS)}
<th>Com fundo</th><th>KID desenhos humanos</th><th>Amostras (2 por grupo · 4 desenhos)</th></tr></thead>
<tbody>{"".join(corpo)}</tbody></table></div>
{ref_txt}
<h2>Como ler</h2>
<ul>
<li><b>Posição</b> = ordem pelo <b>KID sugimori+outros</b>.</li>
<li><b>Teste</b>: 10% das espécies (<code>splits/teste-especies.txt</code>), todas as artes delas em todos os
grupos. Nenhum modelo viu essas espécies em nenhuma forma.</li>
<li><b>KID ×1000</b>, no espaço do DINOv2. Na coluna de um grupo entram as saídas geradas dos contornos daquele
grupo, comparadas com as artes de teste dele; as colunas combinadas pesam cada grupo pelo tamanho.
O <b>±</b> é o desvio pela escolha das entradas. <b>Verde</b> = 1º da coluna ou empatado com ele
(comparação pareada: vence o 1º em 5% ou mais das subamostras).</li>
<li><b>Fidelidade</b>: distância média (px, quadro de 512) de cada traço do contorno de entrada até a
borda mais próxima na saída.</li>
<li><b>KID desenhos humanos</b>: os 10 desenhos de <code>sprites-human-drawn</code> × 4 sementes contra o teste
sugimori+outros. É o caso real do app, mas com só 10 entradas o desvio é de ~±110: não use para desempatar.</li>
<li>Saídas cruas, sem remoção de fundo nem pixelização. ControlNet força 0,7, 28 passos, CFG 7,5.</li>
</ul>
</main></body></html>'''


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refazer', default=None)
    ap.add_argument('--so-html', action='store_true')
    ap.add_argument('--so', default=None, help='mede so este id (teste rapido)')
    args = ap.parse_args()

    SAIDA.mkdir(exist_ok=True)
    res = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    ids = {m[0] for m in MODELOS} | {'__ref__'}
    res = {k: v for k, v in res.items() if k in ids}
    if args.refazer:
        for k in list(res):
            if args.refazer in ('todos', k):
                del res[k]

    if not args.so_html:
        pend = [m for m in MODELOS if res.get(m[0], {}).get('versao') != VERSAO
                and (args.so is None or m[0] == args.so)]
        faltam = [m for m in pend if m[3].endswith('.safetensors') and not (RAIZ / m[3]).exists()]
        for m in faltam:
            print(f'pulado, arquivo ainda nao existe: {m[3]}', flush=True)
        pend = [m for m in pend if m not in faltam]
        precisa_ref = res.get('__ref__', {}).get('versao') != VERSAO
        incompletos = [m for m in MODELOS if m not in pend
                       and res.get(m[0], {}).get('versao') == VERSAO
                       and set(REFERENCIAS) - set(res[m[0]].get('kid', {}))]
        if pend or precisa_ref or incompletos:
            ent = entradas()
            grupos_ent = [g for g, _, _ in ent]
            print(f'{len(ent)} entradas de teste: '
                  + ', '.join(f'{g} {grupos_ent.count(g)}' for g in dict.fromkeys(grupos_ent)), flush=True)
            dino = Dino()
            f_teste, refs = mede_referencias(dino, ent)
            res['__ref__'] = {'versao': VERSAO, 'colunas': refs}
            FEATURES.mkdir(exist_ok=True)
            for m in incompletos:
                aplica_kids(res[m[0]], np.load(FEATURES / f'{m[0]}.npy').astype(np.float32),
                            grupos_ent, f_teste)
            CACHE.write_text(json.dumps(res, indent=1), encoding='utf-8')
            esb_c, esb_h = esbocos_contornos(ent), esbocos_humanos()
            mini = indices_miniatura(ent)
            dif = None
            for mid, rot, fam, fonte, nota in pend:
                t0 = time.time()
                dif = dif or Difusao()
                dif.carregar(fonte)
                out_c = dif.gerar(esb_c, [SEMENTE_CONTORNOS])
                out_h = dif.gerar(esb_h, SEMENTES_DESENHOS)
                f_c, f_h = dino(out_c), dino(out_h)
                np.save(FEATURES / f'{mid}.npy', f_c.astype(np.float16))
                np.save(FEATURES / f'{mid}-desenhos.npy', f_h.astype(np.float16))
                esb_h_rep = [e for e in esb_h for _ in SEMENTES_DESENHOS]
                r = {'versao': VERSAO, 'rotulo': rot, 'familia': fam, 'nota': nota,
                     'fidelidade': float(np.mean([fidelidade(e, o) for e, o in zip(esb_c, out_c)])),
                     'fundo': float(np.mean([com_fundo(o) for o in out_c])),
                     'kid_desenhos': kid(f_h, f_teste[PRINCIPAL]) * 1000,
                     'fid_desenhos': float(np.mean([fidelidade(e, o) for e, o in zip(esb_h_rep, out_h)])),
                     'miniatura': miniatura([out_c[i] for i in mini]
                                            + [out_h[4 * k] for k in (0, 3, 6, 9)])}
                aplica_kids(r, f_c, grupos_ent, f_teste)
                res[mid] = r
                CACHE.write_text(json.dumps(res, indent=1), encoding='utf-8')
                print(f'{rot:40s} fid {r["fidelidade"]:4.1f}  '
                      + '  '.join(f'{n} {r["kid"][n]:6.1f}±{r["kid_dp"][n]:.0f}' for n in REFERENCIAS)
                      + f'  fundo {r["fundo"] * 100:3.0f}%  humanos {r["kid_desenhos"]:.0f}'
                      f'  ({time.time() - t0:.0f} s)', flush=True)

    (SAIDA / 'index.html').write_text(html(res), encoding='utf-8')
    print('->', SAIDA / 'index.html')


if __name__ == '__main__':
    main()
