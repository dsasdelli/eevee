"""Servidor local do estudio de sprites: desenhe no navegador, gere o Pokemon.

    python app.py            e abra http://localhost:7860

POR QUE http.server DA BIBLIOTECA PADRAO

E uma aplicacao local, de um usuario, conversando com uma GPU que so atende um pedido
por vez. Framework web aqui acrescentaria dependencia sem resolver problema nenhum.

O CUSTO QUE MANDA NO DESENHO DO SERVIDOR

Carregar o Stable Diffusion leva ~15 s e ocupa ~3,5 GB de VRAM. Isso acontece UMA vez,
na subida, e o pipeline fica vivo entre pedidos -- se fosse por requisicao, cada
desenho levaria 20 s em vez de 5. Trocar de LoRA, ao contrario, custa menos de um
segundo, entao da para alternar entre os modelos a cada pedido sem incomodo; mesmo
assim o servidor lembra qual esta carregada para nao recarregar a mesma a toa.

A GPU atende um de cada vez, entao as geracoes sao serializadas por um lock. O
servidor e multithread so para a interface continuar respondendo (servir arquivos,
consultar estado) enquanto uma geracao esta em andamento.
"""
import base64
import io
import json
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import montagens
import pecas
import sprite3d
from lora_predict import (GERACAO, NEGATIVO, PROMPT, para_controlnet, pixelizar,
                          tirar_fundo)
from esboco import fit_sketch

RAIZ = Path(__file__).parent
# A interface e o app React de studio/; `npm run build` gera esta pasta.
STUDIO = RAIZ / 'studio' / 'dist'
PORTA = 7860

# Prompt para o modelo BASE, sem LoRA. Tem de ser diferente do das LoRAs: o token
# `pkmnsprite` nao significa nada para ele, e 'pixel art' o faz desenhar uma GRADE
# literal por cima. Medido em 08/11/2026 -- com prompt proprio o SD puro produz
# criaturas otimas; com o prompt emprestado, lixo. Oferecer a opcao com o prompt
# errado seria vender uma comparacao injusta como se fosse o modelo.
PROMPT_PURO = ('official pokemon creature artwork, original monster design, '
               'flat vibrant colors, clean cel shading, bold dark outline, '
               'plain white background, full body, centered, game asset')

# Os modelos oferecidos na interface. A chave e o que o navegador manda.
MODELOS = {
    'gen9': {
        'rotulo': 'Gen 9 (acabamento moderno)',
        'arquivo': RAIZ / 'lora-gen9' / 'lora-step003000.safetensors',
        'nota': 'Treinada so nos 551 sprites da gen9. Bate a nitidez dos sprites reais.',
    },
    'art-aug-16k': {
        'rotulo': 'Artwork oficial (aug 16000)',
        'arquivo': RAIZ / 'lora-art-tudo-aug' / 'lora-step016000.safetensors',
        'nota': 'Mesma LoRA augmentada, treino longo: mais detalhe e ornamento, as vezes '
                'acrescenta elementos que nao estao no desenho.',
    },
    'sug-aug-8k': {
        'rotulo': 'Artwork Sugimori (aug 8000)',
        'arquivo': RAIZ / 'lora-art-sugimori-aug' / 'lora-step008000.safetensors',
        'nota': 'So as 983 artes Sugimori, com augmentacao. Estilo mais pintado e suave, '
                'contorno menos marcado; um pouco mais de fundo colorido que a aug 8000.',
    },
    'sprite': {
        'rotulo': 'Sprite classico (9 geracoes)',
        'arquivo': RAIZ / 'lora-pkmnsprite' / 'lora-step008000.safetensors',
        'nota': 'Treinada nas nove geracoes. Pixel art mais dura, cor mais sobria.',
    },
    'puro': {
        'rotulo': 'Sem LoRA (SD puro)',
        'arquivo': None,
        'prompt': PROMPT_PURO,
        'nota': 'So o modelo base, com prompt proprio. Mais vistoso e saturado, '
                'menos parecido com sprite de jogo.',
    },
}


# Quantas imagens CRUAS ficam guardadas para aprimoramento. O cartao do navegador
# mostra o sprite ja reduzido e quantizado; aprimorar a partir dele jogaria fora o
# detalhe da geracao de 512. Entao o servidor guarda a imagem crua e o navegador so
# manda o id de volta -- tambem evita trafegar PNG grande nas duas direcoes.
#
# 64 e nao 24 porque a interface mantem uma ARVORE de aprimoramentos e o usuario pode
# voltar a qualquer no: um galho antigo que saisse do cache viraria um beco sem saida.
# Cada entrada e uma imagem 512x512 RGB, ~0,8 MB; 64 delas custam ~50 MB de RAM.
CACHE_MAX = 64


class Estudio:
    """Dono do pipeline e do lock da GPU."""

    def __init__(self):
        self.pipe = None
        self.refino = None
        self.lora_atual = '__nenhuma__'
        self.lock = threading.Lock()
        self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cache = {}        # id -> PIL crua, tamanho GERACAO
        self.ordem = []        # ids por idade, para podar o mais antigo
        self.nomeador = None

    def carregar(self):
        from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                               StableDiffusionControlNetPipeline, UniPCMultistepScheduler)
        print('carregando modelos...', flush=True)
        t0 = time.time()
        cn = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_scribble',
                                             dtype=torch.float16, variant='fp16')
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-v1-5', controlnet=cn, variant='fp16',
            dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.dispositivo)
        self.pipe.set_progress_bar_config(disable=True)
        # Compartilha os pesos com o pipeline principal em vez de carregar de novo.
        self.refino = StableDiffusionControlNetImg2ImgPipeline(**self.pipe.components)
        self.refino.set_progress_bar_config(disable=True)
        # Nome inventado (nomes.py). Opcional: se falhar, o app gera sem nome.
        try:
            from nomes import Nomeador
            self.nomeador = Nomeador()
        except Exception as e:
            print(f'AVISO: sem nomes ({type(e).__name__}: {e})', flush=True)
        print(f'pronto em {time.time() - t0:.0f}s  ({self.dispositivo})', flush=True)

    def _trocar_lora(self, chave):
        if chave == self.lora_atual:
            return
        self.pipe.unload_lora_weights()
        arq = MODELOS[chave]['arquivo']
        if arq is not None:
            if not arq.exists():
                raise FileNotFoundError(f'LoRA nao encontrada: {arq}')
            self.pipe.load_lora_weights(str(arq.parent), weight_name=arq.name)
        self.lora_atual = chave

    def _guardar(self, img):
        ident = f'{int(time.time() * 1000):x}{len(self.ordem):02x}'
        self.cache[ident] = img
        self.ordem.append(ident)
        while len(self.ordem) > CACHE_MAX:
            self.cache.pop(self.ordem.pop(0), None)
        return ident

    def gerar(self, esboco, p, base_id=None):
        """esboco: PIL em tons de cinza, traco escuro em fundo claro.

        Com base_id, parte de uma imagem ja gerada em vez do ruido: e img2img, e a
        forca do refino decide quanto o modelo pode mudar. Abaixo de ~0,25 quase nada
        acontece; acima de ~0,6 vira geracao nova e o bicho original se perde.
        """
        base = self.cache.get(base_id) if base_id else None
        if base_id and base is None:
            raise KeyError('a imagem base nao esta mais em cache; gere de novo')

        with self.lock:
            self._trocar_lora(p['modelo'])
            enquadrado = fit_sketch(esboco, GERACAO)
            controle = para_controlnet(enquadrado)
            ctrl_hires = para_controlnet(enquadrado, p['hires_lado']) if p['hires'] else None

            saidas = []
            for k in range(p['amostras']):
                semente = p['semente'] + k
                g = torch.Generator(self.dispositivo).manual_seed(semente)
                if base is not None:
                    # Sem ControlNet o esboco nao e reimposto e a criatura pode evoluir;
                    # com ele, o aprimoramento fica preso a forma desenhada. Os dois sao
                    # uteis, entao quem decide e a interface.
                    #
                    # O controle e montado no TAMANHO DA BASE, nao em GERACAO fixo: uma
                    # imagem gerada com hires foi guardada em 768, e usar controle de
                    # 512 sobre base de 768 estourava no ControlNet ("size of tensor a
                    # (96) must match tensor b (64)" -- 96*8=768 contra 64*8=512).
                    lado_base = base.size[0]
                    img = self.refino(
                        p['prompt'], image=base,
                        control_image=(para_controlnet(enquadrado, lado_base)
                                       if p['manter_traco']
                                       else Image.new('RGB', (lado_base, lado_base), 'BLACK')),
                        negative_prompt=p['negativo'], strength=p['refino'],
                        num_inference_steps=p['passos'], guidance_scale=p['cfg'],
                        controlnet_conditioning_scale=p['forca'] if p['manter_traco'] else 0.0,
                        generator=g).images[0]
                else:
                    img = self.pipe(p['prompt'], image=controle, negative_prompt=p['negativo'],
                                    num_inference_steps=p['passos'], guidance_scale=p['cfg'],
                                    controlnet_conditioning_scale=p['forca'],
                                    generator=g).images[0]
                if p['hires']:
                    grande = img.resize((p['hires_lado'], p['hires_lado']), Image.LANCZOS)
                    img = self.refino(p['prompt'], image=grande, control_image=ctrl_hires,
                                      negative_prompt=p['negativo'], strength=p['hires'],
                                      num_inference_steps=p['passos'], guidance_scale=p['cfg'],
                                      controlnet_conditioning_scale=p['forca'],
                                      generator=torch.Generator(self.dispositivo).manual_seed(semente)
                                      ).images[0]
                if p['fundo_branco']:
                    img = tirar_fundo(img)
                ident = self._guardar(img)      # guarda a CRUA, antes de reduzir
                final = (pixelizar(img, p['lado'], p['cores']) if p['pixelizar']
                         else img.resize((p['lado'], p['lado']), Image.BOX))
                saida = {'imagem': para_dataurl(final), 'semente': semente, 'id': ident}
                # Aprimoramento nao renomeia: o nome e do bicho, o navegador herda o da base.
                if self.nomeador is not None and base is None:
                    try:
                        saida.update(self.nomeador.nomear(img, semente))
                    except Exception:
                        traceback.print_exc()
                saidas.append(saida)
            return saidas


ESTUDIO = Estudio()


def para_dataurl(im):
    buf = io.BytesIO()
    im.convert('RGB').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def de_dataurl(s):
    """dataURL do canvas -> PIL em tons de cinza, fundo branco.

    O canvas chega em RGBA e as partes nao desenhadas sao TRANSPARENTES, nao brancas.
    Converter direto para 'L' leria transparente como PRETO e a folha inteira viraria
    traco. Por isso o alfa e achatado sobre branco antes de tudo.
    """
    bruto = base64.b64decode(s.split(',', 1)[1])
    im = Image.open(io.BytesIO(bruto)).convert('RGBA')
    fundo = Image.new('RGBA', im.size, 'WHITE')
    fundo.paste(im, (0, 0), im)
    return fundo.convert('L')


def limitar(v, lo, hi, padrao):
    """Toda entrada que vem do navegador passa por aqui: valores fora de faixa viram
    erro de CUDA la na frente, que e muito pior de diagnosticar."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return padrao
    return max(lo, min(hi, v))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STUDIO), **kw)

    def translate_path(self, path):
        caminho = path.split('?', 1)[0].split('#', 1)[0]
        # /pecas/ serve os PNGs da biblioteca de pecas (pecas.py) como arquivos.
        for prefixo, pasta in (('/pecas', pecas.PECAS),):
            if caminho == prefixo or caminho.startswith(prefixo + '/'):
                original = self.directory
                self.directory = str(pasta)
                try:
                    return super().translate_path(path[len(prefixo):] or '/')
                finally:
                    self.directory = original
        return super().translate_path(path)

    def _png(self, arquivo):
        dados = arquivo.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(dados)))
        self.send_header('Cache-Control', 'max-age=86400')   # os contornos nao mudam
        self.end_headers()
        self.wfile.write(dados)

    def _corpo(self):
        n = int(self.headers.get('Content-Length', 0))
        if n > 20 * 1024 * 1024:
            raise ValueError('pedido grande demais')
        return json.loads(self.rfile.read(n))

    def do_DELETE(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        ident = (parse_qs(u.query).get('id') or [''])[0]
        if u.path == '/api/montagens':
            try:
                montagens.apagar(ident)
            except KeyError:
                return self._json(404, {'erro': f'montagem nao encontrada: {ident}'})
            return self._json(200, {'montagens': montagens.listar()})
        if u.path != '/api/pecas':
            return self._json(404, {'erro': 'rota desconhecida'})
        try:
            pecas.apagar_peca(ident)
        except KeyError:
            return self._json(404, {'erro': f'peca nao encontrada: {ident}'})
        return self._json(200, {'pecas': pecas.ler_indice()})

    def log_message(self, fmt, *args):
        if '/api/' in (self.path or ''):
            print(f'  {self.command} {self.path}', flush=True)

    def _json(self, codigo, corpo):
        dados = json.dumps(corpo).encode()
        self.send_response(codigo)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        # O app React morou em /montar/ enquanto convivia com a pagina antiga; links e
        # favoritos daquela epoca caem na raiz.
        if self.path == '/montar' or self.path.startswith('/montar/'):
            self.send_response(301)
            self.send_header('Location', '/')
            self.end_headers()
            return
        # O GLB e servido por GET, e nao devolvido em base64 num JSON, por dois
        # motivos: o navegador pode transmiti-lo direto para o visualizador, e o
        # mesmo endereco serve de link de download sem duplicar quase 1 MB na
        # memoria da pagina.
        if self.path.startswith('/api/3d'):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            ident = (q.get('id') or [''])[0]
            im = ESTUDIO.cache.get(ident)
            if im is None:
                return self._json(410, {'erro': 'a imagem nao esta mais em cache'})
            try:
                espessura = float((q.get('espessura') or ['0.22'])[0])
            except ValueError:
                espessura = sprite3d.ESPESSURA
            espessura = max(0.05, min(0.45, espessura))
            try:
                malha = sprite3d.construir(im, espessura=espessura)
                dados = malha.export(file_type='glb')
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {'erro': f'{type(e).__name__}: {e}'})
            self.send_response(200)
            self.send_header('Content-Type', 'model/gltf-binary')
            self.send_header('Content-Length', str(len(dados)))
            self.send_header('Content-Disposition', f'attachment; filename="pokemon-{ident}.glb"')
            self.end_headers()
            self.wfile.write(dados)
            return

        if self.path.startswith('/api/montagens'):
            from urllib.parse import parse_qs, urlparse
            ident = (parse_qs(urlparse(self.path).query).get('id') or [''])[0]
            if not ident:
                return self._json(200, {'montagens': montagens.listar()})
            try:
                return self._json(200, {'id': ident, **montagens.ler(ident)})
            except KeyError:
                return self._json(404, {'erro': f'montagem nao encontrada: {ident}'})
        if self.path == '/api/pokemons':
            return self._json(200, {'pokemons': pecas.listar_pokemons()})
        if self.path == '/api/pecas':
            return self._json(200, {'categorias': pecas.CATEGORIAS, 'pecas': pecas.ler_indice()})
        if self.path.startswith('/api/imagem'):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            f = pecas.caminho_imagem(*[(q.get(k) or [''])[0] for k in ('tipo', 'geracao', 'nome')])
            if f is None:
                return self._json(404, {'erro': 'imagem nao encontrada'})
            return self._png(f)

        if self.path == '/api/estado':
            return self._json(200, {
                'dispositivo': ESTUDIO.dispositivo,
                'pronto': ESTUDIO.pipe is not None,
                'modelos': [{'chave': k, 'rotulo': v['rotulo'], 'nota': v['nota'],
                             'prompt': v.get('prompt') or PROMPT,
                             'disponivel': v['arquivo'] is None or v['arquivo'].exists()}
                            for k, v in MODELOS.items()],
                'prompt': PROMPT,
                'negativo': NEGATIVO,
            })
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/segmentar':
            # Import tardio: o SAM (~375 MB) so carrega se alguem usar o recorte por clique.
            import segmentar
            try:
                r = self._corpo()
                poli, score = segmentar.poligono(r.get('geracao'), r.get('nome'),
                                                 r.get('positivos') or [], r.get('negativos') or [])
            except ValueError as e:
                return self._json(400, {'erro': str(e)})
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {'erro': f'{type(e).__name__}: {e}'})
            return self._json(200, {'poligono': poli, 'score': score})
        if self.path == '/api/montagens':
            try:
                r = self._corpo()
                ident = montagens.salvar(r.get('nome'), r.get('cena'), r.get('miniatura'))
            except ValueError as e:
                return self._json(400, {'erro': str(e)})
            return self._json(200, {'id': ident, 'montagens': montagens.listar()})
        if self.path == '/api/pecas':
            try:
                r = self._corpo()
                item = pecas.salvar_peca(r.get('geracao'), r.get('nome'),
                                         r.get('categoria'), r.get('poligono'))
            except ValueError as e:
                return self._json(400, {'erro': str(e)})
            return self._json(200, {'peca': item, 'pecas': pecas.ler_indice()})
        if self.path != '/api/gerar':
            return self._json(404, {'erro': 'rota desconhecida'})
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n > 20 * 1024 * 1024:
                return self._json(413, {'erro': 'desenho grande demais'})
            req = json.loads(self.rfile.read(n))

            modelo = req.get('modelo', 'gen9')
            if modelo not in MODELOS:
                return self._json(400, {'erro': f'modelo desconhecido: {modelo}'})

            p = {
                'modelo': modelo,
                'forca': limitar(req.get('forca'), 0.1, 1.0, 0.7),
                'amostras': int(limitar(req.get('amostras'), 1, 4, 1)),
                'passos': int(limitar(req.get('passos'), 10, 50, 28)),
                'cfg': limitar(req.get('cfg'), 1.0, 15.0, 7.5),
                'lado': int(limitar(req.get('lado'), 64, 512, 256)),
                'cores': int(limitar(req.get('cores'), 8, 256, 64)),
                'pixelizar': bool(req.get('pixelizar', True)),
                'fundo_branco': bool(req.get('fundo_branco', True)),
                'hires': limitar(req.get('hires'), 0.1, 0.6, 0.35) if req.get('hires') else None,
                # Faixa util do refino: abaixo de 0,25 quase nada muda, acima de 0,6
                # vira geracao nova e o bicho original se perde.
                'refino': limitar(req.get('refino'), 0.15, 0.75, 0.4),
                'manter_traco': bool(req.get('manter_traco', True)),
                'hires_lado': 768,
                # Cada modelo tem seu prompt: ver PROMPT_PURO.
                'prompt': (req.get('prompt')
                           or MODELOS[modelo].get('prompt') or PROMPT).strip()[:600],
                'negativo': (req.get('negativo') or NEGATIVO).strip()[:600],
                'semente': int(req.get('semente') if req.get('semente') not in (None, '')
                               else torch.randint(0, 2**31 - 1, ()).item()),
            }
            esboco = de_dataurl(req['desenho'])
            # Folha em branco: fit_sketch nao acha caixa delimitadora e o resultado
            # seria uma criatura puramente alucinada, sem relacao com pedido nenhum.
            if np.asarray(esboco).min() > 200:
                return self._json(400, {'erro': 'o desenho esta vazio'})

            t0 = time.time()
            try:
                saidas = ESTUDIO.gerar(esboco, p, base_id=req.get('base_id') or None)
            except KeyError as e:
                return self._json(410, {'erro': str(e.args[0])})
            return self._json(200, {'saidas': saidas, 'ms': int((time.time() - t0) * 1000),
                                    'parametros': {k: v for k, v in p.items()
                                                   if k not in ('prompt', 'negativo')}})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {'erro': f'{type(e).__name__}: {e}'})


class Servidor(ThreadingHTTPServer):
    # O padrao do http.server e allow_reuse_address = 1, e no Windows isso deixa um
    # SEGUNDO processo subir na mesma porta sem erro nenhum. Os pedidos passam a cair
    # ora num, ora noutro, e o codigo recem-editado parece nao ter efeito -- custou um
    # diagnostico errado em 08/11/2026. Melhor falhar alto.
    allow_reuse_address = False
    daemon_threads = True


def main():
    if not (STUDIO / 'index.html').exists():
        raise SystemExit(f'interface nao compilada: falta {STUDIO / "index.html"}. '
                         f'Rode `npm install` e `npm run build` dentro de studio/.')
    faltando = [v['arquivo'] for v in MODELOS.values()
                if v['arquivo'] is not None and not v['arquivo'].exists()]
    for f in faltando:
        print(f'AVISO: LoRA ausente, o modelo ficara desabilitado -- {f}')
    try:
        srv = Servidor(('127.0.0.1', PORTA), Handler)
    except OSError as e:
        raise SystemExit(f'porta {PORTA} ja esta em uso ({e}). Encerre o outro '
                         f'servidor antes de subir este.')
    # So carrega os pesos DEPOIS de garantir a porta: 15 s de carregamento para
    # descobrir que nao da para servir seria desperdicio.
    ESTUDIO.carregar()
    print(f'{chr(10)}  Pokemon Studio em http://localhost:{PORTA}   (ctrl+c para parar){chr(10)}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('encerrando')


if __name__ == '__main__':
    main()
