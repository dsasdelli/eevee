"""Gera sprites a partir de esbocos, usando um gerador treinado por pix2pix_train.py.

Este e o arquivo que transforma o projeto de experimento em ferramenta: o notebook
original nao tinha caminho de inferencia nenhum -- so dava para ver predicoes sobre
o proprio conjunto de teste, dentro do Jupyter.

Modos:

    # um esboco seu (qualquer PNG/JPG com traco escuro em fundo claro)
    python pix2pix_predict.py --input meu_esboco.png --output resultado.png

    # uma pasta inteira de esbocos
    python pix2pix_predict.py --input esbocos/ --output-dir resultados/

    # com cor: --palette diz QUAIS cores, --scribble diz ONDE cada uma vai.
    # Exigido pelos checkpoints de 13 canais (paleta) e 16 (paleta + rabisco).
    python pix2pix_predict.py --input meu_esboco.png         --palette "#141414,#e63f19,#fa8c19,#ffe033"         --scribble 0.50,0.20,#e63f19 --scribble 0.50,0.75,#ffe033 --save-input

    # amostras do conjunto de teste, com o alvo real ao lado para comparar
    python pix2pix_predict.py --from-test 12 --output-dir resultados/

    # metrica no conjunto de teste inteiro (L1 medio), para comparar rodadas
    python pix2pix_predict.py --eval

Por padrao carrega o checkpoint mais recente de
<input-dir>_torch_training_checkpoints/.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from pix2pix_data import (IMG_SIZE, PALETTE_CACHE, PALETTE_K, PATH_INPUT, PATH_REAL,
                          SCRIBBLE_CELL, PokemonPairs, fixed_split, list_pairs,
                          load_palettes)
from pix2pix_model import Generator, in_channels_from_state, up_mode_from_state

SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# Quantos canais cada condicionante ocupa, na ordem em que PokemonPairs.__getitem__
# os concatena: esboco, paleta, rabisco. A inferencia TEM de repetir essa ordem --
# trocar dois blocos nao da erro de forma, so resultado errado em silencio.
CH_ESBOCO, CH_PALETA, CH_RABISCO = 1, PALETTE_K * 3, 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Os tres modos sao mutuamente exclusivos e um deles e obrigatorio: o argparse
    # rejeita sozinho `--input x --eval` ou a ausencia de todos.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', help='arquivo ou pasta de esbocos')
    src.add_argument('--from-test', type=int, metavar='N',
                     help='usa N pares do conjunto de teste (nunca vistos no treino)')
    src.add_argument('--eval', action='store_true',
                     help='calcula o L1 medio sobre o conjunto de teste inteiro')

    p.add_argument('--output', help='arquivo de saida (so com --input apontando para 1 arquivo)')
    p.add_argument('--output-dir', default='predictions')
    p.add_argument('--checkpoint', help='arquivo .pt especifico (default: o mais recente)')
    p.add_argument('--input-dir', default=PATH_INPUT, help='define onde achar o checkpoint')
    p.add_argument('--real-dir', default=PATH_REAL)
    p.add_argument('--img-size', type=int, default=IMG_SIZE)
    p.add_argument('--device', default=None)
    p.add_argument('--no-fit', action='store_true',
                   help='nao recortar/centralizar a entrada; so redimensionar')
    p.add_argument('--deterministic', action='store_true',
                   help='gerador em modo eval (sem dropout). O pix2pix amostra com '
                        'dropout ligado, entao o default reproduz o treino')
    p.add_argument('--batch-size', type=int, default=1,
                   help='usado por --eval. Default 1 DE PROPOSITO: o gerador roda em '
                        'modo treino (ver --deterministic) e o BatchNorm normaliza pela '
                        'estatistica do lote. Com lote > 1 ele mistura pokemon diferentes, '
                        'o que nao e como o modelo foi treinado (batch 1) e distorce a L1 '
                        '-- medido em 08/11/2026: 0,1511 com lote 1 contra 0,5299 com lote 8, '
                        'e a ordem entre modelos chega a inverter. So aumente com '
                        '--deterministic, que desliga essa dependencia.')

    # --- condicionantes de cor, exigidas pelos modelos de 13 e 16 canais ---------
    # So fazem sentido com --input: nos modos --from-test e --eval o PokemonPairs
    # monta paleta e rabisco a partir do alvo real, que ali existe.
    cor = p.add_argument_group(
        'cor (modelos de 13/16 canais)',
        'O esboco diz a FORMA; estas flags dizem a COR. --palette diz QUAIS cores o '
        'bicho tem, --scribble diz ONDE cada uma vai. Sem --scribble o modelo recebe '
        '"nada foi dito" e escolhe sozinho, que e um caso visto em 20%% do treino.')
    cor.add_argument('--palette', metavar='CORES',
                     help='4 cores hex separadas por virgula (#1a1a1a,#e63f19,...), '
                          'ou o caminho de uma imagem de onde extrai-las por k-means, '
                          'ou "random" para sortear uma paleta real de ' + PALETTE_CACHE)
    cor.add_argument('--scribble', action='append', metavar='X,Y,COR', default=[],
                     help='mancha de cor em coordenada fracionaria da imagem: '
                          '"0.5,0.2,#ff0000" poe vermelho no alto do centro. Pode '
                          'repetir a flag ate uma mancha por regiao que quiser pintar.')
    cor.add_argument('--scribble-cell', type=int, default=SCRIBBLE_CELL,
                     help=f'lado da mancha em pixels (default {SCRIBBLE_CELL}, o do treino)')
    cor.add_argument('--save-input', action='store_true',
                     help='salva tambem o esboco com as manchas desenhadas por cima, '
                          'para conferir o que foi pedido ao lado do que saiu')
    return p.parse_args()


def fit_sketch(im, size):
    """Enquadra um esboco do jeito que sprite-processer.py preparou o dataset.

    POR QUE ISSO E NECESSARIO

    A rede so viu imagens com um enquadramento muito especifico: fundo branco,
    desenho recortado no conteudo, escalado para encostar nas bordas e centralizado
    em 256x256. Um desenho seu, feito a mao num canto da tela, nao se parece com
    isso -- e uma rede convolucional e sensivel a escala e posicao. Sem este
    pre-processamento a saida degrada muito.

    Os tres passos reproduzem sprite-processer.py:
      1. transparencia vira branco (um PNG com fundo transparente leria como preto);
      2. recorta na bounding box do conteudo;
      3. escala pelo maior lado e centraliza, preservando a proporcao.
    """
    # getbbox() acha o retangulo do que NAO e preto; como o conteudo aqui e escuro
    # sobre claro, inverte-se a imagem primeiro para achar o retangulo do desenho.
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        fundo = Image.new('RGBA', im.size, 'WHITE')
        fundo.paste(im, (0, 0), im)  # o terceiro argumento usa o alfa como mascara
        im = fundo
    im = im.convert('L')

    bbox = ImageOps.invert(im).getbbox()
    if bbox:  # None quando a imagem e inteiramente branca
        im = im.crop(bbox)

    # Escala pelo MAIOR lado para a imagem caber inteira no quadrado, sem distorcer.
    escala = size / max(im.size)
    novo = Image.new('L', (size, size), 255)  # 255 = branco, o fundo do dataset
    red = im.resize((max(1, int(im.width * escala)), max(1, int(im.height * escala))),
                    Image.BOX)
    novo.paste(red, ((size - red.width) // 2, (size - red.height) // 2))
    return novo


def to_tensor(im):
    """PIL -> tensor (C, H, W) em [-1, 1]. Mesma conversao de PokemonPairs._to_tensor."""
    a = torch.from_numpy(np.array(im, dtype=np.uint8))
    if a.ndim == 2:
        a = a.unsqueeze(-1)
    return a.permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def to_pil(t):
    """Tensor (C, H, W) em [-1, 1] -> imagem PIL.

    O clamp protege contra valores fora da faixa (a tanh garante [-1, 1], mas
    entradas vindas de outro caminho podem escapar) e o round evita o vies de
    truncamento que um cast direto para uint8 introduziria.
    """
    a = t.detach().float().cpu().mul(0.5).add(0.5).clamp(0, 1)
    a = (a.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a.squeeze(-1) if a.shape[-1] == 1 else a)


def side_by_side(*images):
    """Cola imagens horizontalmente numa folha unica, alinhadas pelo topo.

    Converte tudo para RGB porque a entrada e em tons de cinza e o alvo e colorido:
    o PIL recusa colar modos diferentes na mesma folha.
    """
    w = sum(im.width for im in images)
    h = max(im.height for im in images)
    folha = Image.new('RGB', (w, h), 'white')
    x = 0
    for im in images:
        folha.paste(im.convert('RGB'), (x, 0))
        x += im.width
    return folha


def parse_cor(txt):
    """'#e63f19' ou 'e63f19' -> (r, g, b) em 0..1."""
    t = txt.strip().lstrip('#')
    if len(t) != 6:
        raise SystemExit(f'cor invalida: {txt!r} -- esperado hex de 6 digitos, ex. #e63f19')
    try:
        return tuple(int(t[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        raise SystemExit(f'cor invalida: {txt!r} -- esperado hex de 6 digitos, ex. #e63f19')


def ordena_por_luminancia(cores):
    """A ORDEM CANONICA DA PALETA, a mesma de tools_palette.py.

    Critico, e silencioso quando errado: o modelo aprendeu que a posicao 0 e a cor
    mais escura e a 3 a mais clara. Entregar as mesmas quatro cores em outra ordem
    nao da erro de forma nenhum -- so produz uma cor diferente da pedida.
    """
    return sorted(cores, key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])


def monta_paleta(spec, k=PALETTE_K):
    """Resolve --palette nas tres formas aceitas e devolve (k, 3) em 0..1, ordenada.

    Lista de hex e o controle direto; um caminho de imagem reaproveita o k-means de
    tools_palette.py, util para dizer "a cor daquele pokemon"; 'random' sorteia uma
    das paletas reais do dataset, que e o jeito de ver o modelo trabalhar sem ter de
    escolher cor nenhuma.
    """
    if spec == 'random':
        import numpy as _np
        z = _np.load(PALETTE_CACHE, allow_pickle=True)
        pals, keys = z['palettes'], z['keys']
        i = int(torch.randint(0, len(pals), ()))
        print(f'paleta sorteada: {keys[i]}')
        return pals[i].astype('float32')

    if ',' in spec or spec.lstrip('#').isalnum() and len(spec.lstrip('#')) == 6:
        cores = [parse_cor(c) for c in spec.split(',')]
        if len(cores) != k:
            raise SystemExit(f'--palette precisa de exatamente {k} cores, recebi {len(cores)}')
        return np.array(ordena_por_luminancia(cores), dtype=np.float32)

    caminho = Path(spec)
    if not caminho.exists():
        raise SystemExit(f'--palette: {spec!r} nao e lista de hex, nem "random", nem '
                         f'arquivo existente')
    from tools_palette import palette as extrai_paleta
    return extrai_paleta(caminho, k=k)


def parse_rabiscos(specs, size):
    """['0.5,0.2,#ff0000', ...] -> [((cy, cx), (r, g, b)), ...] em pixels.

    As coordenadas sao FRACIONARIAS (0..1) e nao em pixels porque o esboco passa por
    fit_sketch antes: recortado no conteudo, escalado e centralizado. Um pixel do
    arquivo original nao corresponde ao mesmo ponto da entrada da rede; uma fracao
    da moldura final, sim.
    """
    saida = []
    for spec in specs:
        partes = spec.split(',')
        if len(partes) != 3:
            raise SystemExit(f'--scribble invalido: {spec!r} -- esperado "X,Y,COR", '
                             f'ex. 0.5,0.2,#ff0000')
        try:
            fx, fy = float(partes[0]), float(partes[1])
        except ValueError:
            raise SystemExit(f'--scribble invalido: {spec!r} -- X e Y devem ser numeros')
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            raise SystemExit(f'--scribble fora da imagem: {spec!r} -- X e Y vao de 0 a 1')
        saida.append(((int(fy * size), int(fx * size)), parse_cor(partes[2])))
    return saida


def planos_condicionantes(in_channels, paleta, rabiscos, size, cell):
    """Os canais extras que o modelo espera depois do esboco, na ordem do treino.

    A paleta entra como canais CONSTANTES no espaco (o mesmo valor em todo pixel):
    e o que faz a condicao de cor atravessar as skip connections da U-Net, que
    contornariam uma injecao feita so no gargalo. O rabisco, ao contrario, e
    espacial -- e justamente a posicao dele que carrega a informacao.

    Fora das manchas o canal de rabisco fica em -1, que e como o modelo le "aqui nao
    foi dito nada". Nenhuma mancha significa o plano inteiro em -1, situacao que o
    treino viu em SCRIBBLE_EMPTY_P das amostras e portanto sabe tratar.
    """
    if in_channels == CH_ESBOCO:
        return None

    planos = []
    pal = torch.from_numpy(np.asarray(paleta, dtype=np.float32).reshape(-1))
    planos.append(pal.mul(2).sub(1)[:, None, None].expand(-1, size, size))

    if in_channels >= CH_ESBOCO + CH_PALETA + CH_RABISCO:
        rb = torch.full((CH_RABISCO, size, size), -1.0)
        for (cy, cx), cor in rabiscos:
            # min/max prendem a mancha dentro da moldura quando o centro pedido cai
            # perto da borda, em vez de recortar e virar uma mancha menor.
            topo = max(0, min(cy - cell // 2, size - cell))
            esq = max(0, min(cx - cell // 2, size - cell))
            for ch in range(CH_RABISCO):
                rb[ch, topo:topo + cell, esq:esq + cell] = cor[ch] * 2 - 1
        planos.append(rb)

    return torch.cat([p.contiguous() for p in planos], dim=0)


def desenha_rabiscos(esboco, rabiscos, cell):
    """O esboco com as manchas por cima: o que foi PEDIDO, para por ao lado do que saiu."""
    a = np.array(esboco.convert('RGB'))
    size = a.shape[0]
    for (cy, cx), cor in rabiscos:
        topo = max(0, min(cy - cell // 2, size - cell))
        esq = max(0, min(cx - cell // 2, size - cell))
        a[topo:topo + cell, esq:esq + cell] = (np.array(cor) * 255).astype(np.uint8)
    return Image.fromarray(a)


def find_checkpoint(args):
    """O .pt explicito, ou o de passo mais alto do diretorio derivado de --input-dir.

    Depende do nome com zeros a esquerda (ckpt-005000.pt) para que a ordem alfabetica
    do sorted() coincida com a numerica.
    """
    if args.checkpoint:
        return Path(args.checkpoint)
    ckpt_dir = Path(f'{Path(args.input_dir).name}_torch_training_checkpoints')
    found = sorted(ckpt_dir.glob('ckpt-*.pt'))
    if not found:
        raise FileNotFoundError(
            f'nenhum checkpoint em {ckpt_dir}/. Treine antes com pix2pix_train.py, '
            f'ou aponte um arquivo com --checkpoint.')
    return found[-1]


def load_generator(args, device):
    """Reconstroi o gerador e carrega os pesos salvos.

    O checkpoint guarda so os state_dict (tensores nomeados), nao a classe: por isso
    o modelo precisa ser instanciado antes, com a MESMA arquitetura do treino. Se
    --img-size nao bater com o que foi treinado, load_state_dict falha com erro de
    formas incompativeis -- o que e melhor do que carregar errado em silencio.

    O discriminador nao e carregado: ele so existe para treinar o gerador.
    """
    path = find_checkpoint(args)
    state = torch.load(path, map_location=device)
    # O modo do decoder vem do proprio checkpoint: assim os modelos treinados antes
    # da troca de ConvTranspose2d por upsample + convolucao continuam carregando.
    # skip_drop vem gravado no checkpoint; os treinados antes dele nao o tem e sao,
    # por definicao, skip_drop=0 (o pix2pix do artigo).
    generator = Generator(in_channels=in_channels_from_state(state), img_size=args.img_size,
                          up_mode=up_mode_from_state(state),
                          skip_drop=state.get('skip_drop', 0)).to(device)
    generator.load_state_dict(state['generator'])
    # O pix2pix amostra com dropout e BatchNorm em modo treino: e dali que vem a
    # variacao da saida. Em modo eval o gerador vira deterministico e tende a dar
    # resultados mais lavados. --deterministic permite comparar os dois.
    generator.eval() if args.deterministic else generator.train()
    sd = state.get('skip_drop', 0)
    print(f'checkpoint: {path.name} (passo {state["step"]}'
          + (f', skip_drop={sd}' if sd else '') + ')')
    return generator


def dataset_para(generator, input_paths, real_paths, idx, img_size):
    """PokemonPairs com as condicionantes que ESTE gerador espera.

    Aqui, ao contrario do modo --input, o alvo real existe: paleta e rabisco saem
    dele exatamente como no treino, entao a avaliacao mede o modelo e nao a minha
    escolha de cor. Sem isto, um checkpoint de 13 ou 16 canais recebe 1 canal e
    quebra com erro de formas.
    """
    n = generator.in_channels
    return PokemonPairs(
        input_paths, real_paths, idx, train=False, img_size=img_size,
        palettes=load_palettes(input_paths, real_paths) if n > CH_ESBOCO else None,
        scribble=n >= CH_ESBOCO + CH_PALETA + CH_RABISCO)


@torch.no_grad()  # inferencia nao precisa de gradiente em nenhum dos modos
def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    generator = load_generator(args, device)
    out_dir = Path(args.output_dir)

    # --- Modo 1: metrica sobre o conjunto de teste inteiro -----------------------
    if args.eval:
        # Reusa list_pairs + fixed_split com os defaults, entao este e exatamente o
        # mesmo conjunto que o treino nunca viu -- e o mesmo do notebook TensorFlow,
        # porque a semente do split e a mesma.
        input_paths, real_paths = list_pairs(args.input_dir, args.real_dir)
        _, test_idx = fixed_split(len(input_paths))
        ds = dataset_para(generator, input_paths, real_paths, test_idx, args.img_size)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = generator(x)
            # Media por imagem (dim 1,2,3 = canais, altura, largura) e depois soma,
            # em vez de uma media geral: assim o resultado nao depende de o ultimo
            # lote ser menor que os demais.
            total += torch.abs(y - pred).mean(dim=(1, 2, 3)).sum().item()
            n += x.size(0)
        print(f'L1 medio no conjunto de teste ({n} pares, nunca vistos): {total / n:.4f}')
        return

    # --- Modo 2: amostras do teste, com o alvo real ao lado ----------------------
    if args.from_test:
        input_paths, real_paths = list_pairs(args.input_dir, args.real_dir)
        _, test_idx = fixed_split(len(input_paths))
        ds = dataset_para(generator, input_paths, real_paths, test_idx[:args.from_test], args.img_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(ds)):
            x, y = ds[i]
            # unsqueeze(0) cria o eixo de lote: a rede sempre espera (N, C, H, W).
            pred = generator(x.unsqueeze(0).to(device))[0]
            nome = Path(ds.inputs[i]).parent.name + '-' + Path(ds.inputs[i]).stem
            # x[:1] = so o canal do esboco. Os canais de paleta e rabisco existem
            # para a rede, nao sao imagem: to_pil os recusaria.
            folha = side_by_side(to_pil(x[:1]), to_pil(y), to_pil(pred))
            folha.save(out_dir / f'{nome}.png')
            print(f'  {nome}.png  (entrada | alvo real | gerado)')
        print(f'{len(ds)} comparacoes em {out_dir}/')
        return

    # --- Modo 3: esbocos do usuario ----------------------------------------------
    origem = Path(args.input)
    arquivos = ([origem] if origem.is_file()
                else sorted(p for p in origem.iterdir() if p.suffix.lower() in SUFFIXES))
    if not arquivos:
        raise FileNotFoundError(f'nenhuma imagem em {origem}')

    if args.output and len(arquivos) > 1:
        raise SystemExit('--output serve para um arquivo so; use --output-dir')
    if not args.output:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Os modelos de 13 e 16 canais nao rodam so com o esboco: falta dizer a cor. Como
    # aqui NAO existe alvo de onde extrai-la (o desenho e do usuario, nao ha sprite
    # verdadeiro), ela tem de vir das flags. Recusar cedo, com a flag que falta no
    # texto, evita o erro de forma cru do load/forward la na frente.
    n_ch = generator.in_channels
    if n_ch > CH_ESBOCO and not args.palette:
        tem_rabisco = n_ch >= CH_ESBOCO + CH_PALETA + CH_RABISCO
        raise SystemExit('\n'.join([
            f'este checkpoint espera {n_ch} canais (esboco + paleta'
            + (' + rabisco' if tem_rabisco else '') + '), mas --palette nao foi dada.',
            '  escolha as cores:         --palette "#1a1a1a,#e63f19,#fa8c19,#ffe033"',
            '  ou copie de uma imagem:   --palette sprites-processed/gen1/charmander.png',
            '  ou deixe o acaso decidir: --palette random',
        ]))
    if args.scribble and n_ch < CH_ESBOCO + CH_PALETA + CH_RABISCO:
        raise SystemExit(f'--scribble exige um checkpoint de '
                         f'{CH_ESBOCO + CH_PALETA + CH_RABISCO} canais; este tem {n_ch}')

    paleta = monta_paleta(args.palette) if args.palette else None
    rabiscos = parse_rabiscos(args.scribble, args.img_size)
    extras = planos_condicionantes(n_ch, paleta, rabiscos, args.img_size, args.scribble_cell)

    for arq in arquivos:
        im = Image.open(arq)
        esboco = (im.convert('L').resize((args.img_size, args.img_size), Image.NEAREST)
                  if args.no_fit else fit_sketch(im, args.img_size))
        x = to_tensor(esboco)
        if extras is not None:
            x = torch.cat([x, extras], dim=0)
        pred = generator(x.unsqueeze(0).to(device))[0]
        destino = Path(args.output) if args.output else out_dir / f'{arq.stem}-gerado.png'
        to_pil(pred).save(destino)
        print(f'  {arq.name} -> {destino}')
        if args.save_input:
            pedido = destino.with_name(f'{destino.stem}-pedido.png')
            desenha_rabiscos(esboco, rabiscos, args.scribble_cell).save(pedido)
            print(f'  {arq.name} -> {pedido}  (entrada, com o rabisco pedido)')


if __name__ == '__main__':
    main()
