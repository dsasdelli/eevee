"""Pipeline de dados: pares (esboco de contorno, sprite colorido).

COMO O DATASET FOI CONSTRUIDO (pelos scripts que ja existiam no projeto)

    sprite-downloader.py  -> sprites/                      PNG originais, 9 geracoes
    sprite-processer.py   -> sprites-processed/            256x256, fundo branco, centralizado
    sprite-contours.py    -> sprites-processed-countours/  o contorno, em tons de cinza

Sao 4.289 pares. Cada par e o MESMO arquivo nos dois diretorios: o contorno e a
entrada da rede, o sprite colorido e o alvo.

AS DUAS CORRECOES QUE ESTE ARQUIVO CARREGA

Este modulo porta para o PyTorch dois bugs que foram encontrados e corrigidos na
versao TensorFlow (ver img_gen_contours.ipynb):

  1. SPLIT COM VAZAMENTO. No codigo original o split era take(1000)/skip(1000)
     aplicado DEPOIS de um .shuffle() que reembaralha a cada passada. Como take e
     skip sao preguicosos, cada passada recortava um conjunto diferente: medido,
     638 das 1000 imagens de "teste" trocavam de lado entre duas leituras. O teste
     vazava para o treino e a avaliacao nao media generalizacao nenhuma.
     Aqui o split e sorteado UMA vez sobre a lista de arquivos, com semente fixa.

  2. AUGMENTACAO CONGELADA. A semente do recorte aleatorio era sorteada com
     np.random dentro de uma funcao tracada pelo tf.data, entao virava uma
     constante no grafo: as 4.289 imagens recebiam o MESMO recorte e o MESMO
     espelhamento, durante os 40.000 passos. Medido: 1 semente distinta em 6.
     O equivalente desse bug em PyTorch e os workers do DataLoader herdarem a mesma
     semente. Aqui o sorteio usa o RNG do torch (que o DataLoader semeia por worker)
     e o worker_init_fn cobre tambem o RNG do numpy.

A semente do split e a mesma do notebook (1337) de proposito: assim o conjunto de
teste e EXATAMENTE o mesmo nas duas implementacoes, e os resultados sao
diretamente comparaveis.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

PATH_INPUT = 'sprites-processed-countours'  # entrada da rede: o esboco
PATH_REAL = 'sprites-processed'             # alvo: o sprite colorido

IMG_SIZE = 256
# O jitter amplia para 286 e recorta de volta para 256. A folga de 30 pixels e o
# quanto o recorte pode deslocar a imagem -- e o valor usado no artigo do pix2pix.
JITTER_SIZE = 286
TEST_SIZE = 1000
SPLIT_SEED = 1337
# Fracao minima de pixels de traco que uma entrada augmentada precisa ter. Abaixo
# disso a imagem esta vazia demais para ser um desenho, e o par vira ruido.
MIN_INK = 0.01
# Paleta dominante de cada sprite, pre-calculada por tools_palette.py. Condiciona a
# cor da saida: sem ela o contorno nao carrega informacao de cor nenhuma e a predicao
# que minimiza a L1 e a mediana das cores plausiveis, que sai dessaturada.
PALETTE_CACHE = 'palettes.npz'
PALETTE_K = 4
# Ruido somado a paleta no treino. O usuario vai escolher cores que nao sao exatamente
# as de nenhum sprite; sem esse jitter a rede so veria as 4.289 paletas exatas do
# dataset e ficaria fragil fora delas.
PALETTE_JITTER = 0.04
# Rabisco de cor: manchas grosseiras posicionadas sobre o desenho, dizendo ONDE cada
# cor vai. A paleta sozinha diz QUAIS cores -- medido, com silhueta limpa ela so
# derruba 5% do erro de crominancia contra 21% com o contorno denso do dataset, porque
# sem textura interna as cores certas vao para os lugares errados.
SCRIBBLE_CELL = 24      # lado da celula de cor, em pixels: mancha grossa, nao pintura
SCRIBBLE_MAX = 7        # quantas manchas no maximo por amostra
SCRIBBLE_EMPTY_P = 0.2  # fracao de amostras SEM rabisco nenhum, para o modelo
                        # continuar funcionando quando o usuario nao rabiscar nada
# Infantilizacao do traco. O caso de uso declarado do projeto e "rabisco infantil ->
# pokemon plausivel": a saida deve sair MAIS detalhada e realista que a entrada. Os
# contornos do dataset, e mesmo um desenho feito com cuidado, tem linha firme e
# proporcao correta -- nada ensina a rede a consertar um traco ruim.
#
# A deformacao vai so na ENTRADA, com o alvo intacto. Isso desalinha o par de
# proposito: e o unico jeito de o modelo aprender a CORRIGIR a forma em vez de copiar
# a distorcao. O preco conhecido e hedge da L1 (borrao) se a amplitude passar do que o
# modelo consegue recuperar, por isso o campo e SUAVE -- um deslocamento de baixa
# frequencia preserva a correspondencia global, enquanto ruido por pixel a destruiria.
TREMOR_P = 0.5          # fracao de amostras de treino que recebem deformacao
# Duas escalas somadas, porque um desenho ruim erra em dois registros ao mesmo tempo:
TREMOR_FORMA = (28.0, 7.0)   # (sigma, amplitude) em px: proporcao torta, assimetria
TREMOR_LINHA = (12.0, 2.6)   # (sigma, amplitude) em px: mao tremida, linha ondulada
TREMOR_ESPESSURA_P = 0.3     # chance de engrossar o traco (crianca aperta o lapis)


def list_pairs(path_input=PATH_INPUT, path_real=PATH_REAL):
    """Lista ordenada dos dois lados do par, com o pareamento conferido.

    O pareamento depende de os dois diretorios terem a mesma estrutura
    (<geracao>/<pokemon>.png) e de sorted() produzir a mesma ordem nos dois. Isso e
    verdade, mas e uma suposicao silenciosa e fragil: se um arquivo faltar de um
    lado, todos os pares a partir dali ficam deslocados e a rede treina com alvos
    trocados, sem erro nenhum aparecer. Dai a conferencia explicita abaixo.
    """
    input_paths = sorted(Path(path_input).glob('*/*.png'))
    real_paths = sorted(Path(path_real).glob('*/*.png'))

    if not input_paths:
        raise FileNotFoundError(f'nenhum png encontrado em {path_input}/*/*.png')
    if len(input_paths) != len(real_paths):
        raise ValueError(f'{len(input_paths)} entradas para {len(real_paths)} alvos')
    for p_in, p_re in zip(input_paths, real_paths):
        # Compara geracao e nome do arquivo; o diretorio raiz e que difere.
        if (p_in.parent.name, p_in.name) != (p_re.parent.name, p_re.name):
            raise ValueError(f'par desalinhado: {p_in} != {p_re}')

    return input_paths, real_paths


def fixed_split(n, test_size=TEST_SIZE, seed=SPLIT_SEED):
    """Split fixo e reproduzivel. Devolve (train_idx, test_idx), disjuntos.

    default_rng(seed) cria um gerador proprio, isolado do estado global do numpy:
    o resultado nao muda se outro trecho do programa sortear alguma coisa antes.
    Com a mesma semente, a mesma permutacao sai sempre -- e e isso que torna o
    conjunto de teste estavel entre rodadas e entre implementacoes.

    ATENCAO -- limitacao conhecida (bug 3 da avaliacao, ainda em aberto): o split e
    por IMAGEM, nao por pokemon. O mesmo pokemon aparece em ate 9 geracoes com arte
    parecida, entao 91,7% das imagens de teste tem um "primo" no treino. O conjunto
    e de fato disjunto e nunca treinado, mas a metrica sai otimista. Corrigir exige
    agrupar por p.name antes de permutar.
    """
    perm = np.random.default_rng(seed).permutation(n)
    return perm[test_size:], perm[:test_size]


class PokemonPairs(Dataset):
    """Pares (contorno 1 canal, sprite RGB) normalizados para [-1, 1], em formato CHW.

    Implementa o protocolo Dataset do PyTorch: __len__ e __getitem__. O DataLoader
    cuida de agrupar em lotes, embaralhar e paralelizar -- este objeto so precisa
    saber entregar UM par por indice.

    O parametro `train` decide se ha augmentacao:
      - treino: jitter aleatorio (recorte + espelhamento), para multiplicar o dataset;
      - teste:  nenhuma, para que a avaliacao seja identica entre rodadas.
    """

    def __init__(self, input_paths, real_paths, indices, train,
                 img_size=IMG_SIZE, jitter_size=JITTER_SIZE, palettes=None,
                 scribble=False):
        # Guarda so os caminhos dos indices deste split; as imagens sao lidas do
        # disco sob demanda em __getitem__. Carregar as 4.289 de uma vez custaria
        # muita RAM e nao traria ganho, ja que o jitter precisa reprocessar a cada
        # epoca de qualquer forma.
        self.inputs = [input_paths[i] for i in indices]
        self.reals = [real_paths[i] for i in indices]
        self.train = train
        self.img_size = img_size
        self.jitter_size = jitter_size
        # (N, K, 3) em 0..1 na mesma ordem de self.reals, ou None para o modo antigo
        # de 1 canal. Guardado por indice para nao reconsultar o dicionario por amostra.
        self.palettes = None if palettes is None else palettes[list(indices)]
        self.scribble = scribble

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        # convert() explicito garante o numero de canais esperado pela rede,
        # independentemente de como o PNG foi gravado: 1 canal na entrada, 3 no alvo.
        x = Image.open(self.inputs[i]).convert('L')
        y = Image.open(self.reals[i]).convert('RGB')

        if self.train:
            # Estilo do traco ANTES do jitter geometrico: a morfologia trabalha em
            # pixels, e fazer isso no tamanho nativo mantem x e y alinhados para o
            # recorte seguinte, que precisa ser identico nos dois lados.
            x = self._stroke_variant(x, y)
            # Depois do estilo e antes do jitter: a deformacao atua sobre QUALQUER
            # estilo sorteado acima (contorno denso, erodido ou silhueta), senao so
            # metade das entradas aprenderia a ser consertada.
            x = self._infantiliza(x)
            x, y = self._random_jitter(x, y)
        else:
            # Sem jitter no teste. NEAREST (e nao uma interpolacao suave) porque sao
            # sprites de pixel art: suavizar inventaria cores que nao existem no
            # original e borraria as bordas duras.
            x = x.resize((self.img_size, self.img_size), Image.NEAREST)
            y = y.resize((self.img_size, self.img_size), Image.NEAREST)

        xt, yt = self._to_tensor(x), self._to_tensor(y)
        if self.palettes is not None:
            xt = torch.cat([xt, self._palette_planes(i, xt.shape[-2:])], dim=0)
        if self.scribble:
            xt = torch.cat([xt, self._scribble_planes(yt)], dim=0)
        return xt, yt

    def _scribble_planes(self, yt):
        """3 canais com manchas de cor amostradas do alvo, nas posicoes delas.

        Construido a partir do PROPRIO alvo porque e assim que o rabisco de um usuario
        se parece: ele poe verde onde quer verde. As manchas sao quadrados grandes de
        cor media local -- o objetivo e dizer regiao, nao desenhar. Fora das manchas o
        canal fica em -1 (neutro), que e como o modelo sabe 'aqui nao foi dito nada'.

        Em SCRIBBLE_EMPTY_P das amostras nao ha mancha nenhuma: sem isso o modelo
        passaria a depender do rabisco e degradaria quando o usuario nao desse nenhum.
        """
        c, h, w = 3, *yt.shape[-2:]
        out = torch.full((c, h, w), -1.0)
        if not self.train or float(torch.rand(())) < SCRIBBLE_EMPTY_P:
            return out if self.train else self._scribble_deterministic(yt, out)

        # As manchas tem de cair SOBRE o bicho. Sorteando posicao uniformemente na
        # imagem, a maioria cairia no fundo branco, que ocupa mais area que o conteudo:
        # o rabisco diria 'branco aqui' em vez de dizer a cor de alguma parte. Um
        # usuario rabisca dentro da forma, entao os centros saem da mascara de conteudo.
        conteudo = self._content_centers(yt)
        if conteudo is None:
            return out
        n = int(torch.randint(1, SCRIBBLE_MAX + 1, ()))
        escolha = torch.randint(0, conteudo.shape[0], (n,))
        for j in escolha.tolist():
            cy, cx = int(conteudo[j, 0]), int(conteudo[j, 1])
            out = self._dab(out, yt, cy, cx)
        return out

    @staticmethod
    def _content_centers(yt):
        """Coordenadas dos pixels do bicho (nao do fundo branco), ou None se nao houver.

        Subamostra a cada 4 pixels: a lista completa teria dezenas de milhares de
        posicoes e so precisamos sortear meia duzia de centros.
        """
        rgb = yt.mul(0.5).add(0.5)
        lum = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
        idx = torch.nonzero(lum[::4, ::4] < 0.98)
        return idx * 4 if idx.numel() else None

    @staticmethod
    def _dab(out, yt, cy, cx):
        """Pinta uma mancha chapada de SCRIBBLE_CELL de lado, centrada em (cy, cx),
        com a cor media que o alvo tem ali -- como um dedo de tinta."""
        h, w = out.shape[-2:]
        lado = SCRIBBLE_CELL
        top = max(0, min(cy - lado // 2, h - lado))
        left = max(0, min(cx - lado // 2, w - lado))
        bloco = yt[:, top:top + lado, left:left + lado]
        out[:, top:top + lado, left:left + lado] = bloco.mean(dim=(1, 2))[:, None, None]
        return out

    @staticmethod
    def _scribble_deterministic(yt, out):
        """Rabisco fixo para o conjunto de teste, para a avaliacao nao variar entre
        rodadas. Tres manchas em posicoes fixas sobre o terco central da imagem."""
        centros = PokemonPairs._content_centers(yt)
        if centros is None:
            return out
        # Tres posicoes fixas DENTRO do conteudo: fracoes da lista ordenada, entao a
        # escolha e reproduzivel e sempre cai sobre o bicho, qualquer que seja a forma.
        for frac in (0.25, 0.5, 0.75):
            cy, cx = centros[int(frac * (centros.shape[0] - 1))].tolist()
            out = PokemonPairs._dab(out, yt, cy, cx)
        return out

    def _palette_planes(self, i, hw):
        """A paleta como K*3 canais constantes, do tamanho da imagem.

        Espalhar um valor global por todos os pixels e redundante em memoria, mas e o
        que faz a condicao chegar em TODA camada da U-Net -- inclusive atraves das skip
        connections, que de outro modo contornariam uma injecao feita so no gargalo.
        A cor e uma propriedade global da saida, entao precisa estar disponivel em
        todas as resolucoes.
        """
        pal = torch.from_numpy(self.palettes[i]).reshape(-1)  # (K*3,) em 0..1
        if self.train and PALETTE_JITTER:
            pal = (pal + torch.randn_like(pal) * PALETTE_JITTER).clamp(0, 1)
        pal = pal.mul(2).sub(1)  # mesma faixa [-1, 1] do resto da entrada
        return pal[:, None, None].expand(-1, *hw).contiguous()

    def _stroke_variant(self, x, y):
        """Sorteia o ESTILO do traco da entrada, cobrindo o que um usuario desenha.

        POR QUE ISSO EXISTE

        O contorno do dataset sai do adaptiveThreshold de sprite-contours.py e tem
        ~20% dos pixels escuros (medido: 11% a 29%): linha grossa mais uma textura
        interna densa, em que cada ponto marca uma mudanca de luminancia do sprite
        original. Um desenho humano -- silhueta mais algumas linhas internas -- fica
        em 4% a 8%.

        Treinar so com o contorno do dataset ensina a rede a LER essa textura: todo o
        relevo interno da saida vem dali. Medido no ckpt-050000, dando a ela uma
        silhueta limpa a saida colapsa para um borrao liso, sem interior nenhum.

        Sorteando o estilo a cada amostra, a rede ve a faixa inteira de densidade e
        aprende a INVENTAR o interior quando ele nao vem na entrada -- que e o
        comportamento que um gerador a partir de rabisco precisa ter.

        Os sorteios usam o RNG do torch, que o DataLoader semeia por worker e por
        epoca; com np.random cairiamos de volta no bug 2 descrito no topo do arquivo.
        """
        r = float(torch.rand(()))
        if r < 0.35:
            return x  # contorno do dataset, como esta no disco

        dark = np.array(x, dtype=np.uint8) < 128

        if r < 0.70:
            # Afina o traco: erodir a mascara escura encolhe as linhas e derruba a
            # tinta para perto da faixa de um desenho humano, preservando o tracado.
            #
            # COM PISO: em contornos de traco fino, 2 iteracoes apagam as linhas por
            # inteiro -- medido, 7,8% das amostras saiam praticamente em branco. Uma
            # entrada vazia com alvo colorido ensina a rede a gerar do nada, que e
            # ruido de treino. Quando a erosao derruba demais, cai para 1 iteracao e,
            # se ainda assim sumir, devolve o traco original.
            k = np.ones((3, 3), bool)
            iters = 1 + int(torch.randint(0, 2, ()))
            thin = ndimage.binary_erosion(dark, k, iterations=iters)
            if thin.mean() < MIN_INK and iters > 1:
                thin = ndimage.binary_erosion(dark, k, iterations=1)
            if thin.mean() < MIN_INK:
                thin = dark
            return Image.fromarray(np.where(thin, 0, 255).astype(np.uint8))

        # Silhueta limpa: so a borda externa do sprite. Em metade dos casos leva
        # junto as linhas internas afinadas -- um desenho real costuma ter olhos e
        # separacao de membros, mas nao a textura de sombreado inteira.
        sil = self._silhouette(y, dark.shape)
        if r < 0.85:
            inner = ndimage.binary_erosion(dark, np.ones((3, 3), bool), iterations=2)
            sil = sil | inner
        return Image.fromarray(np.where(sil, 0, 255).astype(np.uint8))

    def _infantiliza(self, x):
        """Deforma o traco como um desenho malfeito: proporcao torta e linha tremida.

        POR QUE ISSO EXISTE

        O projeto quer transformar rabisco infantil em pokemon plausivel. Tudo que a
        rede ve no treino, porem, tem forma correta: os contornos saem de sprites
        reais, e mesmo a silhueta limpa de _stroke_variant e a silhueta EXATA do alvo.
        A rede nunca precisa consertar nada -- so preencher. Com entrada torta ela nao
        tem motivo aprendido para endireitar.

        COMO

        Um campo de deslocamento aleatorio suavizado por gaussiana, somado em duas
        escalas: a larga (sigma ~28 px) move regioes inteiras e produz desproporcao e
        assimetria; a estreita (sigma ~5 px) ondula a linha e imita mao tremida. O
        mesmo campo vale para a imagem toda, entao o desenho continua conexo.

        O ALVO NAO E DEFORMADO, de proposito. Em _random_jitter o alinhamento entre
        entrada e alvo e sagrado, porque la o objetivo e nao mentir sobre a posicao.
        Aqui o desalinhamento E o sinal de treino: o par passa a dizer "desta entrada
        torta, produza esta forma correta". E a unica maneira de a correcao de forma
        virar objetivo em vez de acidente.

        order=0 (vizinho mais proximo) mantem a imagem binaria; uma interpolacao suave
        criaria tons de cinza intermediarios que nao existem num traco de verdade.
        """
        if float(torch.rand(())) >= TREMOR_P:
            return x

        a = np.array(x, dtype=np.uint8)
        h, w = a.shape

        # torch.rand e nao np.random: o DataLoader semeia o RNG do torch por worker e
        # por epoca. Com np.random todos os workers repetiriam a mesma sequencia --
        # e o bug 2 descrito no topo deste arquivo.
        desloc = []
        for eixo in range(2):
            campo = np.zeros((h, w), np.float32)
            for sigma, amp in (TREMOR_FORMA, TREMOR_LINHA):
                bruto = torch.rand((h, w)).mul(2).sub(1).numpy().astype(np.float32)
                suave = ndimage.gaussian_filter(bruto, sigma)
                # Suavizar derruba a amplitude de forma dependente do sigma; renormalizar
                # pelo desvio observado faz 'amp' significar pixels de verdade, e nao um
                # numero que muda de efeito quando o sigma muda.
                dp = float(suave.std()) or 1.0
                campo += suave / dp * amp
            desloc.append(campo)

        ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing='ij')
        # mode='constant' com cval=255: o que vier de fora da moldura e fundo branco,
        # nao borda espelhada -- espelhar inventaria traco que o usuario nao desenhou.
        saida = ndimage.map_coordinates(a, [ys + desloc[0], xs + desloc[1]],
                                        order=0, mode='constant', cval=255)

        if float(torch.rand(())) < TREMOR_ESPESSURA_P:
            escuro = ndimage.binary_dilation(saida < 128, np.ones((3, 3), bool))
            saida = np.where(escuro, 0, 255).astype(np.uint8)

        # Mesmo piso de _stroke_variant: se a deformacao esvaziou o desenho, devolve o
        # original. Entrada vazia com alvo colorido ensina a rede a gerar do nada.
        if (saida < 128).mean() < MIN_INK:
            return x
        return Image.fromarray(saida.astype(np.uint8))

    @staticmethod
    def _silhouette(y, shape):
        """Borda externa do sprite: mascara do conteudo, buracos preenchidos, gradiente.

        binary_fill_holes evita que vazios internos claros (um olho branco, por
        exemplo) virem contorno; o gradiente morfologico (dilatacao menos erosao) da
        um traco de 1 a 2 pixels, parecido com o de quem desenha com o mouse.
        """
        if y.size != (shape[1], shape[0]):
            y = y.resize((shape[1], shape[0]), Image.NEAREST)
        lum = np.array(y.convert('L'), dtype=np.uint8)
        filled = ndimage.binary_fill_holes(lum < 250)
        k = np.ones((3, 3), bool)
        return ndimage.binary_dilation(filled, k) & ~ndimage.binary_erosion(filled, k)

    def _random_jitter(self, x, y):
        """Amplia para 286, recorta 256 numa posicao aleatoria e talvez espelha.

        E a augmentacao do artigo do pix2pix. Com so 3.289 pares de treino e 40.000
        passos, cada imagem e vista ~12 vezes; sem variacao a rede decoraria o
        conjunto. O deslocamento tambem ensina que a identidade do pokemon nao
        depende da posicao exata na moldura.
        """
        s, c = self.jitter_size, self.img_size
        x = x.resize((s, s), Image.NEAREST)
        y = y.resize((s, s), Image.NEAREST)

        # CRITICO: o mesmo recorte e o mesmo espelhamento nos dois lados. Entrada e
        # alvo tem de continuar alinhados pixel a pixel depois da augmentacao --
        # se divergirem, a perda L1 passa a comparar coisas deslocadas e a rede
        # aprende a borrar para compensar. Por isso os sorteios sao feitos uma vez
        # aqui e aplicados aos dois, em vez de um transform independente por imagem.
        #
        # torch.randint usa o RNG do torch, que o DataLoader semeia de forma distinta
        # em cada worker e a cada epoca -- e o que evita o bug 2 descrito no topo.
        top, left = torch.randint(0, s - c + 1, (2,)).tolist()
        box = (left, top, left + c, top + c)
        x, y = x.crop(box), y.crop(box)

        if torch.rand(()) < 0.5:
            x = x.transpose(Image.FLIP_LEFT_RIGHT)
            y = y.transpose(Image.FLIP_LEFT_RIGHT)

        return x, y

    @staticmethod
    def _to_tensor(im):
        """PIL (H, W[, C]) uint8 0..255  ->  tensor (C, H, W) float32 em [-1, 1].

        Tres conversoes de uma vez:
          - layout: o PIL/numpy usa (altura, largura, canais); o PyTorch espera
            (canais, altura, largura), dai o permute;
          - eixo de canal: imagem em tons de cinza vem como (H, W), sem o eixo, e
            precisa do unsqueeze para virar (H, W, 1);
          - escala: div(127.5).sub(1) leva 0..255 para -1..1, a mesma faixa da saida
            tanh do gerador. As duas pontas precisam casar.

        div_ e sub_ sao in-place (o underscore), evitando copias intermediarias --
        isso roda 4.289 vezes por epoca.
        """
        a = torch.from_numpy(np.array(im, dtype=np.uint8))
        if a.ndim == 2:
            a = a.unsqueeze(-1)
        return a.permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def _worker_init(worker_id):
    """Garante semente distinta tambem para o numpy em cada worker do DataLoader.

    O PyTorch ja semeia o RNG do *torch* por worker automaticamente, mas nao o do
    numpy: sem isso, todos os workers herdariam o mesmo estado e devolveriam a mesma
    sequencia de sorteios -- a versao PyTorch do bug 2. Este codigo nao usa numpy
    para sortear nada hoje, mas a protecao fica para quem editar o jitter depois.
    """
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def load_palettes(input_paths, real_paths, cache=PALETTE_CACHE):
    """Paletas na MESMA ordem de real_paths. Devolve (N, K, 3) ou levanta erro.

    O cache guarda a chave '<geracao>/<arquivo>.png' de cada sprite em vez de confiar
    na ordem do arquivo: se um sprite for adicionado ou removido depois, reordenar por
    chave ainda casa, enquanto um indice posicional casaria errado em silencio.
    """
    import numpy as _np
    d = _np.load(cache)
    por_chave = {k: v for k, v in zip(d['keys'], d['palettes'])}
    faltando = [p for p in real_paths if f'{p.parent.name}/{p.name}' not in por_chave]
    if faltando:
        raise KeyError(f'{len(faltando)} sprites sem paleta em {cache} '
                       f'(ex.: {faltando[0]}). Rode tools_palette.py de novo.')
    return _np.stack([por_chave[f'{p.parent.name}/{p.name}'] for p in real_paths])


def make_loaders(batch_size=1, num_workers=4, img_size=IMG_SIZE,
                 path_input=PATH_INPUT, path_real=PATH_REAL,
                 test_size=TEST_SIZE, seed=SPLIT_SEED, pin_memory=True,
                 palette=False, scribble=False):
    """Monta os dois DataLoader com o split fixo ja aplicado.

    Sobre os parametros do DataLoader:
      num_workers       processos paralelos lendo e preparando imagens. Decodificar
                        PNG e fazer o jitter e trabalho de CPU; sem workers a GPU
                        fica esperando o disco.
      pin_memory        aloca os lotes em memoria "fixada", que a GPU copia mais
                        rapido. So faz sentido com CUDA.
      persistent_workers mantem os processos vivos entre epocas. No Windows criar um
                        processo e caro (reimporta os modulos), e o treino percorre
                        o dataset ~12 vezes.
      drop_last         descarta o ultimo lote incompleto no treino, para que todos
                        os lotes tenham o mesmo tamanho (a BatchNorm se comporta
                        diferente com lotes de tamanhos variados).
      shuffle           ligado so no treino. No teste a ordem fixa mantem a
                        avaliacao reproduzivel.
    """
    input_paths, real_paths = list_pairs(path_input, path_real)
    train_idx, test_idx = fixed_split(len(input_paths), test_size, seed)
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())

    # Mantem a proporcao 286/256 do artigo quando se treina em outra resolucao.
    jitter = round(img_size * JITTER_SIZE / IMG_SIZE)
    paletas = load_palettes(input_paths, real_paths) if palette else None
    common = dict(img_size=img_size, jitter_size=jitter, palettes=paletas,
                  scribble=scribble)
    train_ds = PokemonPairs(input_paths, real_paths, train_idx, train=True, **common)
    test_ds = PokemonPairs(input_paths, real_paths, test_idx, train=False, **common)

    loader_args = dict(num_workers=num_workers, worker_init_fn=_worker_init,
                       pin_memory=pin_memory,
                       persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_args)

    print(f'{len(input_paths)} pares: {path_input} -> {path_real}'
          + (f' | paleta de {PALETTE_K} cores' if palette else '')
          + (' | rabisco de cor' if scribble else ''))
    print(f'treino: {len(train_ds)} | teste: {len(test_ds)} | disjuntos, seed={seed}')
    return train_loader, test_loader
