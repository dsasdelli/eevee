"""Arquitetura do pix2pix: gerador U-Net e discriminador PatchGAN.

CONTEXTO -- o que o pix2pix faz

O pix2pix e uma GAN condicional para traducao imagem-a-imagem. Duas redes sao
treinadas ao mesmo tempo, em oposicao:

  - O GERADOR recebe o esboco e tenta produzir o sprite colorido.
  - O DISCRIMINADOR recebe o par (esboco, imagem) e tenta dizer se aquela imagem
    e o sprite real ou uma falsificacao do gerador.

O gerador melhora tentando enganar o discriminador; o discriminador melhora
tentando nao ser enganado. "Condicional" quer dizer que o discriminador sempre ve
o esboco junto, entao nao basta o gerador produzir um sprite bonito qualquer -- tem
de ser um sprite coerente COM AQUELE esboco.

Alem da perda adversarial, o gerador leva uma perda L1 (diferenca absoluta media
contra o sprite real), com peso LAMBDA=100. A divisao de trabalho e:

  - a L1 cuida do acerto global -- cor e forma aproximadas;
  - a perda adversarial cuida da nitidez -- sozinha, a L1 produz borroes, porque a
    media de todas as saidas plausiveis e um borrao.

Este arquivo e um porte de img_gen_contours.ipynb. As formas e a contagem de
parametros conferem com a versao TensorFlow (54.411.907 contra 54.412.931 -- a
diferenca de 1.024 esta explicada no comentario do gargalo, em Generator).
"""
import math

import torch
from torch import nn

# Inclinacao do LeakyReLU para entradas negativas.
#
# O notebook usa tf.keras.layers.LeakyReLU() sem argumento, e o default do Keras e
# 0.3 (o artigo original do pix2pix usa 0.2). Mantido em 0.3 de proposito: assim as
# curvas desta implementacao sao comparaveis com a rodada das sombras de janeiro.
# O default do PyTorch seria 0.01, bem diferente -- por isso e passado explicitamente.
LEAKY_SLOPE = 0.3


def _init_weights(m):
    """Inicializa pesos de convolucao com ruido gaussiano de desvio 0.02.

    Equivale ao kernel_initializer=tf.random_normal_initializer(0., 0.02) do
    notebook. GANs sao sensiveis a inicializacao: pesos grandes demais fazem o
    discriminador vencer imediatamente e o gradiente do gerador sumir. O valor
    0.02 vem do artigo da DCGAN e virou padrao no pix2pix.

    Aplicado via self.apply(), que percorre recursivamente todos os submodulos.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Downsample(nn.Sequential):
    """Bloco do caminho descendente (encoder): Conv2D(stride=2) -> BatchNorm -> LeakyReLU.

    Reduz a resolucao pela metade e dobra (ate o teto) a profundidade de canais.
    A reducao e feita pelo proprio stride=2 da convolucao, nao por pooling -- assim
    a rede aprende COMO reduzir, em vez de descartar pixels por uma regra fixa.

    Sobre o padding: kernel 4 com stride 2 e padding 1 reproduz exatamente o
    padding='same' do TensorFlow. Conferindo com entrada 256:
        saida = (256 + 2*1 - 4) / 2 + 1 = 128
    ou seja, metade exata, sem resto.

    bias=False porque logo em seguida vem a BatchNorm, que subtrai a media de cada
    canal: qualquer bias somado antes seria cancelado ali. Economiza parametros sem
    mudar nada no resultado.
    """

    def __init__(self, in_ch, out_ch, size=4, batchnorm=True):
        layers = [nn.Conv2d(in_ch, out_ch, size, stride=2, padding=1, bias=False)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        # LeakyReLU em vez de ReLU: no caminho descendente, deixar passar um residuo
        # das ativacoes negativas evita que neuronios "morram" (fiquem presos em zero
        # e parem de receber gradiente), problema comum em discriminadores de GAN.
        layers.append(nn.LeakyReLU(LEAKY_SLOPE, inplace=True))
        super().__init__(*layers)


class Upsample(nn.Sequential):
    """Bloco do caminho ascendente (decoder): dobra a resolucao -> BatchNorm -> [Dropout] -> ReLU.

    DOIS MODOS DE DOBRAR A RESOLUCAO

    'transpose' e o do artigo: ConvTranspose2d com kernel 4, stride 2 e padding 1,
    que e o inverso da convolucao com stride
        saida = (entrada - 1) * 2 - 2*1 + 4 = 2 * entrada

    'resize' e a correcao de Odena et al. para artefato de tabuleiro: amplia por
    vizinho mais proximo e so depois convolui, com stride 1. A convolucao transposta
    sobrepoe janelas de forma desigual quando o kernel nao e divisivel pelo stride, e
    essa irregularidade vira listra periodica na saida. Medido neste projeto: com
    entrada de silhueta limpa o modelo 'transpose' produz 19% mais gradiente
    horizontal que vertical (razao 1,23 contra 1,04 dos sprites reais) -- listra.

    O vizinho mais proximo (e nao bilinear) tambem combina com o resto do pipeline,
    que usa Image.NEAREST em toda parte por serem sprites de pixel art.

    O Dropout aqui NAO e so regularizacao. No pix2pix ele continua ligado durante a
    amostragem (ver save_preview em pix2pix_train.py) e funciona como a unica fonte
    de aleatoriedade do gerador: sem ele a rede seria uma funcao deterministica do
    esboco e sempre produziria a mesma saida. O artigo aplica dropout so nos tres
    primeiros blocos ascendentes, que e o que Generator faz.
    """

    def __init__(self, in_ch, out_ch, size=4, dropout=False, mode='resize'):
        if mode == 'resize':
            dobra = [nn.Upsample(scale_factor=2, mode='nearest'),
                     nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)]
        elif mode == 'transpose':
            dobra = [nn.ConvTranspose2d(in_ch, out_ch, size, stride=2, padding=1, bias=False)]
        else:
            raise ValueError(f"mode deve ser 'resize' ou 'transpose', veio {mode!r}")
        layers = [*dobra, nn.BatchNorm2d(out_ch)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        # ReLU comum (nao Leaky) no caminho ascendente, seguindo o artigo.
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


def up_mode_from_state(state):
    """Descobre com que modo um checkpoint foi treinado, olhando as formas salvas.

    Existe para que os checkpoints antigos continuem carregaveis depois da troca: sem
    isso, load_state_dict falharia com erro de formas e os modelos ja treinados
    viravam lixo. ConvTranspose2d guarda peso (in, out, 4, 4); a Conv2d do caminho
    'resize' guarda (out, in, 3, 3) -- o tamanho do kernel separa os dois casos.

    Aceita tanto o dict do gerador quanto o checkpoint inteiro.
    """
    sd = state.get('generator', state)
    for k, v in sd.items():
        if k.startswith('ups.0.') and v.dim() == 4:
            return 'transpose' if v.shape[-1] == 4 else 'resize'
    return 'resize'


def in_channels_from_state(state):
    """Quantos canais de entrada um checkpoint espera, pelas formas salvas.

    Mesma razao de up_mode_from_state: um modelo treinado com paleta tem 13 canais de
    entrada e um sem paleta tem 1, e carregar um no outro falharia com erro de formas.
    O peso da primeira convolucao do encoder tem forma (out, in, k, k).
    """
    sd = state.get('generator', state)
    for k, v in sd.items():
        if k.startswith('downs.0.') and v.dim() == 4:
            return v.shape[1]
    return 1


def _down_filters(n_down):
    """Quantidade de canais de cada nivel do encoder: [64, 128, 256, 512, 512, ...].

    Dobra a cada nivel ate o teto de 512. A logica e compensar a perda de resolucao:
    a cada bloco a imagem encolhe 4x em area, entao dobrar os canais mantem mais ou
    menos a mesma capacidade de representacao por nivel. O teto existe porque alem
    de 512 o custo de memoria cresce sem ganho pratico.

    Para n_down=8 (imagem 256) devolve exatamente o down_stack do notebook:
    [64, 128, 256, 512, 512, 512, 512, 512].
    """
    filters, ch = [], 64
    for _ in range(n_down):
        filters.append(min(ch, 512))
        ch *= 2
    return filters


class Generator(nn.Module):
    """U-Net: encoder-decoder com skip connections.

    O caminho e um funil e depois um funil invertido. Com imagem 256x256:

        256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1      (8 downsamples)
          1 ->   2 ->  4 ->  8 -> 16 -> 32 -> 64 -> 128 -> 256 (7 upsamples + a camada final)

    POR QUE SKIP CONNECTIONS

    Num encoder-decoder puro, tudo precisa atravessar o gargalo de 1x1 -- a imagem
    inteira comprimida num unico vetor. Detalhe espacial fino nao sobrevive a isso.
    As skip connections ligam cada nivel do encoder ao nivel de mesma resolucao do
    decoder (via torch.cat, concatenando canais), dando ao decoder acesso direto ao
    detalhe que o encoder viu antes de comprimir.

    Na traducao esboco -> sprite isso e essencial: o contorno de entrada e a saida
    estao alinhados pixel a pixel, e as skips preservam esse alinhamento.

    A profundidade e derivada de img_size, entao treinar em 128 ou 64 (recomendacao 3
    da avaliacao) nao exige mexer na arquitetura.
    """

    def __init__(self, in_channels=1, out_channels=3, img_size=256, up_mode='resize',
                 skip_drop=0):
        super().__init__()
        # Cada downsample divide a resolucao por 2, e o gargalo tem de chegar em 1x1.
        # Logo o numero de niveis e log2(img_size): 256 -> 8, 128 -> 7, 64 -> 6.
        n_down = int(math.log2(img_size))
        if 2 ** n_down != img_size:
            raise ValueError(f'img_size deve ser potencia de 2, recebi {img_size}')

        # Guardado porque a inferencia precisa saber quais condicionantes montar:
        # 1 canal e so o esboco, 13 inclui a paleta, 16 inclui tambem o rabisco.
        # Ler isso do modelo evita reinspecionar o state_dict em quem ja o carregou.
        self.in_channels = in_channels

        # Quantas skip connections de MAIOR resolucao sao anuladas. Ver _forward:
        # elas sao o que prende a saida a geometria exata da entrada.
        self.skip_drop = skip_drop

        down_f = _down_filters(n_down)

        # --- Encoder -------------------------------------------------------------
        self.downs = nn.ModuleList()
        prev = in_channels  # 1 canal: o esboco e em tons de cinza
        for i, f in enumerate(down_f):
            # Dois blocos ficam SEM BatchNorm:
            #
            # i == 0 (primeiro): e a convencao do pix2pix. Normalizar logo na entrada
            #   apagaria informacao de contraste absoluto que ainda e util.
            #
            # innermost (ultimo): no gargalo a resolucao e 1x1 e, com batch pequeno,
            #   cada canal tem um unico valor -- a variancia e zero e nao ha o que
            #   normalizar. O PyTorch levanta erro nesse caso. O TensorFlow aceita em
            #   silencio e devolve beta, isto e, uma CONSTANTE: verifiquei que duas
            #   entradas completamente diferentes produzem saida identica. Ou seja, no
            #   notebook com BATCH_SIZE=1 o gargalo nao carregava informacao nenhuma e
            #   tudo trafegava pelas skip connections.
            #   Omitir a norma aqui e o que a implementacao de referencia do pix2pix em
            #   PyTorch faz. E a unica divergencia de arquitetura em relacao ao TF, e
            #   responde pelos 1.024 parametros a menos (gamma + beta de 512 canais).
            innermost = (i == len(down_f) - 1)
            self.downs.append(Downsample(prev, f, batchnorm=not (i == 0 or innermost)))
            prev = f

        # --- Decoder -------------------------------------------------------------
        # Espelha o encoder, de tras para frente e sem o ultimo nivel (o gargalo ja
        # foi produzido pelo encoder; o decoder parte dele).
        self.ups = nn.ModuleList()
        for i, f in enumerate(reversed(down_f[:-1])):
            self.ups.append(Upsample(prev, f, dropout=(i < 3), mode=up_mode))
            # A entrada do PROXIMO bloco ascendente tem o dobro de canais, porque a
            # saida deste sera concatenada com a skip connection correspondente.
            prev = f * 2

        # Camada final: dobra a resolucao uma ultima vez (128 -> 256) e reduz para os
        # 3 canais RGB. Tem bias (ao contrario dos blocos), por nao ter BatchNorm depois.
        # Segue o mesmo modo dos demais: e a ultima camada que escreve o pixel, entao
        # deixa-la como transposta reintroduziria a listra que os blocos evitaram.
        if up_mode == 'resize':
            self.last = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.Conv2d(prev, out_channels, 3, stride=1, padding=1))
        else:
            self.last = nn.ConvTranspose2d(prev, out_channels, 4, stride=2, padding=1)

        self.apply(_init_weights)

    def forward(self, x):
        # Descida, guardando a saida de cada nivel para as skip connections.
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Subida. reversed(skips[:-1]) descarta o gargalo (que ja e o proprio x) e
        # percorre os demais de tras para frente, casando cada bloco ascendente com o
        # nivel do encoder de mesma resolucao.
        #
        # SKIP_DROP: por que anular as skips de maior resolucao
        #
        # A skip de 128x128 entrega ao decoder, pixel a pixel, o que havia naquele
        # mesmo pixel da entrada. E o que da ao pix2pix a fidelidade de contorno --
        # e tambem o que o impede de MOVER qualquer coisa de lugar. Medido em
        # 07/11/2026: a deformacao aplicada a entrada chega a saida com 108% a 127%
        # da intensidade, isto e, o modelo traca o esboco em vez de redesenha-lo.
        # Para transformar rabisco torto em forma plausivel o decoder precisa poder
        # ignorar onde a linha estava, e so consegue isso se a geometria fina nao
        # lhe for entregue de bandeja.
        #
        # Anular (multiplicar por zero) em vez de remover: as formas do state_dict
        # ficam intactas, entao todo checkpoint antigo continua carregando e as duas
        # variantes sao comparaveis parametro a parametro. O custo e um punhado de
        # pesos que aprendem a ignorar canais mortos.
        #
        # As ULTIMAS iteracoes do zip sao as de maior resolucao (reversed), por isso
        # o corte conta a partir do fim.
        n_ups = len(self.ups)
        for i, (up, skip) in enumerate(zip(self.ups, reversed(skips[:-1]))):
            x = up(x)
            if self.skip_drop and i >= n_ups - self.skip_drop:
                skip = torch.zeros_like(skip)
            x = torch.cat([x, skip], dim=1)  # dim=1 = eixo de canais (N, C, H, W)

        # tanh leva a saida para [-1, 1], a mesma faixa em que as imagens reais foram
        # normalizadas em pix2pix_data.py. As duas pontas precisam casar para a perda
        # L1 fazer sentido.
        return torch.tanh(self.last(x))


class Discriminator(nn.Module):
    """PatchGAN 70x70: julga trechos da imagem, nao a imagem inteira.

    Em vez de devolver um unico numero ("real ou falso?"), devolve um mapa 30x30 em
    que cada posicao julga um trecho de 70x70 pixels da entrada. A perda e a media
    sobre esse mapa.

    POR QUE POR TRECHOS

    A L1 do gerador ja cuida do acerto em escala global (cor e forma gerais). O que
    falta e nitidez LOCAL -- textura, bordas limpas. Um discriminador focado em
    trechos pequenos pressiona exatamente onde a L1 e fraca, e de quebra tem muito
    menos parametros (2,8M contra 54,4M do gerador) e roda mais rapido.

    O "70x70" e o campo receptivo efetivo de cada posicao do mapa de saida,
    resultado da pilha de convolucoes abaixo -- nao um parametro explicito.

    A entrada e o par CONCATENADO (esboco + imagem a julgar) = 1 + 3 = 4 canais.
    E isso que torna a GAN condicional: o discriminador nunca julga uma imagem no
    vacuo, sempre em relacao ao esboco que a originou.
    """

    def __init__(self, in_channels=1, target_channels=3):
        super().__init__()
        self.down1 = Downsample(in_channels + target_channels, 64, batchnorm=False)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)

        # A partir daqui o stride vira 1: a resolucao para de cair e a rede so
        # aprofunda o julgamento local. O ZeroPad2d(1) equivale ao
        # tf.keras.layers.ZeroPadding2D() e, com padding=0 nas convolucoes, reproduz
        # o encolhimento controlado do notebook:
        #   32 -> pad 34 -> conv k4 -> 31 -> pad 33 -> conv k4 -> 30
        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(256, 512, 4, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)

        # Saida com 1 canal e SEM sigmoid: sao logits. A funcao de perda
        # (BCEWithLogitsLoss) aplica o sigmoid internamente, de forma
        # numericamente estavel.
        self.last = nn.Conv2d(512, 1, 4, stride=1, padding=0)

        self.apply(_init_weights)

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)  # (N, 4, 256, 256)
        x = self.down3(self.down2(self.down1(x)))
        x = self.act(self.bn(self.conv(self.pad(x))))
        return self.last(self.pad(x))  # (N, 1, 30, 30)


# Entropia cruzada binaria que recebe logits (aplica o sigmoid por dentro, de forma
# estavel). Equivale ao tf.keras.losses.BinaryCrossentropy(from_logits=True).
_bce = nn.BCEWithLogitsLoss()


def generator_loss(disc_generated_output, gen_output, target, lam=100):
    """Perda do gerador: enganar o discriminador + ficar perto do alvo real.

    Devolve (total, parte_adversarial, parte_L1) -- as tres sao registradas no
    TensorBoard porque so o total nao diz qual lado esta dominando.

    gan_loss: o gerador quer que o discriminador classifique sua saida como REAL,
    entao o alvo aqui e um mapa de uns. Repare que e o oposto exato do alvo usado em
    discriminator_loss para a mesma imagem -- e literalmente a oposicao da GAN.

    l1_loss: erro absoluto medio contra o sprite verdadeiro. Multiplicada por
    lam=100 porque, sem esse peso, a perda adversarial domina e o gerador produz
    imagens nitidas mas sem relacao com o alvo.
    """
    gan_loss = _bce(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = torch.mean(torch.abs(target - gen_output))
    return gan_loss + lam * l1_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """Perda do discriminador: acertar quem e real e quem e falso.

    Duas parcelas somadas: classificar o sprite verdadeiro como real (alvo = uns) e
    a saida do gerador como falsa (alvo = zeros).

    Para interpretar o valor no TensorBoard: ~1,38 (2 * ln 2) e o chute aleatorio.
    Muito abaixo disso significa que o discriminador esta vencendo com folga e
    sobrando pouco gradiente util para o gerador -- foi o que aconteceu na rodada das
    sombras, em que disc_loss caiu de 0,65 para 0,47 enquanto a do gerador subia.
    """
    real_loss = _bce(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = _bce(disc_generated_output, torch.zeros_like(disc_generated_output))
    return real_loss + generated_loss
