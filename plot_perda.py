"""Curva de perda das rodadas de LoRA, para comparar treino saudavel com divergencia.

POR QUE ESTE GRAFICO EXISTE

A rodada de 07/11/2026 divergiu por treinar a UNet em fp16 sem GradScaler, e o unico
sinal antes de gastar 3 h foi a perda SUBINDO. Em difusao a perda e dominada pela
variancia do timestep sorteado, entao ela e ruidosa por natureza e o olho num numero
isolado nao decide nada -- o que denuncia e a TENDENCIA. Este grafico torna essa
tendencia visivel de relance, que e a unica forma de pegar o problema cedo.

Le os CSV que lora_train.py grava (`<saida>/perda.csv`) e tambem os logs antigos em
texto, de antes do CSV existir.

    python plot_perda.py --saida predictions/difusao/08-curva-de-perda.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Paleta categorica de referencia, slots 1 a 3 (azul, laranja, aqua). Os tres
# primeiros slots sao os validados para todos os pares em modo claro.
CORES = ['#2a78d6', '#eb6834', '#1baf7a']
SUPERFICIE = '#fcfcfb'
TINTA = '#0b0b0b'
TINTA_FRACA = '#52514e'
GRADE = '#e3e2dd'


def de_csv(p):
    d = np.genfromtxt(p, delimiter=',', names=True)
    return np.atleast_1d(d['passo']), np.atleast_1d(d['perda'])


def de_log(p):
    """Logs antigos: linhas 'passo N/M | perda X | ...' no stdout."""
    passos, perdas = [], []
    for linha in Path(p).read_text(encoding='utf-8', errors='ignore').splitlines():
        m = re.search(r'passo (\d+)/\d+ \| perda ([\d.]+)', linha)
        if m:
            passos.append(int(m.group(1)))
            perdas.append(float(m.group(2)))
    return np.array(passos), np.array(perdas)


def suavizar(y, n=9):
    """Media movel centrada que PRESERVA AS PONTAS.

    np.convolve com mode='same' preenche a borda com zero, e a media despenca nos
    extremos -- erro observado em 08/11/2026: a ponta da curva marcava 0,029 quando
    o valor real era 0,051, e e justamente na ponta que fica o rotulo que o leitor
    anota. Aqui cada ponto da borda e a media so da janela que existe de fato.
    """
    if len(y) < 3:
        return np.asarray(y, float)
    y = np.asarray(y, float)
    m = max(3, min(n, len(y)))
    meia = m // 2
    return np.array([y[max(0, i - meia):min(len(y), i + meia + 1)].mean()
                     for i in range(len(y))])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--saida', default='predictions/difusao/08-curva-de-perda.png')
    p.add_argument('--log-divergiu',
                   default=r'C:\Users\Lomir\AppData\Local\Temp\claude\D--Projetos-Github-eevee\6116fa34-1652-486d-b8ec-f1811ba28098\scratchpad\lora.log')
    p.add_argument('--log-corrigida',
                   default=r'C:\Users\Lomir\AppData\Local\Temp\claude\D--Projetos-Github-eevee\6116fa34-1652-486d-b8ec-f1811ba28098\scratchpad\lora2.log')
    # O CSV mora aqui, e nao na pasta da rodada: os pesos de 9k a 12k foram
    # apagados em 08/11/2026 por nao terem melhorado nada, mas a curva deles
    # continua valendo como registro.
    p.add_argument('--csv-continuacao',
                   default='predictions/difusao/perda-continuacao-12k.csv')
    args = p.parse_args()

    series = []
    for rot, carregar, origem, desloc in (
            ('fp16 sem GradScaler — divergiu', de_log, args.log_divergiu, 0),
            ('corrigida (fp32 + autocast)', de_log, args.log_corrigida, 0),
            ('continuação até 12k', de_csv, args.csv_continuacao, 8000)):
        if not Path(origem).exists():
            print(f'  (pulando "{rot}": {origem} nao existe ainda)')
            continue
        x, y = carregar(origem)
        if len(x) == 0:
            print(f'  (pulando "{rot}": sem pontos)')
            continue
        series.append((rot, x + desloc, y))

    if not series:
        raise SystemExit('nenhuma serie encontrada')

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    pontas = []
    for i, (rot, x, y) in enumerate(series):
        cor = CORES[i % len(CORES)]
        sy = suavizar(y)
        ax.plot(x, y, color=cor, alpha=0.22, linewidth=1.0)        # serie crua, recessiva
        ax.plot(x, sy, color=cor, linewidth=2.0, label=rot)
        pontas.append((float(x[-1]), float(sy[-1]), cor))

    # Rotulo direto no fim de cada linha, alem da legenda: identidade nunca so por cor.
    # Duas rodadas que terminam no mesmo ponto escreveriam um numero por cima do outro,
    # entao os rotulos proximos sao afastados na vertical antes de desenhar.
    faixa = ax.get_ylim()[1] - ax.get_ylim()[0] or 1.0
    ordem = sorted(range(len(pontas)), key=lambda k: pontas[k][1])
    deslocamento = [0.0] * len(pontas)
    minimo = faixa * 0.035
    for a, b in zip(ordem, ordem[1:]):
        folga = (pontas[b][1] + deslocamento[b]) - (pontas[a][1] + deslocamento[a])
        if folga < minimo:
            deslocamento[b] += minimo - folga
    for (px, py, cor), dy in zip(pontas, deslocamento):
        ax.annotate(f'{py:.3f}', xy=(px, py + dy), xytext=(7, 0),
                    textcoords='offset points', color=cor, fontsize=9, va='center')

    ax.set_xlabel('passo de treino', color=TINTA_FRACA, fontsize=10)
    ax.set_ylabel('perda (MSE do ruido previsto)', color=TINTA_FRACA, fontsize=10)
    ax.set_title('Perda ao longo do treino da LoRA',
                 color=TINTA, fontsize=13, pad=30, loc='left', fontweight='bold')
    ax.text(0, 1.015, 'em difusão a perda é ruidosa por natureza — o que denuncia '
                      'problema é a tendência, não o valor',
            transform=ax.transAxes, color=TINTA_FRACA, fontsize=9.5, va='bottom')

    ax.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    for lado in ('left', 'bottom'):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA_FRACA, labelsize=9)

    leg = ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(TINTA_FRACA)

    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destino, facecolor=SUPERFICIE)
    print(f'-> {destino}')

    # Tabela, para o numero ficar legivel sem depender da leitura do grafico.
    print(f'{chr(10)}{"rodada":34s} {"inicio":>8s} {"fim":>8s} {"tendencia":>12s}')
    for rot, x, y in series:
        ini, fim = np.mean(y[:10]), np.mean(y[-10:])
        print(f'{rot:34s} {ini:8.4f} {fim:8.4f} {(fim - ini) / ini * 100:+11.1f}%')


if __name__ == '__main__':
    main()
