"""Treino do pix2pix em PyTorch, portado de img_gen_contours.ipynb.

Uso tipico:

    python pix2pix_train.py --steps 40000
    python pix2pix_train.py --steps 40000 --resume      # retoma do ultimo checkpoint
    python pix2pix_train.py --steps 40000 --img-size 128 --amp

As curvas vao para logs/fit/<run>/ e as previas para logs/previews/<run>/, ambos
visiveis com `tensorboard --logdir logs/fit`.

COMO UM PASSO DE TREINO DE GAN FUNCIONA

Diferente de uma rede comum, aqui ha DUAS redes com objetivos opostos, atualizadas
no mesmo passo. A ordem importa e e sutil -- ver o comentario do laco principal.

O treino e contado em PASSOS (lotes), nao em epocas: 40.000 passos com lote 1 sobre
3.289 pares equivalem a ~12 passagens pelo dataset. E a convencao do pix2pix, e
deixa o custo do treino independente do tamanho do conjunto.
"""
import argparse
import time
from pathlib import Path

import torch

from pix2pix_data import (IMG_SIZE, PALETTE_K, PATH_INPUT, PATH_REAL, SPLIT_SEED,
                          TEST_SIZE, make_loaders)
from pix2pix_model import Discriminator, Generator, discriminator_loss, generator_loss


def parse_args():
    """Hiperparametros e opcoes de execucao.

    Os defaults reproduzem a configuracao do notebook (e do artigo do pix2pix), para
    que esta rodada seja comparavel com a das sombras de janeiro.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=1,
                   help='o pix2pix original usa 1; o U-Net rende melhor assim')
    p.add_argument('--img-size', type=int, default=IMG_SIZE,
                   help='potencia de 2; 128 ou 64 treinam muito mais rapido')
    # lr e beta1 do artigo. beta1=0.5 (em vez do 0.9 padrao do Adam) reduz a inercia
    # do otimizador: em GAN os dois jogadores mudam o tempo todo, e um otimizador com
    # memoria longa fica perseguindo um alvo que ja se moveu.
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--lam', type=float, default=100.0, help='peso da perda L1')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--input-dir', default=PATH_INPUT)
    p.add_argument('--real-dir', default=PATH_REAL)
    p.add_argument('--test-size', type=int, default=TEST_SIZE)
    p.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    p.add_argument('--device', default=None, help='cuda ou cpu (default: detecta)')
    p.add_argument('--amp', action='store_true',
                   help='precisao mista; acelera bastante em GPU Ampere ou mais nova')
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--preview-every', type=int, default=1000)
    p.add_argument('--ckpt-every', type=int, default=5000)
    p.add_argument('--keep-last', type=int, default=3,
                   help='quantos checkpoints manter (cada um ocupa centenas de MB)')
    p.add_argument('--scribble', action='store_true',
                   help='3 canais com manchas de cor posicionadas, dizendo ONDE cada '
                        'cor vai; complementa --palette, que diz quais')
    p.add_argument('--palette', action='store_true',
                   help='condiciona a cor por paleta de 4 cores extraida do alvo '
                        '(exige palettes.npz; rode tools_palette.py antes)')
    p.add_argument('--skip-drop', type=int, default=0, metavar='N',
                   help='anula as N skip connections de MAIOR resolucao. 0 (default) e '
                        'o pix2pix do artigo, fiel ao contorno da entrada; 1 ou 2 soltam '
                        'o decoder da geometria fina, para ele poder corrigir a forma de '
                        'um rabisco torto em vez de traca-lo')
    p.add_argument('--up-mode', default='transpose', choices=['resize', 'transpose'],
                   help="'transpose' (padrao) reproduz o artigo. 'resize' corta a listra "
                        "do artefato de tabuleiro (razao de gradiente 1,23 -> 1,10), mas o "
                        "gerador fica menor (39,2M contra 54,4M) e com lam=15 o "
                        "discriminador venceu e a saida colapsou numa paleta magenta. "
                        "Se for usar, suba o lam para perto de 50 antes.")
    p.add_argument('--resume', action='store_true')
    p.add_argument('--run-name', default=None)
    return p.parse_args()


def pick_device(requested):
    """Escolhe GPU se houver, avisando alto quando cair para CPU.

    O aviso importa: este modelo roda a ~0,036 s/passo na RTX 3060 e a ~47 s/passo
    na CPU desta maquina. Comecar 40.000 passos sem perceber que a GPU nao foi
    detectada custaria semanas em vez de minutos.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    print('AVISO: CUDA nao disponivel, caindo para CPU (ordens de grandeza mais lento).')
    return torch.device('cpu')


def cycle(loader):
    """Repete o loader indefinidamente, como o .repeat() do tf.data.

    O laco de treino conta passos, nao epocas, entao precisa de uma fonte infinita de
    lotes. A cada volta o DataLoader reembaralha sozinho (shuffle=True), de modo que
    as passagens seguintes veem o dataset numa ordem nova.
    """
    while True:
        for batch in loader:
            yield batch


def to_display(t):
    """De [-1, 1] CHW (formato da rede) para [0, 1] HWC (formato do matplotlib).

    O squeeze no fim remove o eixo de canal quando ele tem tamanho 1, porque o
    imshow espera (H, W) para imagens em tons de cinza, nao (H, W, 1).
    """
    img = t.detach().float().cpu().permute(1, 2, 0).mul(0.5).add(0.5).clamp(0, 1)
    return img.squeeze(-1) if img.shape[-1] == 1 else img


@torch.no_grad()  # amostragem nao precisa de gradiente: economiza memoria e tempo
def save_preview(generator, example, path, step):
    """Salva um painel (entrada | alvo real | gerado), equivalente ao generate_images().

    O notebook chama o gerador com training=True mesmo na amostragem, e isso NAO e
    descuido: no pix2pix o dropout continua ligado na inferencia e e a unica fonte de
    aleatoriedade do gerador. A BatchNorm em modo treino tambem usa as estatisticas do
    proprio lote em vez das acumuladas. Replicado aqui com generator.train().

    Devolve a predicao em [0, 1] para que o chamador registre a imagem no TensorBoard.
    """
    # Imports locais de proposito: no Windows cada worker do DataLoader reimporta
    # este modulo, e o matplotlib (e o tensorboard, que arrasta o TensorFlow junto)
    # custariam segundos de inicializacao em cada processo filho sem serem usados la.
    # Medido: 6 cargas do TensorFlow por rodada antes da mudanca, 2 depois.
    import matplotlib
    matplotlib.use('Agg')  # sem janela: o script salva PNG em vez de exibir
    import matplotlib.pyplot as plt

    inp, tar = example
    was_training = generator.training
    generator.train()
    pred = generator(inp)
    generator.train(was_training)  # restaura o modo anterior, seja qual for

    # O canal 0 e o esboco; de 1 em diante, quando existem, sao os K*3 canais da
    # paleta, constantes no espaco. Exibi-los como imagem daria erro de forma -- e nao
    # diria nada, por serem chapados. Viram um painel proprio: ver a paleta PEDIDA ao
    # lado da cor que saiu e o que torna a previa util nesta configuracao.
    titles = ['Input Image', 'Ground Truth', 'Predicted Image']
    images = [to_display(inp[0, :1]), to_display(tar[0]), to_display(pred[0])]
    paleta = None
    n_pal = PALETTE_K * 3
    if inp.shape[1] >= 1 + n_pal:
        faixa = inp[0, 1:1 + n_pal, 0, 0].detach().float().cpu().mul(0.5).add(0.5).clamp(0, 1)
        paleta = faixa.reshape(-1, 3).unsqueeze(0)  # (1, K, 3): uma linha de K cores
        titles.insert(1, 'Palette')
        images.insert(1, paleta)
    if inp.shape[1] == 1 + n_pal + 3 or inp.shape[1] == 4:
        titles.insert(len(titles) - 2, 'Scribble')
        images.insert(len(images) - 2, to_display(inp[0, -3:]))

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for ax, title, img in zip(axes, titles, images):
        ax.set_title(title)
        if img is paleta:
            ax.imshow(img, aspect='auto')
        else:
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')
    fig.suptitle(f'step {step}')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)  # sem isso as figuras acumulam na memoria ao longo do treino
    return torch.stack([pred[0].detach().float().cpu().mul(0.5).add(0.5).clamp(0, 1)])


def save_checkpoint(ckpt_dir, step, generator, discriminator, opt_g, opt_d, scaler, keep_last):
    """Grava o estado completo do treino e apaga os checkpoints mais antigos.

    Salvar so os pesos nao bastaria para retomar: o estado dos otimizadores Adam
    (medias moveis de gradiente por parametro) e parte do treino, e recomecar com ele
    zerado provoca um solavanco nas perdas. Por isso o dicionario leva tudo.

    O nome tem o passo com zeros a esquerda (ckpt-005000.pt) para que a ordem
    alfabetica coincida com a ordem numerica -- e o que faz latest_checkpoint()
    funcionar com um simples sorted().

    keep_last limita o espaco em disco: cada arquivo tem ~686 MB (54M parametros do
    gerador mais os estados do Adam). A rodada das sombras deixou 8 deles, 5,2 GB.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'ckpt-{step:06d}.pt'
    torch.save({
        'step': step,
        # Gravado explicitamente porque, diferente de up_mode e in_channels, skip_drop
        # nao altera a forma de nenhum tensor -- nao da para deduzi-lo do state_dict
        # depois. Sem isto, carregar o modelo para inferencia reconstruiria a variante
        # errada em silencio, com as skips ligadas e saida diferente da treinada.
        'skip_drop': getattr(generator, 'skip_drop', 0),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'generator_optimizer': opt_g.state_dict(),
        'discriminator_optimizer': opt_d.state_dict(),
        'scaler': scaler.state_dict(),
    }, path)

    existing = sorted(ckpt_dir.glob('ckpt-*.pt'))
    for old in existing[:-keep_last] if keep_last > 0 else []:
        old.unlink()
    return path


def latest_checkpoint(ckpt_dir):
    """O checkpoint de passo mais alto, ou None. Depende do nome com zeros a esquerda."""
    found = sorted(ckpt_dir.glob('ckpt-*.pt'))
    return found[-1] if found else None


def main():
    from torch.utils.tensorboard import SummaryWriter  # local: ver nota em save_preview

    args = parse_args()
    device = pick_device(args.device)
    print(f'dispositivo: {device}'
          + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''))
    if device.type == 'cuda':
        # Todos os lotes tem exatamente a mesma forma aqui, entao vale deixar o cuDNN
        # medir os algoritmos disponiveis no primeiro passo e reusar o mais rapido.
        # Com formas variaveis isso seria contraproducente (remediria a cada forma nova).
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size, num_workers=args.workers, img_size=args.img_size,
        path_input=args.input_dir, path_real=args.real_dir,
        test_size=args.test_size, seed=args.split_seed,
        pin_memory=(device.type == 'cuda'), palette=args.palette,
        scribble=args.scribble)

    # 1 canal de esboco + K*3 da paleta. O discriminador recebe os mesmos canais de
    # condicao: sem a paleta ele nao teria como julgar se a saida respeitou as cores
    # pedidas, e o termo adversarial ficaria cego justamente para o que mudou aqui.
    in_ch = 1 + (PALETTE_K * 3 if args.palette else 0) + (3 if args.scribble else 0)

    generator = Generator(in_channels=in_ch, img_size=args.img_size, up_mode=args.up_mode,
                          skip_drop=args.skip_drop).to(device)
    discriminator = Discriminator(in_channels=in_ch).to(device)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f'gerador: {n_params / 1e6:.1f}M parametros | '
          f'discriminador: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.1f}M')

    # Dois otimizadores SEPARADOS, um por rede. E o que permite atualizar cada uma
    # com sua propria perda; um otimizador unico sobre todos os parametros faria as
    # duas minimizarem a mesma coisa, e a oposicao da GAN deixaria de existir.
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Com --amp os calculos rodam em 16 bits onde for seguro. Gradientes em 16 bits
    # podem virar zero por falta de alcance numerico, entao o GradScaler multiplica a
    # perda por um fator grande antes do backward e desfaz antes do passo do
    # otimizador. Com enabled=False e um objeto inerte, sem custo.
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    run_name = args.run_name or f'{Path(args.input_dir).name}-torch-{time.strftime("%Y%m%d-%H%M%S")}'
    # Nome derivado do diretorio de entrada: treinar com contornos e com sombras nao
    # sobrescreve um ao outro. O sufixo _training_checkpoints casa com a regra que ja
    # existe no .gitignore.
    ckpt_dir = Path(f'{Path(args.input_dir).name}_torch_training_checkpoints')

    start_step = 0
    if args.resume:
        found = latest_checkpoint(ckpt_dir)
        if found is None:
            print(f'--resume pedido, mas nao ha checkpoint em {ckpt_dir}; comecando do zero.')
        else:
            state = torch.load(found, map_location=device)
            # O modo do decoder e propriedade do checkpoint, nao da linha de comando:
            # retomar com o modo errado falharia com erro de formas. Reconstroi o
            # gerador no modo salvo e avisa quando difere do pedido.
            salvo = up_mode_from_state(state)
            if salvo != args.up_mode:
                print(f'aviso: {found.name} foi treinado com up-mode={salvo}; '
                      f'ignorando --up-mode {args.up_mode} para poder retomar.')
                generator = Generator(in_channels=in_ch, img_size=args.img_size, up_mode=salvo,
                                      skip_drop=args.skip_drop).to(device)
                opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr,
                                         betas=(args.beta1, 0.999))
            generator.load_state_dict(state['generator'])
            discriminator.load_state_dict(state['discriminator'])
            opt_g.load_state_dict(state['generator_optimizer'])
            opt_d.load_state_dict(state['discriminator_optimizer'])
            scaler.load_state_dict(state['scaler'])
            start_step = state['step']
            print(f'retomando de {found.name} no passo {start_step}')

    writer = SummaryWriter(f'logs/fit/{run_name}')
    preview_dir = Path('logs/previews') / run_name
    print(f'logs desta rodada: logs/fit/{run_name}')

    # Um exemplo FIXO do conjunto de teste, tirado uma unica vez. Usar sempre o mesmo
    # par faz as previas contarem uma historia: a evolucao ao longo dos passos vem da
    # rede melhorando, nao de a imagem ter mudado.
    example = next(iter(test_loader))
    example = (example[0].to(device), example[1].to(device))

    # Listas materializadas uma vez: sao passadas em todo passo para backward(inputs=)
    # e reconstrui-las 40.000 vezes seria desperdicio.
    gen_params = list(generator.parameters())
    disc_params = list(discriminator.parameters())

    generator.train()
    discriminator.train()
    batches = cycle(train_loader)
    start = time.time()
    window_start = start

    for step in range(start_step, args.steps):
        input_image, target = next(batches)
        # non_blocking=True deixa a copia CPU->GPU acontecer em paralelo com o
        # calculo; so tem efeito junto com pin_memory no DataLoader.
        input_image = input_image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # set_to_none=True libera os tensores de gradiente em vez de preenche-los com
        # zeros: um pouco mais rapido e mais economico em memoria.
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)

        # --- Um unico forward alimenta as duas perdas ----------------------------
        with torch.amp.autocast(device.type, enabled=args.amp):
            gen_output = generator(input_image)

            # O discriminador julga duas vezes: o par verdadeiro e o par falsificado.
            disc_real_output = discriminator(input_image, target)
            disc_generated_output = discriminator(input_image, gen_output)

            # Repare que disc_generated_output entra nas DUAS perdas, com alvos
            # opostos: o gerador quer que de "real", o discriminador quer que de
            # "falso". E a oposicao da GAN, expressa em duas linhas.
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target, args.lam)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # --- Dois backwards sobre o MESMO grafo ----------------------------------
        # Reproduz os dois GradientTape do notebook. Dois detalhes importam:
        #
        # retain_graph=True no primeiro: por padrao o PyTorch libera o grafo durante o
        #   backward, e o segundo backward nao teria mais o que percorrer.
        #
        # inputs=: restringe a acumulacao de gradiente aos parametros daquela rede.
        #   Sem isso, o backward da perda do gerador tambem acumularia gradiente no
        #   discriminador (a perda passa por ele), sujando a atualizacao seguinte.
        #
        # E por que nao fazer o passo do discriminador primeiro, como e comum? Porque
        # mudaria o resultado: as duas perdas aqui sao medidas com os pesos ANTES de
        # qualquer atualizacao, e so depois os dois passos sao aplicados. Atualizar o
        # discriminador antes faria o gerador ser avaliado por um juiz ja diferente
        # daquele que produziu seu gradiente.
        scaler.scale(gen_total_loss).backward(retain_graph=True, inputs=gen_params)
        scaler.scale(disc_loss).backward(inputs=disc_params)
        scaler.step(opt_g)
        scaler.step(opt_d)
        scaler.update()  # um unico update() para os dois: padrao do AMP multi-otimizador

        if args.log_every and step % args.log_every == 0:
            # As tres parcelas do gerador vao separadas porque so o total nao revela
            # qual lado domina. Na rodada das sombras foi assim que ficou visivel que
            # a parte adversarial subia enquanto a L1 estagnava.
            # .item() traz o valor para a CPU e sincroniza com a GPU -- barato aqui,
            # mas e o motivo de nao se logar tensores dentro de lacos quentes.
            writer.add_scalar('gen_total_loss', gen_total_loss.item(), step)
            writer.add_scalar('gen_gan_loss', gen_gan_loss.item(), step)
            writer.add_scalar('gen_l1_loss', gen_l1_loss.item(), step)
            writer.add_scalar('disc_loss', disc_loss.item(), step)

        if args.preview_every and (step % args.preview_every == 0 or step + 1 == args.steps):
            grid = save_preview(generator, example, preview_dir / f'step-{step:06d}.png', step)
            writer.add_images('prediction', grid, step)
            elapsed = time.time() - window_start
            done = step - start_step
            if done:
                # Dois tempos: o da janela desde a ultima previa (ritmo instantaneo) e
                # a media acumulada, que inclui a inicializacao e so converge no fim.
                print(f'step {step}: {elapsed:.1f}s desde a ultima previa | '
                      f'{(time.time() - start) / done:.3f} s/passo | '
                      f'l1={gen_l1_loss.item():.4f} disc={disc_loss.item():.4f}')
            else:
                print(f'step {step}: inicio | l1={gen_l1_loss.item():.4f}')
            window_start = time.time()

        # step + 1 para que o checkpoint caia em multiplos redondos (5000, 10000, ...)
        # e nao em 4999. O numero gravado e o de passos ja completados.
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            path = save_checkpoint(ckpt_dir, step + 1, generator, discriminator,
                                   opt_g, opt_d, scaler, args.keep_last)
            print(f'  checkpoint salvo: {path}')

    # Checkpoint final, sempre -- inclusive com --ckpt-every 0, para nao terminar um
    # treino inteiro sem nada gravado em disco.
    save_checkpoint(ckpt_dir, args.steps, generator, discriminator, opt_g, opt_d,
                    scaler, args.keep_last)
    writer.close()
    total = time.time() - start
    print(f'\nconcluido: {args.steps - start_step} passos em {total / 3600:.2f} h '
          f'({total / max(1, args.steps - start_step):.3f} s/passo)')


if __name__ == '__main__':
    # O guard e obrigatorio no Windows: o DataLoader com workers > 0 cria processos
    # por 'spawn', e cada filho reimporta este modulo. Sem o guard, cada filho
    # executaria main() de novo e criaria mais filhos, em recursao infinita.
    main()
