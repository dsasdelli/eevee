# Bancada de medicao

Scripts usados em 17/10/2026 para avaliar o gerador contorno -> sprite. Medem o que o
`pix2pix_predict.py --eval` nao mede: ele so calcula L1 medio, e L1 nao distingue cor
certa de cor morta nem estrutura de artefato.

    python eval/eval_color.py <ckpt> [<ckpt> ...]
        L1, saturacao da predicao e do alvo, e a separacao luminancia/crominancia,
        sobre os 1.000 pares de teste. Roda cada checkpoint nos dois modos do gerador
        (train/eval), porque o dropout fica ligado na amostragem do pix2pix e os
        numeros diferem entre os modos.

    python eval/silhouette_eval.py <ckpt> [<ckpt> ...]
        O teste que importa para o produto: entrada de SILHUETA LIMPA, parecida com o
        que um usuario desenha. Mede saturacao e estrutura interna dentro da silhueta.

    python eval/domain_gap.py <ckpt> [n]
        Grade visual com tres entradas por pokemon (contorno do dataset, afinado,
        silhueta limpa) e as saidas correspondentes.

ARMADILHAS APRENDIDAS NO DIA, a aplicar antes de concluir qualquer coisa destes numeros:

  - Saturacao alta NAO e sinal de cor boa. Uma rodada colapsou numa paleta magenta
    unica e marcou a saturacao mais alta de todas. Sempre olhar a imagem junto.
  - Gradiente interno alto NAO e sinal de estrutura. Listra vertical infla o gradiente
    horizontal. Separar |gx| de |gy|: no alvo real a razao e 1,04.
  - A convencao de saturacao usada aqui (media HSV, fundo incluido) nao reproduz uma
    medicao anterior do projeto. Comparar so numeros gerados pelo mesmo script.
  - O split e por imagem, nao por pokemon: 91,7% do teste tem um primo no treino.
    Nao ha folga treino/teste (medido: 0,2333 contra 0,2334), entao o modelo nao
    decora -- mas a metrica tambem nao mede generalizacao para forma inedita.
