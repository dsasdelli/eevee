# Estúdio de Sprites

Aplicação local: desenhe no navegador, receba o Pokémon.

```
python app.py
```

Abra **http://localhost:7860**. A primeira subida leva uns 15 segundos carregando o
Stable Diffusion; depois cada desenho sai em 5 a 12 segundos.

Para parar: `Ctrl+C` no terminal.

## Arquivos

| arquivo | o que é |
|---|---|
| `app.py` | servidor, carrega os modelos uma vez e serve a API e a interface |
| `studio/` | a interface, em React + react-konva; `studio/dist` é o que o `app.py` serve |
| `pecas.py` | biblioteca de peças: recorte, índice, apagar |
| `segmentar.py` | recorte por clique com o SAM |
| `montagens.py` | montagens salvas |
| `pecas/`, `montagens/` | os dados: PNGs das peças + `indice.json`, e um JSON por montagem |

O servidor usa o `http.server` da biblioteca padrão. A interface precisa ser compilada
uma vez (e de novo depois de mudar algo em `studio/src`), com o Node instalado:

```
cd studio
npm install
npm run build
```

Para mexer na interface com recarga automática: `npm run dev` dentro de `studio/`,
com o `app.py` rodando ao lado, e abra http://localhost:5173.

## Desenhando

Traço preto em fundo branco, que é o que o `fit_sketch` espera. A borracha apaga só o
lápis, nunca as peças. `Ctrl+Z` desfaz tudo, inclusive mover e apagar peças.

## Peças

Além do lápis, dá para montar o Pokémon com **peças** (olhos, orelhas, bocas,
caudas...) recortadas de Pokémon existentes.

**Recortar peças** (aba no topo): escolha um Pokémon, recorte a peça e salve numa
categoria. Dois modos:

- **Laço**: clique ponto a ponto ou arraste à mão em volta da peça.
- **Clique (SAM)**: clique dentro da peça e o modelo SAM acha a borda sozinho. Se pegar
  demais, clique com o botão direito no que deve ficar fora. O primeiro clique depois
  de subir o servidor leva alguns segundos (carrega o modelo); os seguintes, menos de
  um décimo de segundo.

A peça sai do **contorno** do Pokémon, não da arte colorida: o traço é exatamente o que
o modelo viu no treino. A peça guarda só a tinta: o branco é transparente, então pôr
uma peça sobre outra (ou sobre o lápis) soma os traços em vez de esconder o de baixo.

**Montar** (aba Desenhar): clique numa peça da paleta, ou arraste-a para a tela. Em
**Mover peças**, arraste para posicionar e use os cantos para girar e redimensionar.
Com uma peça selecionada: **Espelhar** (H), **Par** (cópia espelhada do outro lado,
para olhos e orelhas), **Duplicar** (Ctrl+D), **Frente**/**Trás**, **Apagar** (Delete);
setas fazem ajuste fino (Shift = 10 px). O lápis desenha por cima das peças.

Cuidado ao ampliar muito uma peça: a linha engrossa junto, e acima de ~160% o esboço
começa a sair do traço que o modelo conhece. A tela avisa.

**Montagens**: dê um nome e salve; **Abrir** mostra as salvas com miniatura. Salvar com
o mesmo nome substitui. A tela atual também fica guardada no navegador, então
recarregar a página não perde o desenho.

O desenho é reenquadrado automaticamente antes de gerar: recortado na caixa do
conteúdo, escalado e centralizado. Não precisa se preocupar com onde desenhou na
folha, só com a forma.

## Os quatro modelos

| modelo | o que é |
|---|---|
| **Gen 9 (acabamento moderno)** | LoRA treinada só nos 551 sprites da gen9. Padrão. |
| **Gen 9 (cor mais viva)** | mesma LoRA, treino mais curto: cor mais saturada |
| **Sprite clássico** | LoRA das nove gerações. Pixel art mais dura. |
| **Sem LoRA** | só o Stable Diffusion base |

O "Sem LoRA" usa um **prompt próprio**, diferente do das LoRAs. Não é detalhe: o token
`pkmnsprite` não significa nada para o modelo base, e pedir "pixel art" a ele faz
desenhar uma grade literal por cima. Com o prompt emprestado a comparação seria
injusta e a opção pareceria pior do que é.

## Parâmetros

**Fidelidade ao traço** é o que mais importa. Alto (0,7 a 1,0) segue o desenho de
perto. Se o seu desenho for abstrato — geometria em vez de bicho — baixe para 0,4 ou
0,5: com fidelidade alta o modelo tenta seguir um traço que não é uma criatura e
devolve uma forma sem vida. Medido, não achismo.

**Variações** gera até 4 com sementes diferentes. Cada semente dá um bicho diferente,
e a semente aparece embaixo de cada resultado — anote a que você gostou para
reproduzir depois.

**Reduzir a sprite** faz a redução para 256 px com paleta curta. A redução usa média
de célula (BOX) e a quantização não usa dithering, porque dither espalha pontinhos, o
oposto de arte chapada.

**Refino de detalhe** é o hires fix: uma segunda passada em 768 que acrescenta detalhe
pequeno. Dobra o tempo e o ganho é sutil — ligue quando quiser caprichar num resultado
específico, não por padrão.

## A tela do sprite

**Clique num sprite gerado** para abrir a tela dele. Lá ficam o aprimoramento e o 3D.
O estúdio continua só com desenho, parâmetros e a grade de resultados.

## Aprimorar

Na tela do sprite, o aprimoramento devolve a imagem ao modelo com ruído parcial, e o
modelo reconstrói acrescentando detalhe em vez de começar do zero.

O uso que mais rende: **gere com uma LoRA e aprimore com "Sem LoRA"**. A LoRA dá a
forma e o acabamento de sprite; o modelo base, que é mais detalhado, passa por cima
acrescentando focinho, olho definido, dobra de pano — coisas que o sprite não tinha.
O seletor de modelo vale para o aprimoramento, então é só trocar antes de clicar.

**Intensidade** decide quanto pode mudar. Abaixo de 0,25 quase nada acontece. Em
torno de 0,4 o acabamento melhora e a criatura continua a mesma. Em 0,6 aparece
detalhe novo de verdade. Acima disso vira criatura nova e o bicho original se perde.

**Manter o traço original** re-impõe o desenho durante o aprimoramento. Ligado, a
forma fica presa ao que você desenhou; desligado, a criatura pode evoluir para longe
dele.

O aprimoramento parte da imagem **crua de 512 px**, não do sprite reduzido que aparece
no cartão — por isso o servidor guarda as últimas 24 geraçoes em memória. Se você
deixar a página aberta muito tempo e a base sair do cache, o botão avisa e é só gerar
de novo.

## A árvore de evolução

A tela do sprite mostra uma **árvore** com todas as etapas. Cada aprimoramento vira um
nó filho de onde você partiu, e **clicar em qualquer nó volta para ele** — a imagem
grande, o download e o 3D passam a ser daquela etapa.

Voltar não apaga nada. Se você voltar duas etapas e aprimorar de novo, nasce um
**galho novo** ao lado do antigo, e os dois ficam disponíveis. É assim que dá para
tentar dois caminhos diferentes a partir do mesmo ponto e comparar.

Cada nó mostra de onde veio: o modelo que o produziu e a intensidade usada.

O servidor guarda as 64 imagens cruas mais recentes. Numa árvore maior que isso, os
nós mais antigos saem do cache e deixam de aceitar aprimoramento ou 3D — a miniatura
continua aparecendo, mas o botão avisa.

## Modelo 3D

Na tela do sprite, **Gerar modelo 3D** constrói uma malha texturizada e a mostra num
visualizador que gira. O botão **baixar .glb** salva o arquivo, que abre no Blender,
no Windows 3D Viewer, no Godot, no Unity e em qualquer coisa que leia glTF.

**Como o volume é inventado.** Um sprite chapado não tem profundidade, então ela é
construída inflando a silhueta: para cada ponto do bicho, calcula-se a distância até
a borda mais próxima, e essa distância vira altura. O centro do corpo fica alto, a
borda fecha em zero. É relevo, não escultura — o verso é o espelho da frente, porque
não há como saber o que está atrás. Mas gira, recebe luz e exporta como qualquer
modelo.

**Espessura** controla o quanto infla. Em 0,22 (padrão) parece um boneco de vinil;
baixo demais vira adesivo, alto demais vira bola e a silhueta perde a leitura.

Limitação honesta: partes muito finas — a ponta de um espinho, um dedo — ficam com
espessura mínima e, em alguns pontos, viram folha de uma camada só. O modelo não é
estanque nessas regiões, o que não atrapalha visualizar nem importar, mas atrapalharia
imprimir em 3D sem reparar a malha antes.

O visualizador usa `model-viewer`, que vem junto no build da interface: funciona
offline.

## Problemas conhecidos

**"interface nao compilada"** — falta o `studio/dist`. Rode `npm install` e
`npm run build` dentro de `studio/`.

**"porta 7860 já está em uso"** — há outro servidor rodando. No Windows dois processos
conseguem subir na mesma porta sem erro, e aí os pedidos caem ora num ora noutro; o
servidor agora recusa subir nesse caso em vez de confundir. Encerre o outro antes.

**Resultado ignora o desenho** — fidelidade baixa demais, ou o desenho tem pouca
tinta. Suba a fidelidade ou engrosse o traço.

**Fundo colorido sobrando** — a remoção de fundo apaga regiões conexas que encostam na
borda, e falha quando o modelo desenha um cenário em vez de um fundo chapado. Gere
outra variação; é mais comum no modelo "Sem LoRA".
