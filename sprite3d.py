"""Transforma um sprite 2D num modelo 3D texturizado, exportavel em glTF/GLB.

A TECNICA: INFLAR A SILHUETA

Nao ha informacao de profundidade num sprite chapado, entao ela precisa ser inventada
de forma plausivel. A transformada de distancia da mascara resolve isso bem: para cada
pixel do bicho, a distancia ate a borda mais proxima. O centro do corpo fica longe da
borda e vira a parte mais alta; a silhueta tem distancia zero e fecha o volume. E o
mesmo principio dos algoritmos de "inflacao" usados para dar volume a desenho 2D.

A curva importa. Distancia crua daria um perfil CONICO, com bico no centro. Elevar a
uma potencia < 1 (`CURVA`) arredonda o perfil e o resultado parece inflado, que e o
que se quer de um bicho.

O modelo e fechado por construcao: a altura e zerada no anel da silhueta, entao a
superficie frontal (+z) e a traseira (-z) se encontram ali. Costurar as laterais a mao
seria o caminho obvio e produzia triangulos degenerados; zerar a borda e mais simples
e nao deixa buraco.

A textura e o proprio sprite, nas duas faces. O verso fica espelhado, que e o
comportamento certo para um recorte de papel grosso -- nao se inventa o que esta atras.

    python sprite3d.py entrada.png saida.glb
"""
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

# Resolucao da malha. 128 da ~16 mil vertices por face, suave o bastante na tela e
# leve o bastante para o navegador abrir na hora. Subir para 256 quadruplica o arquivo
# sem diferenca visivel num bicho deste tamanho.
GRADE = 128
# Espessura maxima, como fracao da largura. 0,22 da um relevo de boneco; acima disso
# o bicho vira uma bola e a silhueta perde a leitura.
ESPESSURA = 0.22
# < 1 arredonda o perfil; 1,0 daria um cone com bico no centro.
CURVA = 0.55
# Espessura minima do miolo, como fracao de ESPESSURA. Ver mascara_e_altura.
PISO = 0.18
# Pixels quase brancos sao fundo. O mesmo limiar que `tirar_fundo` usa.
LIMIAR_FUNDO = 0.85


def mascara_e_altura(im, grade=GRADE, espessura=ESPESSURA, curva=CURVA):
    """Devolve (mascara booleana, altura normalizada) na resolucao da malha."""
    pequena = im.convert('RGB').resize((grade, grade), Image.BOX)
    a = np.asarray(pequena, np.float32) / 255
    mascara = a.min(axis=2) < LIMIAR_FUNDO

    # Buracos internos (um olho branco, por exemplo) nao sao fundo: sem preencher,
    # a transformada de distancia abriria um vale no meio do corpo.
    cheia = ndimage.binary_fill_holes(mascara)
    dist = ndimage.distance_transform_edt(cheia)
    # Anel externo com altura ZERO: e ali que frente e verso se encontram e o volume
    # fecha, sem costurar lateral a mao. O limiar e 1,5 e nao 1,0 porque a superficie
    # so cobre quadrados com os QUATRO cantos dentro da mascara -- um pixel a
    # distancia raiz(2) ~ 1,41 ainda pode ter vizinho DIAGONAL fora e cair na borda
    # da malha. Com 1,0 esses ficavam de fora da solda e abriam o modelo.
    anel = cheia & (dist <= 1.5)
    d = np.maximum(dist - 1.5, 0.0)
    if d.max() <= 0:
        return cheia, np.zeros_like(d, np.float32)
    altura = (d / d.max()) ** curva * espessura
    # PISO de espessura no miolo. Sem ele, partes finas -- a ponta de um espinho, o
    # dedo de uma pata -- ficam inteiras dentro do anel, com altura zero dos dois
    # lados: viram folha de espessura nula, com frente e verso coincidentes, e o
    # modelo deixa de ser estanque. Com o piso elas ficam finas, mas solidas.
    altura = np.where(cheia & ~anel, np.maximum(altura, PISO * espessura), 0.0)
    return cheia, altura.astype(np.float32)


def construir(im, grade=GRADE, espessura=ESPESSURA, curva=CURVA):
    """Sprite PIL -> trimesh.Trimesh texturizado, centrado na origem."""
    mascara, altura = mascara_e_altura(im, grade, espessura, curva)
    if not mascara.any():
        raise ValueError('o sprite esta vazio, nao ha o que modelar')

    # Indice de vertice por celula da grade, -1 onde nao ha corpo.
    idx = np.full(mascara.shape, -1, np.int64)
    ys, xs = np.nonzero(mascara)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)

    # Coordenadas em [-0,5; 0,5], com y invertido (imagem cresce para baixo, 3D para cima).
    px = xs / (grade - 1) - 0.5
    py = 0.5 - ys / (grade - 1)
    pz = altura[ys, xs]

    # No anel da silhueta a altura e zero, entao frente e verso ocupam o MESMO ponto.
    # Dar a eles indices distintos deixaria o modelo com duas superficies apenas
    # encostadas -- visualmente fechado, mas nao estanque, e um editor 3D ou uma
    # impressora reclamariam. Aqui a borda e soldada: um vertice so, compartilhado.
    borda = pz <= 0
    idx_verso = np.where(borda, np.arange(n), -1)
    moveis = np.nonzero(~borda)[0]
    idx_verso[moveis] = n + np.arange(len(moveis))

    vertices = np.concatenate([
        np.stack([px, py, pz], axis=1),                               # frente + borda
        np.stack([px[moveis], py[moveis], -pz[moveis]], axis=1),      # verso, sem a borda
    ])
    # UV: a textura e o sprite inteiro, entao a coordenada sai direto do pixel.
    uv = np.stack([xs / (grade - 1), 1 - ys / (grade - 1)], axis=1)
    uvs = np.concatenate([uv, uv[moveis]])

    faces = []
    # Dois triangulos por quadrado da grade, so onde as quatro celulas tem corpo.
    a = idx[:-1, :-1]
    b = idx[:-1, 1:]
    c = idx[1:, 1:]
    d = idx[1:, :-1]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    # A ordem e a-c-b, e nao a-b-c: como py DECRESCE com a linha da imagem, a ordem
    # direta produziria normal apontando para -z e o modelo inteiro sairia do avesso
    # (volume negativo, iluminacao errada no visualizador).
    faces.append(np.stack([a, c, b], axis=1))
    faces.append(np.stack([a, d, c], axis=1))
    # Verso com a ordem oposta a da frente. Passa pelo mapa idx_verso, que devolve o
    # proprio vertice da frente quando a celula e do anel soldado.
    av, bv, cv, dv = idx_verso[a], idx_verso[b], idx_verso[c], idx_verso[d]
    # Onde os TRES vertices de um triangulo estao no anel soldado, o verso cai nos
    # mesmos indices da frente e o triangulo sairia repetido -- malha nao-manifold.
    # O teste tem de ser POR TRIANGULO: um quadrado com tres cantos soldados e um
    # livre ainda repete o primeiro triangulo, e testar o quadrado inteiro deixava
    # passar justamente esses casos. Partes finas viram folha de uma camada so, e por
    # isso o material e doubleSided.
    t1 = ~((av == a) & (bv == b) & (cv == c))
    t2 = ~((av == a) & (cv == c) & (dv == d))
    faces.append(np.stack([av[t1], bv[t1], cv[t1]], axis=1))
    faces.append(np.stack([av[t2], cv[t2], dv[t2]], axis=1))

    faces = np.concatenate(faces).astype(np.int64)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=im.convert('RGB'),
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
    )
    malha = trimesh.Trimesh(
        vertices=vertices, faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False)
    malha.apply_translation(-malha.bounds.mean(axis=0))
    return malha


def previa(im, grade=GRADE, **kw):
    """Render rapido com sombreamento Lambert, so para conferir o relevo.

    Existe porque abrir um GLB para saber se o volume ficou bom e lento; aqui as
    normais saem direto do mapa de altura e um render chega em milissegundos.
    """
    mascara, altura = mascara_e_altura(im, grade, **kw)
    gy, gx = np.gradient(altura * grade)
    normais = np.stack([-gx, gy, np.ones_like(altura)], axis=-1)
    normais /= np.linalg.norm(normais, axis=-1, keepdims=True)
    luz = np.array([-0.4, 0.5, 0.75])
    luz /= np.linalg.norm(luz)
    lamb = np.clip((normais * luz).sum(axis=-1), 0, 1)

    cor = np.asarray(im.convert('RGB').resize((grade, grade), Image.BOX), np.float32) / 255
    sombreado = cor * (0.35 + 0.65 * lamb[..., None])
    sombreado[~mascara] = 1.0
    return Image.fromarray((np.clip(sombreado, 0, 1) * 255).astype(np.uint8))


def main():
    import sys
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    im = Image.open(sys.argv[1])
    malha = construir(im)
    malha.export(sys.argv[2])
    print(f'{len(malha.vertices)} vertices, {len(malha.faces)} faces -> {sys.argv[2]}')


if __name__ == '__main__':
    main()
