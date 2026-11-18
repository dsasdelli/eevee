import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Image as KImage, Layer, Line, Rect, Stage, Transformer } from 'react-konva';
import usarImagem from './usarImagem.js';

// O modelo recebe 512 x 512. A tela e exibida no tamanho que couber e as
// coordenadas ficam sempre no espaco de 512: o Stage e escalado, nao o desenho.
export const LADO = 512;

// Acima disto o traco da peca fica grosso demais perto do traco do resto: a linha
// escala junto com a peca. Ver a memoria sobre a lacuna de dominio do traco.
export const ESCALA_ALTA = 1.6;

let proximoUid = 1;

// A cena atual tambem fica no navegador: recarregar a pagina nao perde a montagem.
const CHAVE = 'studio.cena';
const VAZIA = { itens: [], tracos: [] };
function cenaGuardada() {
  try {
    const c = JSON.parse(localStorage.getItem(CHAVE));
    if (c && Array.isArray(c.itens) && Array.isArray(c.tracos)) return c;
  } catch { /* armazenamento bloqueado ou corrompido: comeca vazio */ }
  return VAZIA;
}
// uids novos nunca colidem com os de uma cena carregada
const acertarUid = c => { proximoUid = Math.max(proximoUid, 1 + Math.max(0, ...c.itens.map(i => i.uid))); };

// Uma peca na tela. O offset no centro faz girar e espelhar em torno do meio da peca,
// e nao do canto de cima.
function PecaNo({ item, movivel, registrar, aoSelecionar, aoMudar }) {
  const img = usarImagem(item.src);
  return (
    <KImage ref={n => registrar(item.uid, n)} image={img}
            x={item.x} y={item.y} width={item.w} height={item.h}
            offsetX={item.w / 2} offsetY={item.h / 2}
            rotation={item.rotacao} scaleX={item.sx} scaleY={item.sy}
            draggable={movivel}
            onPointerDown={e => { if (movivel) { e.cancelBubble = true; aoSelecionar(item.uid); } }}
            onDragStart={() => aoSelecionar(item.uid)}
            onDragEnd={e => aoMudar(item.uid, { x: e.target.x(), y: e.target.y() })}
            onTransformEnd={e => {
              const n = e.target;
              aoMudar(item.uid, { x: n.x(), y: n.y(), rotacao: n.rotation(), sx: n.scaleX(), sy: n.scaleY() });
            }} />
  );
}

const Tela = forwardRef(function Tela({ ferramenta, espessura, pecas, aoMudarSelecao, aoSoltar }, ref) {
  const caixa = useRef(null), stage = useRef(null);
  const camadaPecas = useRef(null), camadaTracos = useRef(null), tr = useRef(null);
  const nos = useRef({});
  const [largura, setLargura] = useState(LADO);
  // A cena inteira num estado so: desfazer volta pecas E tracos juntos.
  const [cena, setCena] = useState(() => { const c = cenaGuardada(); acertarUid(c); return c; });
  const [sel, setSel] = useState(null);
  const pilha = useRef([]);
  const desenhando = useRef(false);
  const cenaRef = useRef(cena);
  cenaRef.current = cena;

  // Grava meio segundo depois da ultima mudanca, e nao a cada ponto do lapis.
  useEffect(() => {
    const t = setTimeout(() => {
      try { localStorage.setItem(CHAVE, JSON.stringify(cena)); } catch { /* sem espaco: ignora */ }
    }, 500);
    return () => clearTimeout(t);
  }, [cena]);

  useEffect(() => {
    // Escondida em outra aba, a caixa mede 0: guarda a ultima largura em vez de
    // encolher o Stage a nada.
    const ro = new ResizeObserver(([e]) => {
      const w = Math.floor(e.contentRect.width);
      if (w > 0) setLargura(w);
    });
    ro.observe(caixa.current);
    return () => ro.disconnect();
  }, []);

  // ---------------------------------------------------------------- historico
  const mudarCena = f => {
    pilha.current.push(cenaRef.current);
    if (pilha.current.length > 80) pilha.current.shift();
    setCena(f(cenaRef.current));
  };
  const desfazer = () => {
    if (!pilha.current.length) return;
    const ant = pilha.current.pop();
    setCena(ant);
    setSel(s => (ant.itens.some(i => i.uid === s) ? s : null));
  };
  const mudarItem = (uid, campos) =>
    mudarCena(c => ({ ...c, itens: c.itens.map(i => (i.uid === uid ? { ...i, ...campos } : i)) }));

  // ---------------------------------------------------------------- selecao
  useEffect(() => { if (ferramenta !== 'mover') setSel(null); }, [ferramenta]);
  useEffect(() => {
    const n = sel && nos.current[sel];
    tr.current.nodes(n ? [n] : []);
    tr.current.getLayer().batchDraw();
  }, [sel, cena]);
  const itemSel = cena.itens.find(i => i.uid === sel) || null;
  useEffect(() => { aoMudarSelecao?.(itemSel); }, [itemSel]);

  // ---------------------------------------------------------------- acoes de peca
  const acoes = {
    adicionar(peca, onde) {
      const uid = proximoUid++;
      // Escala 1: a peca saiu de um contorno de 512 e a tela tem 512, entao o traco
      // chega com a mesma espessura que tinha no original.
      const p = onde || { x: LADO / 2, y: LADO / 2 };
      mudarCena(c => ({ ...c, itens: [...c.itens, {
        uid, pecaId: peca.id, src: `/pecas/${peca.arquivo}`, w: peca.largura, h: peca.altura,
        x: p.x, y: p.y, rotacao: 0, sx: 1, sy: 1,
      }] }));
      setSel(uid);
    },
    espelhar() { if (itemSel) mudarItem(sel, { sx: -itemSel.sx }); },
    duplicar() {
      if (!itemSel) return;
      const uid = proximoUid++;
      mudarCena(c => ({ ...c, itens: [...c.itens, { ...itemSel, uid, x: itemSel.x + 20, y: itemSel.y + 20 }] }));
      setSel(uid);
    },
    // Duplicar ja espelhado: o atalho para o par de olhos ou de orelhas.
    par() {
      if (!itemSel) return;
      const uid = proximoUid++;
      const x = LADO - itemSel.x;   // reflete em torno do meio da tela
      mudarCena(c => ({ ...c, itens: [...c.itens, {
        ...itemSel, uid, x: Math.abs(x - itemSel.x) < 10 ? itemSel.x + 40 : x,
        sx: -itemSel.sx, rotacao: -itemSel.rotacao,
      }] }));
      setSel(uid);
    },
    frente() {
      if (!itemSel) return;
      mudarCena(c => ({ ...c, itens: [...c.itens.filter(i => i.uid !== sel), itemSel] }));
    },
    tras() {
      if (!itemSel) return;
      mudarCena(c => ({ ...c, itens: [itemSel, ...c.itens.filter(i => i.uid !== sel)] }));
    },
    apagar() {
      if (!itemSel) return;
      mudarCena(c => ({ ...c, itens: c.itens.filter(i => i.uid !== sel) }));
      setSel(null);
    },
    empurrar(dx, dy) { if (itemSel) mudarItem(sel, { x: itemSel.x + dx, y: itemSel.y + dy }); },
  };

  useEffect(() => {
    const tecla = e => {
      if (caixa.current.offsetParent === null) return;   // aba escondida
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      const ctrl = e.ctrlKey || e.metaKey;
      if (ctrl && e.key === 'z') { e.preventDefault(); desfazer(); return; }
      if (!itemSel) return;
      const passo = e.shiftKey ? 10 : 1;
      const mapa = {
        Delete: acoes.apagar, Backspace: acoes.apagar,
        ArrowLeft: () => acoes.empurrar(-passo, 0), ArrowRight: () => acoes.empurrar(passo, 0),
        ArrowUp: () => acoes.empurrar(0, -passo), ArrowDown: () => acoes.empurrar(0, passo),
      };
      if (ctrl && e.key === 'd') { e.preventDefault(); acoes.duplicar(); }
      else if (!ctrl && e.key.toLowerCase() === 'h') acoes.espelhar();
      else if (mapa[e.key]) { e.preventDefault(); mapa[e.key](); }
    };
    document.addEventListener('keydown', tecla);
    return () => document.removeEventListener('keydown', tecla);
  });

  // ---------------------------------------------------------------- exportar
  // Cada camada e rasterizada SOZINHA e composta sobre branco aqui. A borracha usa
  // 'destination-out', e no toDataURL do Stage todas as camadas caem no mesmo
  // canvas: a borracha furaria tambem o fundo e as pecas, e o furo transparente
  // chegaria ao servidor como tinta. Assim ela apaga so o lapis. O Transformer mora
  // numa terceira camada e nunca entra no esboco.
  const rasterizar = () => {
    const escala = LADO / largura;
    const c = document.createElement('canvas');
    c.width = c.height = LADO;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, LADO, LADO);
    for (const camada of [camadaPecas, camadaTracos])
      ctx.drawImage(camada.current.toCanvas({ pixelRatio: escala }), 0, 0, LADO, LADO);
    return c;
  };

  useImperativeHandle(ref, () => ({
    desfazer, ...acoes,
    limpar() {
      if (cenaRef.current.itens.length || cenaRef.current.tracos.length)
        mudarCena(() => VAZIA);
      setSel(null);
    },
    obterCena: () => cenaRef.current,
    // Abrir uma montagem entra no historico: Ctrl+Z volta ao que estava na tela.
    carregar(c) {
      acertarUid(c);
      mudarCena(() => ({ itens: c.itens, tracos: c.tracos }));
      setSel(null);
    },
    miniatura(lado = 128) {
      const c = document.createElement('canvas');
      c.width = c.height = lado;
      c.getContext('2d').drawImage(rasterizar(), 0, 0, lado, lado);
      return c.toDataURL('image/png');
    },
    exportar() {
      const c = rasterizar();
      // Fracao de tinta (escuro < 128, o mesmo limiar dos contornos do dataset). O contorno de
      // artwork tem ~7%; muito acima disso o esboco sai do que o modelo conhece.
      const d = c.getContext('2d').getImageData(0, 0, LADO, LADO).data;
      let tinta = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i] < 128) tinta++;
      return { desenho: c.toDataURL('image/png'), tinta: tinta / (LADO * LADO),
               escalaMax: Math.max(1, ...cenaRef.current.itens.map(i => Math.abs(i.sx))) };
    },
  }));

  // Soltar uma peca arrastada da paleta.
  const soltar = e => {
    e.preventDefault();
    const peca = pecas.find(p => p.id === e.dataTransfer.getData('text/peca'));
    if (!peca) return;
    stage.current.setPointersPositions(e);
    acoes.adicionar(peca, camadaPecas.current.getRelativePointerPosition());
    aoSoltar?.();
  };

  // ---------------------------------------------------------------- lapis
  const movendo = ferramenta === 'mover';
  const ponto = () => camadaTracos.current.getRelativePointerPosition();
  const inicio = e => {
    if (movendo) {   // clique fora de qualquer peca solta a selecao
      if (e.target === e.target.getStage()) setSel(null);
      return;
    }
    e.evt.preventDefault();
    const p = ponto();
    desenhando.current = true;
    mudarCena(c => ({ ...c, tracos: [...c.tracos, {
      apagar: ferramenta === 'borracha',
      largura: espessura * (ferramenta === 'borracha' ? 2.5 : 1),   // como na pagina antiga
      pontos: [p.x, p.y, p.x + 0.01, p.y],                            // um clique vira um ponto
    }] }));
  };
  // Mover o lapis NAO entra no historico: o traco inteiro e um passo so de desfazer.
  const move = () => {
    if (!desenhando.current) return;
    const p = ponto();
    setCena(c => {
      const ult = c.tracos[c.tracos.length - 1];
      return { ...c, tracos: [...c.tracos.slice(0, -1), { ...ult, pontos: [...ult.pontos, p.x, p.y] }] };
    });
  };
  const fim = () => { desenhando.current = false; };

  const escala = largura / LADO;
  return (
    <div ref={caixa} className="tela" style={{ cursor: movendo ? 'default' : 'crosshair' }}
         onDragOver={e => e.preventDefault()} onDrop={soltar}>
      <Stage ref={stage} width={largura} height={largura} scaleX={escala} scaleY={escala}
             onPointerDown={inicio} onPointerMove={move}
             onPointerUp={fim} onPointerLeave={fim} onPointerCancel={fim}>
        <Layer listening={false}>
          <Rect width={LADO} height={LADO} fill="#fff" />
        </Layer>
        <Layer ref={camadaPecas} listening={movendo}>
          {cena.itens.map(i => (
            <PecaNo key={i.uid} item={i} movivel={movendo}
                    registrar={(uid, n) => { if (n) nos.current[uid] = n; else delete nos.current[uid]; }}
                    aoSelecionar={setSel} aoMudar={mudarItem} />
          ))}
        </Layer>
        <Layer ref={camadaTracos} listening={false}>
          {cena.tracos.map((t, i) => (
            <Line key={i} points={t.pontos} stroke="#000" strokeWidth={t.largura}
                  lineCap="round" lineJoin="round"
                  globalCompositeOperation={t.apagar ? 'destination-out' : 'source-over'} />
          ))}
        </Layer>
        <Layer>
          <Transformer ref={tr} rotationSnaps={[0, 90, 180, 270]} rotationSnapTolerance={6}
                       enabledAnchors={['top-left', 'top-right', 'bottom-left', 'bottom-right']}
                       keepRatio flipEnabled={false} anchorSize={9}
                       borderStroke="#2a78d6" anchorStroke="#2a78d6"
                       boundBoxFunc={(velho, novo) => (Math.abs(novo.width) < 8 ? velho : novo)} />
        </Layer>
      </Stage>
    </div>
  );
});

export default Tela;
