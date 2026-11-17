import { useEffect, useMemo, useRef, useState } from 'react';
import { Image as KImage, Layer, Line, Rect, Stage, Circle } from 'react-konva';
import usarImagem from './usarImagem.js';

const LADO = 512;
const urlImagem = (tipo, p) =>
  `/api/imagem?tipo=${tipo}&geracao=${p.geracao}&nome=${encodeURIComponent(p.nome)}`;

// ------------------------------------------------------------------ lista de pokemons
function Busca({ pokemons, atual, escolher }) {
  const [q, setQ] = useState('');
  const achados = useMemo(() => {
    const t = q.trim().toLowerCase();
    const l = t ? pokemons.filter(p => p.nome.includes(t)) : pokemons;
    return l.slice(0, 80);
  }, [q, pokemons]);
  return (
    <section className="cartao">
      <h2>Pokémon de origem</h2>
      <label className="campo"><span>Buscar <i>{pokemons.length} contornos</i></span>
        <input type="search" placeholder="ex.: eevee, pikachu…" value={q}
               onChange={e => setQ(e.target.value)} /></label>
      <div className="lista">
        {achados.map(p => (
          <button key={p.geracao + p.nome}
                  className={atual && atual.nome === p.nome && atual.geracao === p.geracao ? 'ativo' : ''}
                  onClick={() => escolher(p)}>
            {p.nome} <small>{p.geracao}</small>
          </button>
        ))}
        {!achados.length && <p className="vazio">nada com “{q}”</p>}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ tela do laco
function Laco({ pokemon, pontos, setPontos, comCor, modo, cliques, aoClicar }) {
  const caixa = useRef(null), camada = useRef(null);
  const [largura, setLargura] = useState(LADO);
  const [zoom, setZoom] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [espaco, setEspaco] = useState(false);
  const arrasto = useRef(null);       // {inicio, livre} enquanto o botao esta apertado
  const contorno = usarImagem(pokemon && urlImagem('contorno', pokemon));
  const arte = usarImagem(pokemon && urlImagem('arte', pokemon));

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setLargura(Math.floor(e.contentRect.width)));
    ro.observe(caixa.current);
    return () => ro.disconnect();
  }, []);
  useEffect(() => { setZoom(1); setPos({ x: 0, y: 0 }); }, [pokemon]);

  // Espaco segurado = arrastar a vista (como no Photoshop), sem sair do laco.
  useEffect(() => {
    const baixo = e => { if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); setEspaco(true); } };
    const cima = e => { if (e.code === 'Space') setEspaco(false); };
    window.addEventListener('keydown', baixo); window.addEventListener('keyup', cima);
    return () => { window.removeEventListener('keydown', baixo); window.removeEventListener('keyup', cima); };
  }, []);

  const base = largura / LADO, escala = base * zoom;

  const roda = e => {
    e.evt.preventDefault();
    const st = e.target.getStage(), ptr = st.getPointerPosition();
    const novo = Math.min(8, Math.max(1, zoom * (e.evt.deltaY < 0 ? 1.2 : 1 / 1.2)));
    // mantem o ponto sob o cursor parado enquanto amplia
    const img = { x: (ptr.x - pos.x) / escala, y: (ptr.y - pos.y) / escala };
    const e2 = base * novo;
    setZoom(novo);
    setPos(novo === 1 ? { x: 0, y: 0 } : { x: ptr.x - img.x * e2, y: ptr.y - img.y * e2 });
  };

  const ponto = () => {
    const p = camada.current.getRelativePointerPosition();
    return [Math.round(Math.min(LADO, Math.max(0, p.x)) * 10) / 10,
            Math.round(Math.min(LADO, Math.max(0, p.y)) * 10) / 10];
  };
  // Clique solto poe UM ponto; apertar e arrastar desenha o laco a mao livre.
  const baixo = e => {
    if (espaco) return;
    // Modo clique: esquerdo marca DENTRO da peca, direito (ou Shift) marca FORA.
    if (modo === 'sam') {
      if (e.evt.button === 0 || e.evt.button === 2)
        aoClicar(ponto(), e.evt.button === 2 || e.evt.shiftKey);
      return;
    }
    if (e.evt.button !== 0) return;
    const p = ponto();
    arrasto.current = { ultimo: p };
    setPontos(l => [...l, ...p]);
  };
  const move = () => {
    const a = arrasto.current;
    if (!a) return;
    const p = ponto();
    if (Math.hypot(p[0] - a.ultimo[0], p[1] - a.ultimo[1]) * escala < 4) return;
    a.ultimo = p;
    setPontos(l => [...l, ...p]);
  };
  const solta = () => { arrasto.current = null; };

  return (
    <div>
      <div ref={caixa} className="tela" style={{ cursor: espaco ? 'grab' : 'crosshair' }}
           onContextMenu={e => e.preventDefault()}>
        <Stage width={largura} height={largura} scaleX={escala} scaleY={escala}
               x={pos.x} y={pos.y} draggable={espaco}
               onDragEnd={e => setPos({ x: e.target.x(), y: e.target.y() })}
               onWheel={roda} onPointerDown={baixo} onPointerMove={move}
               onPointerUp={solta} onPointerLeave={solta}>
          <Layer ref={camada}>
            <Rect width={LADO} height={LADO} fill="#fff" />
            {comCor && arte && <KImage image={arte} width={LADO} height={LADO} opacity={0.4} />}
            {contorno && <KImage image={contorno} width={LADO} height={LADO}
                                 globalCompositeOperation={comCor ? 'multiply' : 'source-over'} />}
            {pontos.length >= 4 && (
              <Line points={pontos} closed={pontos.length >= 6} stroke="#2a78d6"
                    strokeWidth={1.5 / escala} fill="rgba(42,120,214,0.18)" lineJoin="round" />
            )}
            {modo === 'laco' && pontos.length >= 2 && (
              <Circle x={pontos[0]} y={pontos[1]} radius={4 / escala} fill="#2a78d6" />
            )}
            {modo === 'sam' && cliques.map((c, i) => (
              <Circle key={i} x={c.x} y={c.y} radius={5 / escala} stroke="#fff" strokeWidth={1.5 / escala}
                      fill={c.fora ? '#d23b2b' : '#1baf7a'} />
            ))}
          </Layer>
        </Stage>
      </div>
      <p className="dica" style={{ marginTop: 8 }}>
        {modo === 'sam'
          ? 'Clique DENTRO da peça (verde). Se pegar demais, clique com o botão direito no que deve ficar FORA (vermelho). '
          : 'Clique para pôr pontos ou arraste para contornar à mão. '}
        Roda do mouse = zoom
        {zoom > 1 && ` (${zoom.toFixed(1)}×)`}, espaço + arrastar = mover a vista,
        Backspace = desfazer ponto.
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ galeria
function Galeria({ pecas, categorias, apagar }) {
  const porCat = categorias.map(c => [c, pecas.filter(p => p.categoria === c)]).filter(([, l]) => l.length);
  return (
    <section className="cartao">
      <h2>Peças salvas <span style={{ textTransform: 'none' }}>· {pecas.length}</span></h2>
      {!pecas.length && <p className="vazio">Nenhuma peça ainda.</p>}
      {porCat.map(([c, l]) => (
        <div key={c} className="categoria">
          <h3>{c} <small>{l.length}</small></h3>
          <div className="pecas">
            {l.map(p => (
              <figure key={p.id} className="peca" title={`${p.id} · ${p.largura}×${p.altura} px`}>
                <img src={`/pecas/${p.arquivo}`} alt={p.id} />
                <button className="x" title="apagar esta peça" onClick={() => apagar(p)}>×</button>
              </figure>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

// ------------------------------------------------------------------ tela inteira
export default function Recortar({ pecas, categorias, setPecas }) {
  const [pokemons, setPokemons] = useState([]);
  const [atual, setAtual] = useState(null);
  const [pontos, setPontos] = useState([]);
  const [categoria, setCategoria] = useState('olho');
  const [comCor, setComCor] = useState(true);
  const [aviso, setAviso] = useState(null);
  const [salvando, setSalvando] = useState(false);
  const [modo, setModo] = useState('laco');      // 'laco' (manual) ou 'sam' (clique)
  const [cliques, setCliques] = useState([]);    // [{x, y, fora}] do modo clique
  const [pensando, setPensando] = useState(false);
  const recomecar = () => { setPontos([]); setCliques([]); };

  useEffect(() => {
    fetch('/api/pokemons').then(r => r.json()).then(d => {
      setPokemons(d.pokemons);
      setAtual(d.pokemons.find(p => p.nome === 'eevee') || d.pokemons[0] || null);
    }).catch(() => setAviso({ tipo: 'erro', texto: 'servidor fora do ar' }));
  }, []);
  useEffect(() => { recomecar(); }, [atual]);

  useEffect(() => {
    const tecla = e => {
      if (e.target !== document.body) return;
      // No modo clique o poligono vem do SAM: tirar um ponto dele so o deformaria.
      if (e.key === 'Backspace' && modo === 'laco') { e.preventDefault(); setPontos(l => l.slice(0, -2)); }
      if (e.key === 'Escape') { setPontos([]); setCliques([]); }
    };
    window.addEventListener('keydown', tecla);
    return () => window.removeEventListener('keydown', tecla);
  }, [modo]);

  // Cada clique manda TODOS os cliques ate agora: o SAM refaz a mascara com o
  // conjunto, e um clique de fora corrige a proposta anterior.
  async function clicar(p, fora) {
    const novos = [...cliques, { x: p[0], y: p[1], fora }];
    setCliques(novos); setPensando(true); setAviso(null);
    try {
      const r = await fetch('/api/segmentar', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...atual,
          positivos: novos.filter(c => !c.fora).map(c => [c.x, c.y]),
          negativos: novos.filter(c => c.fora).map(c => [c.x, c.y]) }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.erro || 'falhou');
      setPontos(d.poligono);
    } catch (e) {
      setAviso({ tipo: 'erro', texto: e.message });
    } finally {
      setPensando(false);
    }
  }

  async function salvar() {
    setSalvando(true); setAviso(null);
    try {
      const r = await fetch('/api/pecas', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...atual, categoria, poligono: pontos }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.erro || 'falhou');
      setPecas(d.pecas);
      recomecar();
      setAviso({ tipo: 'info', texto: `salva: ${d.peca.id} (${d.peca.largura}×${d.peca.altura} px)` });
    } catch (e) {
      setAviso({ tipo: 'erro', texto: e.message });
    } finally {
      setSalvando(false);
    }
  }

  async function apagar(p) {
    if (!window.confirm(`Apagar a peça ${p.id}?`)) return;
    const r = await fetch(`/api/pecas?id=${encodeURIComponent(p.id)}`, { method: 'DELETE' });
    const d = await r.json();
    if (r.ok) setPecas(d.pecas); else setAviso({ tipo: 'erro', texto: d.erro });
  }

  return (
    <main className="estudio">
      <section className="cartao">
        <h2>Recortar{atual && <span style={{ textTransform: 'none' }}> · {atual.nome}</span>}</h2>
        <div className="ferramentas" style={{ margin: '0 0 10px' }}>
          <button className={modo === 'laco' ? 'ativo' : ''}
                  onClick={() => { setModo('laco'); recomecar(); }}>✂️ Laço</button>
          <button className={modo === 'sam' ? 'ativo' : ''}
                  onClick={() => { setModo('sam'); recomecar(); }}>👆 Clique (SAM)</button>
          {pensando && <small className="gerando" style={{ color: 'var(--tinta3)' }}>
            recortando… (o 1º clique carrega o modelo)</small>}
        </div>
        <Laco pokemon={atual} pontos={pontos} setPontos={setPontos} comCor={comCor}
              modo={modo} cliques={cliques} aoClicar={clicar} />
        <div className="ferramentas">
          <label className="campo" style={{ margin: 0 }}>
            <select value={categoria} onChange={e => setCategoria(e.target.value)}>
              {categorias.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          {modo === 'laco' &&
            <button onClick={() => setPontos(l => l.slice(0, -2))} disabled={!pontos.length}>↶ Ponto</button>}
          <button onClick={recomecar} disabled={!pontos.length && !cliques.length}>Recomeçar</button>
          <div className="caixa" style={{ margin: 0 }}>
            <input type="checkbox" id="cor" checked={comCor} onChange={e => setComCor(e.target.checked)} />
            <label htmlFor="cor">arte por baixo</label>
          </div>
        </div>
        <button className={'primario' + (salvando ? ' gerando' : '')}
                disabled={pontos.length < 6 || salvando || pensando || !atual} onClick={salvar}>
          Salvar como {categoria}
        </button>
        {aviso && <div className={'aviso ' + aviso.tipo} style={{ marginTop: 12 }}>{aviso.texto}</div>}
      </section>
      <Busca pokemons={pokemons} atual={atual} escolher={setAtual} />
      <Galeria pecas={pecas} categorias={categorias} apagar={apagar} />
    </main>
  );
}
