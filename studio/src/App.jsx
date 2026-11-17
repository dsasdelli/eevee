import { useEffect, useRef, useState } from 'react';
import Tela from './Tela.jsx';
import Parametros, { PADRAO } from './Parametros.jsx';
import Resultados from './Resultados.jsx';
import Recortar from './Recortar.jsx';
import Paleta from './Paleta.jsx';
import Detalhe from './Detalhe.jsx';
import Montagens from './Montagens.jsx';
import { ESCALA_ALTA } from './Tela.jsx';

// O contorno de artwork, que e o dominio de treino, tem ~7% de tinta. Acima do dobro
// disso o esboco ja nao parece o que o modelo viu.
const TINTA_ALTA = 0.15;

const abaDoHash = () => (window.location.hash === '#recortar' ? 'recortar' : 'desenhar');

export default function App() {
  const tela = useRef(null);
  const [aba, setAba] = useState(abaDoHash);
  const [pecas, setPecas] = useState([]);
  const [categorias, setCategorias] = useState([]);

  useEffect(() => {
    const f = () => setAba(abaDoHash());
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  useEffect(() => {
    fetch('/api/pecas').then(r => r.json()).then(d => {
      setPecas(d.pecas); setCategorias(d.categorias);
    }).catch(() => {});
  }, []);
  const [estado, setEstado] = useState('conectando…');
  const [modelos, setModelos] = useState([]);
  const [p, setP] = useState(PADRAO);
  const [ferramenta, setFerramenta] = useState('lapis');
  const [espessura, setEspessura] = useState(6);
  const [gerando, setGerando] = useState(false);
  const [saidas, setSaidas] = useState([]);
  const [aviso, setAviso] = useState(null);
  const [sel, setSel] = useState(null);       // a peca selecionada na tela
  const [detalhe, setDetalhe] = useState(null); // o Pokemon aberto para aprimorar

  useEffect(() => {
    fetch('/api/estado').then(r => r.json()).then(d => {
      setModelos(d.modelos);
      // o primeiro disponivel, como o <select> da pagina antiga faz sozinho
      const m = d.modelos.find(x => x.disponivel);
      setP(q => ({ ...q, modelo: m ? m.chave : '' }));
      setEstado(<><b>pronto</b> · {d.dispositivo}</>);
    }).catch(() => setEstado('servidor fora do ar'));
  }, []);

  async function gerar() {
    setGerando(true);
    setAviso(null);
    try {
      const { desenho, tinta, escalaMax } = tela.current.exportar();
      const alertas = [];
      if (tinta > TINTA_ALTA) alertas.push(`muita tinta (${(tinta * 100).toFixed(0)}%, o normal é ~7%)`);
      if (escalaMax > ESCALA_ALTA) alertas.push(`peça ampliada ${(escalaMax * 100).toFixed(0)}%: o traço engrossa junto`);
      const corpo = {
        desenho,
        modelo: p.modelo, forca: p.forca, amostras: p.amostras,
        hires: p.hires ? 0.35 : null,
        semente: p.semente === '' ? null : +p.semente,
        pixelizar: p.pixelizar, fundo_branco: p.fundo_branco,
        lado: p.lado, cores: p.cores, passos: p.passos,
      };
      const r = await fetch('/api/gerar', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(corpo),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.erro || 'falhou');
      setSaidas(d.saidas);
      setAviso({ tipo: 'info', texto: `${d.saidas.length} em ${(d.ms / 1000).toFixed(1)}s · tinta ${(tinta * 100).toFixed(1)}%`
        + (alertas.length ? ` · atenção: ${alertas.join('; ')}` : '') });
    } catch (e) {
      setAviso({ tipo: 'erro', texto: e.message });
    } finally {
      setGerando(false);
    }
  }

  return (
    <>
      <header className={detalhe ? 'ouro' : ''}>
        {detalhe && <button onClick={() => setDetalhe(null)}>← Voltar ao estúdio</button>}
        <h1>{detalhe ? 'Pokémon' : 'Pokemon Studio'}</h1>
        {detalhe ? <span className="sub">aprimorar e visualizar em 3D</span> : <nav className="abas">
          <a href="#" className={aba === 'desenhar' ? 'ativo' : ''}>Desenhar</a>
          <a href="#recortar" className={aba === 'recortar' ? 'ativo' : ''}>Recortar peças</a>
        </nav>}
        <span id="estado">{estado}</span>
      </header>
      {/* A aba de desenho fica sempre montada, so escondida: desmontar apagaria o
          desenho. A de recorte nao guarda nada que nao esteja no servidor. */}
      {/* key: abrir outro Pokemon recomeca a arvore do zero */}
      {detalhe && <Detalhe key={detalhe.id} origem={detalhe} modelos={modelos} p={p}
                           esboco={() => tela.current.exportar().desenho}
                           aoVoltar={() => setDetalhe(null)}
                           aoExpirar={msg => { setDetalhe(null); setAviso({ tipo: 'erro', texto: msg }); }} />}
      {!detalhe && aba === 'recortar' &&
        <Recortar pecas={pecas} categorias={categorias} setPecas={setPecas} />}
      <main className="estudio" style={!detalhe && aba === 'desenhar' ? undefined : { display: 'none' }}>
        <section className="cartao">
          <h2>Desenho</h2>
          <Tela ref={tela} ferramenta={ferramenta} espessura={espessura} pecas={pecas}
                aoMudarSelecao={setSel} aoSoltar={() => setFerramenta('mover')} />
          <div className="ferramentas">
            <button className={ferramenta === 'lapis' ? 'ativo' : ''}
                    onClick={() => setFerramenta('lapis')}>✏️ Lápis</button>
            <button className={ferramenta === 'borracha' ? 'ativo' : ''}
                    onClick={() => setFerramenta('borracha')}
                    title="apaga só o lápis, não as peças">Borracha</button>
            <button className={ferramenta === 'mover' ? 'ativo' : ''}
                    onClick={() => setFerramenta('mover')}>✋ Mover peças</button>
            <button onClick={() => tela.current.desfazer()}>↶ Desfazer</button>
            <button onClick={() => tela.current.limpar()}>Limpar</button>
            <label className="campo" style={{ flex: 1, minWidth: 130, margin: 0 }}>
              <span>Espessura <i>{espessura}</i></span>
              <input type="range" min="1" max="24" value={espessura}
                     onChange={e => setEspessura(+e.target.value)} />
            </label>
          </div>
          {sel && (
            <div className="ferramentas selecao">
              <small>
                {Math.round(Math.abs(sel.sx) * 100)}% · {Math.round(sel.rotacao)}°
                {Math.abs(sel.sx) > ESCALA_ALTA && <b title="o traço da peça engrossa junto"> · traço grosso</b>}
              </small>
              <button onClick={() => tela.current.espelhar()} title="H">⇋ Espelhar</button>
              <button onClick={() => tela.current.par()} title="cópia espelhada do outro lado">Par</button>
              <button onClick={() => tela.current.duplicar()} title="Ctrl+D">Duplicar</button>
              <button onClick={() => tela.current.frente()}>Frente</button>
              <button onClick={() => tela.current.tras()}>Trás</button>
              <button onClick={() => tela.current.apagar()} title="Delete">Apagar</button>
            </div>
          )}
          <h2 style={{ marginTop: 16 }}>Montagem</h2>
          <Montagens tela={tela} pecas={pecas} />
          <h2 style={{ marginTop: 16 }}>Peças</h2>
          <Paleta pecas={pecas} categorias={categorias}
                  aoEscolher={pc => { tela.current.adicionar(pc); setFerramenta('mover'); }} />
          <p className="dica" style={{ margin: '10px 0 0' }}>
            Em <b>Mover peças</b>: arraste para posicionar, use os cantos para girar e
            redimensionar, setas para ajuste fino (Shift = 10 px). O lápis desenha por
            cima das peças.</p>
        </section>
        <Parametros p={p} mudar={setP} modelos={modelos} gerando={gerando} aoGerar={gerar} />
        <Resultados saidas={saidas} aviso={aviso} aoAbrir={setDetalhe} />
      </main>
    </>
  );
}
