import { useEffect, useRef, useState } from 'react';
import { Cartao, useDpr } from './Resultados.jsx';

const virgula = v => (+v).toFixed(2).replace('.', ',');

// Tela de detalhe: aprimorar, arvore de versoes e 3D.
//
// A arvore guarda, por no, o id da imagem crua no servidor, a miniatura e os filhos.
// Aprimorar a partir de um no antigo cria um GALHO novo em vez de substituir o
// caminho -- por isso arvore, e nao uma pilha de desfazer.
export default function Detalhe({ origem, modelos, p, esboco, aoVoltar, aoExpirar }) {
  const dpr = useDpr();
  const nome = origem.nome || '';
  const [nos, setNos] = useState(() => ({
    [origem.id]: { id: origem.id, src: origem.imagem, semente: origem.semente, filhos: [], rotulo: 'original' },
  }));
  const [baseId, setBaseId] = useState(origem.id);
  const [modelo, setModelo] = useState(() => (modelos.find(m => m.disponivel) || {}).chave || '');
  const [refino, setRefino] = useState(0.4);
  const [manter, setManter] = useState(true);
  const [amostras, setAmostras] = useState(2);
  const [trabalhando, setTrabalhando] = useState(false);
  const [saidas, setSaidas] = useState([]);
  const [aviso, setAviso] = useState(null);
  const [tam, setTam] = useState(null);
  const [esp3d, setEsp3d] = useState(0.22);
  const [glb, setGlb] = useState(null);       // { objeto, url } do modelo 3D atual
  const [construindo, setConstruindo] = useState(false);

  const base = nos[baseId];
  const arquivo = s => `${nome ? nome.toLowerCase() : 'pokemon'}-${s}`;
  useEffect(() => { window.scrollTo({ top: 0, behavior: 'smooth' }); }, []);
  // O visualizador 3D traz o three.js (~1 MB): so carrega quando o detalhe abre.
  useEffect(() => { import('@google/model-viewer'); }, []);
  // trocar de no invalida o 3D, que era do no anterior
  useEffect(() => () => { if (glb) URL.revokeObjectURL(glb.objeto); }, [glb]);
  useEffect(() => { setGlb(null); }, [baseId]);

  async function aprimorar() {
    const pai = baseId;
    setTrabalhando(true); setAviso(null);
    try {
      const r = await fetch('/api/gerar', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          desenho: esboco(),            // serve de controle quando o traco e mantido
          modelo, forca: p.forca, amostras, base_id: pai, refino, manter_traco: manter,
          pixelizar: p.pixelizar, fundo_branco: p.fundo_branco,
          lado: p.lado, cores: p.cores, passos: p.passos,
        }),
      });
      const d = await r.json();
      if (r.status === 410) { aoExpirar(d.erro); return; }
      if (!r.ok) throw new Error(d.erro || 'falhou');
      const m = modelos.find(x => x.chave === modelo);
      const rotulo = (m ? m.rotulo.split(' (')[0] : '?') + ' · ' + virgula(refino);
      for (const s of d.saidas) if (nome) s.nome = nome;     // mesmo bicho, mesmo nome
      setNos(n => {
        const novo = { ...n, [pai]: { ...n[pai], filhos: [...n[pai].filhos, ...d.saidas.map(s => s.id)] } };
        for (const s of d.saidas)
          novo[s.id] = { id: s.id, src: s.imagem, semente: s.semente, filhos: [], rotulo };
        return novo;
      });
      setSaidas(d.saidas);
      setAviso({ tipo: 'info', texto: `${d.saidas.length} em ${(d.ms / 1000).toFixed(1)}s` });
    } catch (e) {
      setAviso({ tipo: 'erro', texto: e.message });
    } finally {
      setTrabalhando(false);
    }
  }

  async function gerar3d() {
    setConstruindo(true);
    try {
      const url = `/api/3d?id=${encodeURIComponent(baseId)}&espessura=${esp3d}`;
      // Confere ANTES de entregar ao visualizador: se o servidor devolver erro em
      // JSON, o model-viewer falharia em silencio e a tela ficaria vazia.
      const r = await fetch(url);
      if (!r.ok) throw new Error((await r.json()).erro || 'falhou');
      const blob = await r.blob();
      setGlb({ objeto: URL.createObjectURL(blob), url });
      setAviso({ tipo: 'info', texto: `modelo 3D pronto (${(blob.size / 1024 / 1024).toFixed(1)} MB)` });
    } catch (e) {
      setAviso({ tipo: 'erro', texto: e.message });
    } finally {
      setConstruindo(false);
    }
  }

  // Percurso em profundidade: o recuo mostra o parentesco.
  const ramos = [];
  (function anda(id, nivel) {
    const no = nos[id];
    if (!no) return;
    ramos.push(
      <div key={id} className="ramo" style={{ paddingLeft: nivel * 22 }}>
        {nivel > 0 && <span className="fio">└</span>}
        <button className={'no' + (id === baseId ? ' atual' : '')} title="voltar a esta etapa"
                onClick={() => setBaseId(id)}>
          <img src={no.src} alt="" /><span>{no.rotulo}</span>
        </button>
      </div>,
    );
    for (const f of no.filhos) anda(f, nivel + 1);
  })(origem.id, 0);

  const atual = modelos.find(m => m.chave === modelo);
  return (
    <main className="estudio ouro-tela">
      <section className="cartao ouro">
        <h2>Partindo desta imagem{nome && ` · ${nome}`}</h2>
        <img className="base-grande" src={base.src} alt="imagem base do aprimoramento"
             onLoad={e => setTam([e.target.naturalWidth, e.target.naturalHeight])}
             style={tam ? { width: tam[0] / dpr, height: tam[1] / dpr } : undefined} />
        <div className="pe">
          <small>semente {base.semente}{tam && ` · ${tam[0]}×${tam[1]} px`}</small>
          <a href={base.src} download={`${arquivo(base.semente)}.png`}>baixar PNG</a>
        </div>
        <p className="dica" style={{ margin: '12px 0 0' }}>
          O aprimoramento parte da versão crua de 512 px guardada no servidor, não do
          Pokémon reduzido mostrado aqui — o detalhe fino é preservado.</p>
      </section>

      <section className="cartao ouro">
        <h2>Como aprimorar</h2>
        <label className="campo"><span>Modelo que vai aprimorar</span>
          <select value={modelo} onChange={e => setModelo(e.target.value)}>
            {modelos.map(m => <option key={m.chave} value={m.chave} disabled={!m.disponivel}>
              {m.rotulo}{m.disponivel ? '' : ' (ausente)'}</option>)}
          </select></label>
        <p className="dica">{atual?.nota}</p>
        <p className="dica" style={{ color: 'var(--ouro)' }}>
          Dica: gere com uma LoRA e aprimore com <b>Sem LoRA</b>. A LoRA dá o estilo; o
          modelo base acrescenta detalhe que ela não tinha.</p>
        <label className="campo"><span>Intensidade <i>{virgula(refino)}</i></span>
          <input type="range" min="0.15" max="0.75" step="0.05" value={refino}
                 onChange={e => setRefino(+e.target.value)} /></label>
        <p className="dica">Abaixo de 0,25 quase nada muda. Em 0,4 o acabamento melhora e a
          criatura continua a mesma. Em 0,6 aparece detalhe novo. Acima disso vira outro bicho.</p>
        <div className="caixa"><input type="checkbox" id="manter" checked={manter}
          onChange={e => setManter(e.target.checked)} /><label htmlFor="manter">Manter o traço original</label></div>
        <p className="dica">Desligado, a criatura pode evoluir para longe do que você desenhou.</p>
        <label className="campo"><span>Variações <i>{amostras}</i></span>
          <input type="range" min="1" max="4" step="1" value={amostras}
                 onChange={e => setAmostras(+e.target.value)} /></label>
        <button className={'primario ouro' + (trabalhando ? ' gerando' : '')}
                disabled={trabalhando || !modelo} onClick={aprimorar}>
          {trabalhando ? 'Aprimorando…' : '✨ Aprimorar'}</button>
      </section>

      <section className="coluna-saidas">
        <div className="cartao ouro" style={{ marginBottom: 20 }}>
          <h2>Evolução</h2>
          <div className="arvore">{ramos}</div>
          <p className="dica" style={{ margin: '10px 0 0' }}>Clique em qualquer etapa para voltar
            a ela. Aprimorar a partir de uma etapa antiga cria um novo galho, sem apagar o que
            já existe.</p>
        </div>

        <div className="cartao ouro" style={{ marginBottom: 20 }}>
          <h2>Modelo 3D</h2>
          {glb
            ? <model-viewer src={glb.objeto} camera-controls="" auto-rotate="" shadow-intensity="0.6"
                            exposure="1.1" environment-image="neutral" alt="modelo 3D do Pokémon" />
            : <p className="vazio" style={{ padding: '18px 0' }}>O volume é inventado inflando a
                silhueta: o centro do corpo fica mais alto, a borda fecha. Não é escultura de
                verdade, é relevo — mas gira, ilumina e exporta como qualquer modelo.</p>}
          <label className="campo" style={{ margin: '12px 0 0' }}>
            <span>Espessura <i>{virgula(esp3d)}</i></span>
            <input type="range" min="0.05" max="0.45" step="0.01" value={esp3d}
                   onChange={e => setEsp3d(+e.target.value)} /></label>
          <div className="ferramentas">
            <button style={{ flex: 1 }} className={construindo ? 'gerando' : ''}
                    disabled={construindo} onClick={gerar3d}>
              {construindo ? 'Construindo…' : 'Gerar modelo 3D'}</button>
            {glb && <a className="baixar" href={glb.url} download={`${arquivo(base.semente)}.glb`}>baixar .glb</a>}
          </div>
        </div>

        {aviso && <div className={'aviso ' + aviso.tipo}>{aviso.texto}</div>}
        <div className="grade">
          {saidas.length
            ? saidas.map(s => <Cartao key={s.id} s={s} dpr={dpr} aoAbrir={() => setBaseId(s.id)} />)
            : <p className="vazio">Ajuste ao lado e clique em Aprimorar.</p>}
        </div>
      </section>
    </main>
  );
}
