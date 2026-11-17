import { useEffect, useState } from 'react';

// Mostra a imagem pixel a pixel: um PNG de 256 px ocupa 256 pixels FISICOS. Com a
// escala do Windows em 125%/150% um px de CSS vale mais de um pixel fisico; sem
// dividir pelo devicePixelRatio o navegador amplia e borra a pixel art.
export function useDpr() {
  const [dpr, setDpr] = useState(window.devicePixelRatio || 1);
  useEffect(() => {
    const f = () => setDpr(window.devicePixelRatio || 1);
    window.addEventListener('resize', f);
    return () => window.removeEventListener('resize', f);
  }, []);
  return dpr;
}

// Clicar no sprite abre o detalhe -- sem botao extra poluindo a grade.
export function Cartao({ s, dpr, aoAbrir }) {
  const [tam, setTam] = useState(null);
  const base = s.nome ? s.nome.toLowerCase() : 'pokemon';
  return (
    <div className="saida">
      <img src={s.imagem} alt="Pokémon gerado" title="abrir" style={{
             cursor: aoAbrir ? 'zoom-in' : undefined,
             ...(tam ? { width: tam[0] / dpr, height: tam[1] / dpr } : {}) }}
           onLoad={e => setTam([e.target.naturalWidth, e.target.naturalHeight])}
           onClick={aoAbrir} />
      {s.nome && <div className="nome" title={'inspirado em: ' + (s.inspiracao || []).join(', ')}>
        {s.nome}</div>}
      <div className="pe">
        <small>semente {s.semente}{tam && ` · ${tam[0]}×${tam[1]} px`}</small>
        <a href={s.imagem} download={`${base}-${s.semente}.png`}>baixar</a>
      </div>
    </div>
  );
}

export default function Resultados({ saidas, aviso, aoAbrir }) {
  const dpr = useDpr();
  return (
    <section className="coluna-saidas">
      {aviso && <div className={'aviso ' + aviso.tipo}>{aviso.texto}</div>}
      <div className="grade">
        {saidas.length
          ? saidas.map(s => <Cartao key={s.id} s={s} dpr={dpr} aoAbrir={() => aoAbrir(s)} />)
          : <p className="vazio">Desenhe à esquerda e clique em Gerar.</p>}
      </div>
      {saidas.length > 0 && <p className="dica" style={{ marginTop: 10 }}>Clique num Pokémon para abri-lo.</p>}
    </section>
  );
}
