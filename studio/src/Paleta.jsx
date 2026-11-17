import { useState } from 'react';

// As pecas recortadas, por categoria. Clicar poe a peca no centro da tela; arrastar
// poe onde soltar.
export default function Paleta({ pecas, categorias, aoEscolher }) {
  const comPecas = categorias.filter(c => pecas.some(p => p.categoria === c));
  const [cat, setCat] = useState(null);
  const atual = comPecas.includes(cat) ? cat : comPecas[0];
  if (!pecas.length)
    return <p className="dica" style={{ margin: '12px 0 0' }}>
      Nenhuma peça ainda. Recorte algumas em <a href="#recortar">Recortar peças</a>.</p>;
  return (
    <div className="paleta">
      <div className="chips">
        {comPecas.map(c => (
          <button key={c} className={c === atual ? 'ativo' : ''} onClick={() => setCat(c)}>
            {c} <small>{pecas.filter(p => p.categoria === c).length}</small>
          </button>
        ))}
      </div>
      <div className="pecas">
        {pecas.filter(p => p.categoria === atual).map(p => (
          <figure key={p.id} className="peca" title={`${p.pokemon} · clique ou arraste`}
                  draggable onDragStart={e => e.dataTransfer.setData('text/peca', p.id)}
                  onClick={() => aoEscolher(p)}>
            <img src={`/pecas/${p.arquivo}`} alt={p.id} draggable={false} />
          </figure>
        ))}
      </div>
    </div>
  );
}
