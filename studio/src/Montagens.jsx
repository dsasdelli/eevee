import { useEffect, useState } from 'react';

// Salvar e abrir montagens (montagens/ no servidor). Salvar com o mesmo nome
// sobrescreve; abrir entra no historico da tela, entao Ctrl+Z desfaz.
export default function Montagens({ tela, pecas }) {
  const [lista, setLista] = useState([]);
  const [nome, setNome] = useState('');
  const [aberta, setAberta] = useState(false);
  const [aviso, setAviso] = useState(null);

  useEffect(() => {
    fetch('/api/montagens').then(r => r.json()).then(d => setLista(d.montagens)).catch(() => {});
  }, []);

  async function salvar() {
    setAviso(null);
    const cena = tela.current.obterCena();
    if (!cena.itens.length && !cena.tracos.length) {
      setAviso({ tipo: 'erro', texto: 'a tela está vazia' }); return;
    }
    if (lista.some(m => m.nome.toLowerCase() === nome.trim().toLowerCase())
        && !window.confirm(`Já existe “${nome.trim()}”. Substituir?`)) return;
    const r = await fetch('/api/montagens', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nome, cena, miniatura: tela.current.miniatura() }),
    });
    const d = await r.json();
    if (!r.ok) { setAviso({ tipo: 'erro', texto: d.erro }); return; }
    setLista(d.montagens);
    setAviso({ tipo: 'info', texto: `salva: ${nome.trim()}` });
  }

  async function abrir(m) {
    setAviso(null);
    const r = await fetch(`/api/montagens?id=${encodeURIComponent(m.id)}`);
    const d = await r.json();
    if (!r.ok) { setAviso({ tipo: 'erro', texto: d.erro }); return; }
    // Peca apagada da biblioteca depois de salvar: sai da cena em vez de virar um
    // buraco invisivel que ainda ocupa a selecao.
    const existentes = new Set(pecas.map(p => p.id));
    const itens = d.cena.itens.filter(i => existentes.has(i.pecaId));
    tela.current.carregar({ ...d.cena, itens });
    setNome(d.nome);
    setAberta(false);
    const faltam = d.cena.itens.length - itens.length;
    setAviso({ tipo: faltam ? 'erro' : 'info',
               texto: `aberta: ${d.nome}` + (faltam ? ` · ${faltam} peça(s) já não existem na biblioteca` : '') });
  }

  async function apagar(m) {
    if (!window.confirm(`Apagar a montagem “${m.nome}”?`)) return;
    const r = await fetch(`/api/montagens?id=${encodeURIComponent(m.id)}`, { method: 'DELETE' });
    const d = await r.json();
    if (r.ok) setLista(d.montagens); else setAviso({ tipo: 'erro', texto: d.erro });
  }

  return (
    <div className="montagens">
      <div className="ferramentas">
        <input type="text" placeholder="nome da montagem" value={nome} maxLength={80}
               onChange={e => setNome(e.target.value)}
               onKeyDown={e => { if (e.key === 'Enter' && nome.trim()) salvar(); }} />
        <button onClick={salvar} disabled={!nome.trim()}>💾 Salvar</button>
        <button className={aberta ? 'ativo' : ''} onClick={() => setAberta(a => !a)}>
          📂 Abrir <small>{lista.length}</small></button>
      </div>
      {aviso && <div className={'aviso ' + aviso.tipo} style={{ margin: '8px 0 0' }}>{aviso.texto}</div>}
      {aberta && (
        <div className="pecas" style={{ marginTop: 10 }}>
          {!lista.length && <p className="dica" style={{ margin: 0 }}>Nenhuma montagem salva.</p>}
          {lista.map(m => (
            <figure key={m.id} className="peca montagem" onClick={() => abrir(m)}
                    title={`${m.nome} · ${m.pecas} peça(s) · ${new Date(m.salvo * 1000).toLocaleString('pt-BR')}`}>
              {m.miniatura && <img src={m.miniatura} alt={m.nome} />}
              <figcaption>{m.nome}</figcaption>
              <button className="x" title="apagar esta montagem"
                      onClick={e => { e.stopPropagation(); apagar(m); }}>×</button>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
