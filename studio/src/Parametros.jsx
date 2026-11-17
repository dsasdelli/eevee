// Padroes dos controles de geracao. Sao os mesmos desde a primeira versao do app:
// mudar um deles muda o que a mesma semente produz.
export const PADRAO = {
  modelo: '', forca: 0.7, amostras: 2, pixelizar: true, fundo_branco: true,
  hires: false, lado: 256, cores: 64, passos: 28, semente: '',
};

const virgula = v => (+v).toFixed(2).replace('.', ',');

function Faixa({ rotulo, valor, mostra = v => v, ...resto }) {
  return (
    <label className="campo">
      <span>{rotulo} <i>{mostra(valor)}</i></span>
      <input type="range" value={valor} {...resto} />
    </label>
  );
}

export default function Parametros({ p, mudar, modelos, gerando, aoGerar }) {
  const muda = chave => e => mudar({
    ...p,
    [chave]: e.target.type === 'checkbox' ? e.target.checked
           : e.target.type === 'range' ? +e.target.value : e.target.value,
  });
  const atual = modelos.find(m => m.chave === p.modelo);

  return (
    <section className="cartao">
      <h2>Parâmetros</h2>
      <label className="campo"><span>Modelo</span>
        <select value={p.modelo} onChange={muda('modelo')}>
          {modelos.map(m => (
            <option key={m.chave} value={m.chave} disabled={!m.disponivel}>
              {m.rotulo}{m.disponivel ? '' : ' (ausente)'}
            </option>
          ))}
        </select>
      </label>
      <p className="dica">{atual?.nota}</p>

      <Faixa rotulo="Fidelidade ao traço" valor={p.forca} mostra={virgula}
             min="0.1" max="1" step="0.05" onChange={muda('forca')} />
      <p className="dica">Alto segue o desenho de perto. Baixe para 0,4–0,5 se o desenho
        for abstrato e o modelo não estiver achando um bicho nele.</p>
      <Faixa rotulo="Variações" valor={p.amostras} min="1" max="4" step="1"
             onChange={muda('amostras')} />

      <div className="caixa"><input type="checkbox" id="pix" checked={p.pixelizar}
        onChange={muda('pixelizar')} /><label htmlFor="pix">Reduzir (pixel art)</label></div>
      <div className="caixa"><input type="checkbox" id="fundo" checked={p.fundo_branco}
        onChange={muda('fundo_branco')} /><label htmlFor="fundo">Fundo branco</label></div>
      <div className="caixa"><input type="checkbox" id="hires" checked={p.hires}
        onChange={muda('hires')} /><label htmlFor="hires">Refino de detalhe (2× mais lento)</label></div>

      <details>
        <summary>Avançado</summary>
        <div style={{ paddingTop: 10 }}>
          <Faixa rotulo="Tamanho do Pokémon (px)" valor={p.lado} min="64" max="512" step="32"
                 onChange={muda('lado')} />
          <Faixa rotulo="Cores" valor={p.cores} min="8" max="256" step="8" onChange={muda('cores')} />
          <Faixa rotulo="Passos de difusão" valor={p.passos} min="10" max="50" step="1"
                 onChange={muda('passos')} />
          <label className="campo"><span>Semente <i>vazio = aleatória</i></span>
            <input type="number" placeholder="aleatória" value={p.semente}
                   onChange={muda('semente')} /></label>
        </div>
      </details>

      <button className={'primario' + (gerando ? ' gerando' : '')} disabled={gerando || !p.modelo}
              onClick={aoGerar}>{gerando ? 'Gerando…' : 'Gerar'}</button>
    </section>
  );
}
