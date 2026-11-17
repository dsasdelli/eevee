"""Montagens salvas: a cena da tela de desenho (pecas com posicao, giro e escala, mais
os tracos de lapis) num JSON por montagem, em `montagens/`.

A cena guarda a REFERENCIA de cada peca (o arquivo em pecas/), nao a imagem: abrir a
montagem relê a peca da biblioteca. Uma peca apagada depois some da montagem.
"""
import json
import re
import threading
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).parent
PASTA = RAIZ / 'montagens'
_ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,60}$')
_trava = threading.Lock()


def _slug(nome):
    s = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')[:60]
    return s or 'montagem'


def _arquivo(ident):
    if not _ID.match(ident or ''):
        raise KeyError(ident)
    return PASTA / f'{ident}.json'


def listar():
    """Resumo de cada montagem, a mais recente primeiro (sem a cena, que e grande)."""
    saida = []
    for f in PASTA.glob('*.json') if PASTA.exists() else []:
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        saida.append({'id': f.stem, 'nome': d.get('nome', f.stem), 'salvo': d.get('salvo', 0),
                      'miniatura': d.get('miniatura'), 'pecas': len(d.get('cena', {}).get('itens', []))})
    return sorted(saida, key=lambda m: -m['salvo'])


def ler(ident):
    f = _arquivo(ident)
    if not f.exists():
        raise KeyError(ident)
    return json.loads(f.read_text(encoding='utf-8'))


def salvar(nome, cena, miniatura):
    """Salva pelo nome: o mesmo nome SOBRESCREVE (e o 'salvar' de sempre); nome novo
    cria outra montagem."""
    nome = (nome or '').strip()[:80]
    if not nome:
        raise ValueError('a montagem precisa de um nome')
    if not isinstance(cena, dict) or not isinstance(cena.get('itens'), list) \
            or not isinstance(cena.get('tracos'), list):
        raise ValueError('cena invalida')
    if miniatura is not None and not (isinstance(miniatura, str)
                                      and miniatura.startswith('data:image/png;base64,')):
        raise ValueError('miniatura invalida')
    ident = _slug(nome)
    with _trava:
        PASTA.mkdir(exist_ok=True)
        f = _arquivo(ident)
        tmp = f.with_suffix('.tmp')
        tmp.write_text(json.dumps({'nome': nome, 'salvo': time.time(), 'cena': cena,
                                   'miniatura': miniatura}, ensure_ascii=False), encoding='utf-8')
        tmp.replace(f)
    return ident


def apagar(ident):
    """Apaga UMA montagem pelo id exato."""
    with _trava:
        f = _arquivo(ident)
        if not f.exists():
            raise KeyError(ident)
        f.unlink()
