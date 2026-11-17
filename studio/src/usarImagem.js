import { useEffect, useState } from 'react';

// Carrega uma imagem para o Konva, que desenha HTMLImageElement e nao URL.
export default function usarImagem(src) {
  const [img, setImg] = useState(null);
  useEffect(() => {
    if (!src) { setImg(null); return; }
    let vivo = true;
    const i = new window.Image();
    i.onload = () => { if (vivo) setImg(i); };
    i.src = src;
    return () => { vivo = false; };
  }, [src]);
  return img;
}
