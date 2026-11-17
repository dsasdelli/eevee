import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// O build (dist/) e servido pelo app.py na raiz. Em desenvolvimento (npm run dev,
// porta 5173) o Vite repassa /api e /pecas ao app.py, que precisa estar rodando em
// 7860: os modelos e a biblioteca de pecas so existem do lado do Python.
export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    port: 5173,
    proxy: { '/api': 'http://localhost:7860', '/pecas': 'http://localhost:7860' },
  },
});
