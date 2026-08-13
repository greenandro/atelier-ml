import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Le proxy évite toute question de CORS en développement.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3001', changeOrigin: true },
    },
  },
});
