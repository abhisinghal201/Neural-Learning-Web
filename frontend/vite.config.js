// frontend/vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // If you need to proxy the API to avoid CORS:
  // server: {
  //   proxy: { '/api': 'http://localhost:4000' }
  // }
});
