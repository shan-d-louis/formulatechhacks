import { resolve } from 'node:path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: '../web',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        pitwall: resolve(__dirname, 'pitwall.html'),
        driver: resolve(__dirname, 'driver.html'),
        crew: resolve(__dirname, 'crew.html'),
        atlas: resolve(__dirname, 'atlas.html'),
      },
    },
  },
});
