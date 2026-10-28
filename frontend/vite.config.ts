/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In development the API runs on :8000; the dev server forwards /api (HTTP and WebSocket) to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true,
        // When the backend is not running, answer 502 with a clear message (like nginx does in Docker) instead of a bare 500.
        configure: proxy => proxy.on('error', (_error, _request, response) => {
          if ('writeHead' in response && !response.headersSent) {
            response.writeHead(502, { 'Content-Type': 'application/json' })
            response.end(JSON.stringify({ detail: 'Cannot reach the backend. Check that the backend window is still open (start-app.cmd), then try again.' }))
          }
        }),
      },
    },
  },
  build: {
    rollupOptions: { output: { manualChunks: { charts: ['recharts'], vendor: ['react', 'react-dom', 'react-router-dom', '@tanstack/react-query'] } } },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
