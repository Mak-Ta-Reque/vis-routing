import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api and /examples to the FastAPI backend on the server side, so the
// browser only ever talks to the Vite dev server's own origin/port. This
// means only ONE port needs to be forwarded over SSH (whatever port this dev
// server runs on) -- the browser never needs to reach the backend's port
// directly.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/examples': 'http://localhost:8000',
    },
  },
})
