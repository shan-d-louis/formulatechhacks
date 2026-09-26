# SIDEWALL frontend (React + TypeScript)

This frontend refactors the previous pure-HTML pages to React + TypeScript while preserving the same routes, DOM IDs, and runtime behavior.

## Pages

- `/` → `index.html`
- `/pitwall` → `pitwall.html`
- `/driver` → `driver.html`
- `/crew` → `crew.html`
- `/atlas` → `atlas.html`

## Build

```bash
cd frontend
npm install
npm run build
```

The Vite build writes compiled multi-page outputs directly into `/web` so the FastAPI server can keep serving `/static/*` from the same location.
