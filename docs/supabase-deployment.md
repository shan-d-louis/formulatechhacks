# Supabase Deployment Notes

Supabase is the backing platform for database, auth, storage, realtime, and Deno/TypeScript
Edge Functions. The current SIDEWALL backend is a Python FastAPI app, so it is not
deployed directly as a Supabase Edge Function. Deploy the FastAPI process to a Python
host, then connect it to Supabase with the production environment values below. If a
Supabase Edge Function is useful, use it as a thin facade that calls the FastAPI host.

## Environment Files

Keep real env files out of git:

- `.env.local` is the instant localhost fallback.
- `.env.prod` is for production deploy commands and mirrors host secrets.
- `.env.local.example` and `.env.prod.example` document the required keys.

Use `.env.local` for local startup:

```bash
uv run python -m sidewall.server.app
```

Use `.env.prod` for production-like local smoke tests:

```powershell
$env:SIDEWALL_ENV = "prod"
uv run python -m sidewall.server.app
```

## Required `.env.prod`

```bash
SIDEWALL_ENV=prod
SIDEWALL_HOST=0.0.0.0
PORT=8000
SIDEWALL_MODEL_VERSION=<git-sha-or-release-id>

SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=<publishable-or-anon-key>

BACKEND_BASE_URL=https://<deployed-fastapi-host>
```

`BACKEND_BASE_URL` can be `http://localhost:8000` for local smoke tests only.
Production Supabase functions need a public HTTPS backend URL because `localhost`
inside Supabase is not your laptop.

## Convenient Script

Run the deployment wrapper from macOS, Linux, WSL, or Git Bash on Windows:

```bash
bash scripts/deploy-supabase.sh --project-ref <project-ref>
```

On Windows PowerShell:

```powershell
.\scripts\deploy-supabase.ps1 -ProjectRef <project-ref>
```

For local function smoke tests, start the FastAPI backend first:

```bash
uv run python -m sidewall.server.app
```

Then serve the Supabase function locally:

```bash
bash scripts/deploy-supabase.sh --env-file .env.local --serve-local --skip-link
```

PowerShell equivalent:

```powershell
.\scripts\deploy-supabase.ps1 -EnvFile .env.local -ServeLocal -SkipLink
```

The script:

- loads `.env.prod` by default;
- refuses production deploys that point to `localhost`;
- links the Supabase project when `--project-ref` is provided;
- sets the `BACKEND_BASE_URL` secret;
- deploys the Edge Function facade.

The current function name remains `sidewall-health` for compatibility, but it now
forwards the incoming path to the FastAPI backend. For example:

```bash
curl https://<project-ref>.supabase.co/functions/v1/sidewall-health/health
curl https://<project-ref>.supabase.co/functions/v1/sidewall-health/api/model/status
```

## Local Fallback

If production is down, do not change code. Start the local server with `.env.local`:

```bash
uv run python -m sidewall.server.app
```

Then use:

- `http://localhost:8000/`
- `http://localhost:8000/pitwall?mode=replay`
- `http://localhost:8000/docs`

The server loads `.env.local` by default. Set `SIDEWALL_ENV=prod` only when you
explicitly want `.env.prod`.
