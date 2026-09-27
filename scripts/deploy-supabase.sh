#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.prod}"
FUNCTION_NAME="${FUNCTION_NAME:-sidewall-health}"
PROJECT_REF="${SUPABASE_PROJECT_REF:-}"
SKIP_LINK="${SKIP_LINK:-0}"
SKIP_SECRETS="${SKIP_SECRETS:-0}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SERVE_LOCAL="${SERVE_LOCAL:-0}"

usage() {
  cat <<'EOF'
Deploy SIDEWALL Supabase Edge Function helpers.

Usage:
  bash scripts/deploy-supabase.sh [options]

Options:
  --env-file PATH        Env file to read, default .env.prod
  --project-ref REF      Supabase project ref for `supabase link`
  --function NAME        Function to deploy, default sidewall-health
  --skip-link            Do not run `supabase link`
  --skip-secrets         Do not run `supabase secrets set`
  --skip-deploy          Do not deploy, useful for validating config
  --serve-local          Run `supabase functions serve` instead of deploying
  -h, --help             Show this help

Windows:
  Run from Git Bash, WSL, or any shell that has bash and npx available.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --project-ref)
      PROJECT_REF="$2"
      shift 2
      ;;
    --function)
      FUNCTION_NAME="$2"
      shift 2
      ;;
    --skip-link)
      SKIP_LINK=1
      shift
      ;;
    --skip-secrets)
      SKIP_SECRETS=1
      shift
      ;;
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --serve-local)
      SERVE_LOCAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  if [[ "$ENV_FILE" == *".env.local"* ]]; then
    echo "Create it from .env.local.example first." >&2
  else
    echo "Create it from .env.prod.example first." >&2
  fi
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required. Install Node.js or run this from a shell where npx is available." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

BACKEND_BASE_URL="${BACKEND_BASE_URL:-}"
SUPABASE_PROJECT_REF="${PROJECT_REF:-${SUPABASE_PROJECT_REF:-}}"

if [[ -z "$BACKEND_BASE_URL" ]]; then
  echo "BACKEND_BASE_URL is empty in $ENV_FILE." >&2
  echo "For local smoke tests use http://localhost:8000. For production use a public HTTPS backend URL." >&2
  exit 1
fi

if [[ "$SERVE_LOCAL" != "1" && "$SKIP_DEPLOY" != "1" && "$BACKEND_BASE_URL" =~ ^http://(localhost|127\.0\.0\.1)(:|/|$) ]]; then
  echo "Refusing production deploy with BACKEND_BASE_URL=$BACKEND_BASE_URL" >&2
  echo "Supabase production cannot call your laptop localhost. Use --serve-local for local tests." >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ "$SKIP_LINK" != "1" ]]; then
  if [[ -z "$SUPABASE_PROJECT_REF" ]]; then
    echo "No project ref provided; skipping link. Pass --project-ref REF when linking is needed."
  else
    npx supabase link --project-ref "$SUPABASE_PROJECT_REF"
  fi
fi

if [[ "$SERVE_LOCAL" == "1" ]]; then
  echo "Serving $FUNCTION_NAME locally with $ENV_FILE"
  npx supabase functions serve "$FUNCTION_NAME" --env-file "$ENV_FILE"
  exit 0
fi

if [[ "$SKIP_SECRETS" != "1" ]]; then
  npx supabase secrets set "BACKEND_BASE_URL=$BACKEND_BASE_URL"
fi

if [[ "$SKIP_DEPLOY" != "1" ]]; then
  npx supabase functions deploy "$FUNCTION_NAME"
  echo "Done. Deployed function: $FUNCTION_NAME"
else
  echo "Done. Validated deployment config for function: $FUNCTION_NAME"
fi

echo "Backend target: $BACKEND_BASE_URL"
