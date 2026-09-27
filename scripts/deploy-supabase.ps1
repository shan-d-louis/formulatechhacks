param(
    [string]$EnvFile = ".env.prod",
    [string]$ProjectRef = $env:SUPABASE_PROJECT_REF,
    [string]$FunctionName = "sidewall-health",
    [switch]$SkipLink,
    [switch]$SkipSecrets,
    [switch]$SkipDeploy,
    [switch]$ServeLocal
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPath = if ([System.IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
} else {
    Join-Path $RootDir $EnvFile
}

if (-not (Test-Path -LiteralPath $EnvPath)) {
    $Example = if ($EnvPath -like "*.env.local*") { ".env.local.example" } else { ".env.prod.example" }
    throw "Missing env file: $EnvPath. Create it from $Example first."
}

if (-not (Get-Command npx -ErrorAction SilentlyContinue)) {
    throw "npx is required. Install Node.js or run from a shell where npx is available."
}

Get-Content -LiteralPath $EnvPath | ForEach-Object {
    $Line = $_.Trim()
    if (-not $Line -or $Line.StartsWith("#") -or -not $Line.Contains("=")) {
        return
    }
    $Parts = $Line.Split("=", 2)
    $Key = $Parts[0].Trim()
    $Value = $Parts[1].Trim().Trim('"').Trim("'")
    if ($Key) {
        [Environment]::SetEnvironmentVariable($Key, $Value, "Process")
    }
}

$BackendBaseUrl = $env:BACKEND_BASE_URL
if (-not $BackendBaseUrl) {
    throw "BACKEND_BASE_URL is empty in $EnvPath. Use http://localhost:8000 for local smoke tests or a public HTTPS backend URL for production."
}

if (-not $ServeLocal -and -not $SkipDeploy -and $BackendBaseUrl -match "^http://(localhost|127\.0\.0\.1)(:|/|$)") {
    throw "Refusing production deploy with BACKEND_BASE_URL=$BackendBaseUrl. Use -ServeLocal for local tests."
}

Set-Location $RootDir

if (-not $SkipLink) {
    if ($ProjectRef) {
        npx supabase link --project-ref $ProjectRef
    } else {
        Write-Host "No project ref provided; skipping link. Pass -ProjectRef when linking is needed."
    }
}

if ($ServeLocal) {
    Write-Host "Serving $FunctionName locally with $EnvPath"
    npx supabase functions serve $FunctionName --env-file $EnvPath
    exit $LASTEXITCODE
}

if (-not $SkipSecrets) {
    npx supabase secrets set "BACKEND_BASE_URL=$BackendBaseUrl"
}

if (-not $SkipDeploy) {
    npx supabase functions deploy $FunctionName
    Write-Host "Done. Deployed function: $FunctionName"
} else {
    Write-Host "Done. Validated deployment config for function: $FunctionName"
}

Write-Host "Backend target: $BackendBaseUrl"
