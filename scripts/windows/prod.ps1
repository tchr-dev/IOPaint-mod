#!/usr/bin/env pwsh
# Build frontend and start backend for production.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LibDir = Join-Path $ScriptDir "..\..\lib"
. "$LibDir\common.ps1"

$Model = $env:IOPAINT_MODEL ?? "openai-compat"
$Port = $env:IOPAINT_PORT ?? "8080"
$Verbose = $env:IOPAINT_VERBOSE ?? ""
$NoSync = $false
$NoNpm = $false
$ForceNpm = $false

function Show-Usage {
    @"
Usage: .\run.ps1 prod [--model MODEL] [--port PORT] [--verbose] [--no-sync] [--no-npm] [--npm-force]

Builds frontend (web_app/dist), copies it into iopaint/web_app/, then starts backend.
"@
}

$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "--model" { $Model = $args[++$i]; $i++ }
        "--port" { $Port = $args[++$i]; $i++ }
        { $_ -in @("--verbose", "-v") } { $Verbose = "1"; $i++ }
        "--no-sync" { $NoSync = $true; $i++ }
        "--no-npm" { $NoNpm = $true; $i++ }
        "--npm-force" { $ForceNpm = $true; $i++ }
        { $_ -in @("-h", "--help") } { Show-Usage; exit 0 }
        default { Write-Error "Unknown argument: $($args[$i])" }
    }
}

Require-Command -Commands @("uv", "node", "npm")

if (-not $NoSync) {
    Write-Log "Syncing python deps (uv sync)..."
    uv sync
}

if (-not $NoNpm) {
    Install-NpmDeps -AppDir "web_app" -Force:$ForceNpm
}

Write-Log "Building frontend..."
Push-Location "web_app"
try {
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    npm run build
}
finally {
    Pop-Location
}

Write-Log "Copying frontend assets into iopaint/web_app/..."
New-Item -ItemType Directory -Force -Path "iopaint/web_app" | Out-Null
Remove-Item -Recurse -Force "iopaint/web_app/*"
Copy-Item -Recurse -Path "web_app/dist/*" -Destination "iopaint/web_app/"

Write-Log "Starting backend (model=$Model, port=$Port)..."
$env:IOPAINT_VERBOSE = $Verbose
uv run python main.py start --model $Model --port $Port
