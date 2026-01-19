#!/usr/bin/env pwsh
# Start development servers (backend + frontend).

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LibDir = Join-Path $ScriptDir "..\..\lib"
. "$LibDir\common.ps1"

$Model = $env:IOPAINT_MODEL ?? "lama"
$Port = $env:IOPAINT_PORT ?? "8080"
$Verbose = $env:IOPAINT_VERBOSE ?? ""
$FrontendPort = $env:IOPAINT_FRONTEND_PORT ?? "5173"
$NoSync = $false
$NoNpm = $false
$ForceNpm = $false

function Show-Usage {
    @"
Usage: .\run.ps1 dev [--model MODEL] [--port PORT] [--frontend-port PORT] [--verbose] [--no-sync] [--no-npm] [--npm-force]

Starts:
  - Backend: uv run python main.py start --model ... --port ...
  - Frontend: Vite dev server (web_app)

Options:
  --model           Model name (default: lama)
  --port            Backend port (default: 8080)
  --frontend-port   Frontend port hint for messaging (default: 5173)
  --verbose, -v     Enable verbose logging (sets IOPAINT_VERBOSE)
  --no-sync         Skip 'uv sync'
  --no-npm          Skip npm install step
  --npm-force       Force reinstall frontend deps
"@
}

$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "--model" { $Model = $args[++$i]; $i++ }
        "--port" { $Port = $args[++$i]; $i++ }
        "--frontend-port" { $FrontendPort = $args[++$i]; $i++ }
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

Write-Log "Starting backend (model=$Model, port=$Port) and frontend dev server..."
Write-Log "Frontend will typically be at http://127.0.0.1:$FrontendPort"

$env:IOPAINT_VERBOSE = $Verbose
$backendJob = Start-Job -ScriptBlock {
    param($Model, $Port, $Verbose)
    $env:IOPAINT_VERBOSE = $Verbose
    uv run python main.py start --model $Model --port $Port
} -ArgumentList $Model, $Port, $Verbose

$frontendJob = Start-Job -ScriptBlock {
    Set-Location "web_app"
    npm run dev
}

try {
    Wait-Job $backendJob, $frontendJob -ErrorAction Stop | Out-Null
}
finally {
    Write-Warn "Stopping servers..."
    Stop-Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Get-Job | Where-Object State -eq "Running" | Stop-Job -ErrorAction SilentlyContinue
}
