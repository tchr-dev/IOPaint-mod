#!/usr/bin/env pwsh
# Build frontend only.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LibDir = Join-Path $ScriptDir "..\..\lib"
. "$LibDir\common.ps1"

$ForceNpm = $false

function Show-Usage {
    @"
Usage: .\run.ps1 build [--npm-force]

Builds frontend (web_app/dist) without starting the backend.
"@
}

$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "--npm-force" { $ForceNpm = $true; $i++ }
        { $_ -in @("-h", "--help") } { Show-Usage; exit 0 }
        default { Write-Error "Unknown argument: $($args[$i])" }
    }
}

Require-Command -Commands @("node", "npm")

Install-NpmDeps -AppDir "web_app" -Force:$ForceNpm

Write-Log "Building frontend..."
Push-Location "web_app"
try {
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    npm run build
}
finally {
    Pop-Location
}

Write-Log "Frontend built successfully (web_app/dist)"
