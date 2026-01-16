# Shared helpers for IOPaint Windows PowerShell scripts.

function Write-Log {
    param([string]$Message)
    Write-Host "[iopaint] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[iopaint][WARN] $Message" -ForegroundColor Yellow 2>&1
}

function Write-Error {
    param([string]$Message)
    Write-Host "[iopaint][ERROR] $Message" -ForegroundColor Red 2>&1
    exit 1
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command -ErrorAction SilentlyContinue -Name $Command
    return $null -ne $null
}

function Require-Command {
    param([string[]]$Commands)
    foreach ($cmd in $Commands) {
        if (-not (Test-Command -Command $cmd)) {
            Write-Error "Missing required command: $cmd"
        }
    }
}

function Find-ProjectRoot {
    $dir = (Get-Location).Path
    while ($dir -and $dir -ne "/") {
        if (Test-Path "$dir/main.py") {
            return $dir
        }
        $dir = Split-Path -Parent $dir
    }

    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $candidate = Join-Path $scriptDir "..\.."
    if (Test-Path "$candidate/main.py") {
        return $candidate
    }

    Write-Error "Could not locate project root (main.py). Run from within the repo."
}

function Set-ProjectRoot {
    $root = Find-ProjectRoot
    Set-Location $root
    return $root
}

function Install-NpmDeps {
    param(
        [string]$AppDir,
        [switch]$Force
    )

    Require-Command -Commands @("npm")

    if (-not (Test-Path $AppDir)) {
        Write-Error "Frontend directory not found: $AppDir"
    }

    $nodeModules = Join-Path $AppDir "node_modules"
    if (-not $Force -and (Test-Path $nodeModules)) {
        return
    }

    Push-Location $AppDir
    try {
        if (Test-Path "package-lock.json") {
            Write-Log "Installing frontend deps (npm ci)..."
            npm ci
        }
        else {
            Write-Log "Installing frontend deps (npm install)..."
            npm install
        }
    }
    finally {
        Pop-Location
    }
}

function Stop-Port {
    param([int]$Port)

    $process = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Ports -and $_.Ports.LocalPort -eq $Port
    }

    if ($process) {
        Write-Log "Stopping processes on port $Port (PIDs: $($process.Id))..."
        $process | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

function Test-PortInUse {
    param([int]$Port)

    $listener = Get-NetTCPConnection -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port -and $_.State -eq "Listen" }
    return $null -ne $listener
}
