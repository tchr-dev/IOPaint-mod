#!/usr/bin/env pwsh
# Run tests.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LibDir = Join-Path $ScriptDir "..\..\lib"
. "$LibDir\common.ps1"

$LogDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { "./logs" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogFile = Join-Path $LogDir "test-run-$Timestamp.log"

function Show-Usage {
    @"
Usage:
  .\run.ps1 test               # interactive menu
  .\run.ps1 test smoke|full|file|k|custom|fe-build|fe-lint [...]

Examples:
  .\run.ps1 test smoke
  .\run.ps1 test file "iopaint\tests\test_model.py"
  .\run.ps1 test k "Model"
  .\run.ps1 test custom "-k Model -v"
"@
}

function Print-Menu {
    @"
Select a test suite:
  1) Backend smoke (iopaint/tests/test_model.py)
  2) Backend full (pytest -v)
  3) Backend single test file
  4) Backend single test name (-k)
  5) Backend custom pytest args
  6) Frontend build (npm run build)
  7) Frontend lint (npm run lint)
  8) Frontend custom npm script
  9) Quit
"@
}

$suite = $args[0]
$remainingArgs = $args[1..$args.Count]

if ($suite -in @("-h", "--help")) {
    Show-Usage
    exit 0
}

Write-Host ""
Write-Host "Friendly Test Runner" -ForegroundColor Cyan
Write-Host "Log file: $LogFile"
Write-Host ""

if (-not $suite) {
    Print-Menu
    $choice = Read-Host "Enter choice [1-9]"
    $suite = switch ($choice) {
        "1" { "smoke" }
        "2" { "full" }
        "3" { "file" }
        "4" { "k" }
        "5" { "custom" }
        "6" { "fe-build" }
        "7" { "fe-lint" }
        "8" { "fe-custom" }
        "9" { "quit" }
        default { "invalid" }
    }
}

function Run-Cmd {
    param([string]$Cmd)
    Write-Host "Running: $Cmd" -ForegroundColor Cyan
    Add-Content -Path $LogFile -Value "Running: $Cmd"
    Add-Content -Path $LogFile -Value "---"
    $output = Invoke-Expression $Cmd 2>&1 | Tee-Object -FilePath $LogFile
    Add-Content -Path $LogFile -Value "---"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Command failed with status $LASTEXITCODE" -ForegroundColor Red
        Add-Content -Path $LogFile -Value "Command failed with status $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

switch ($suite) {
    "smoke" {
        Run-Cmd "uv run pytest iopaint/tests/test_model.py -v"
    }
    "full" {
        Run-Cmd "uv run pytest -v"
    }
    "file" {
        $file = $remainingArgs[0]
        if (-not $file) { Write-Error "Provide test file path." }
        Run-Cmd "uv run pytest $file -v"
    }
    "k" {
        $pattern = $remainingArgs[0]
        if (-not $pattern) { Write-Error "Provide -k pattern." }
        Run-Cmd "uv run pytest -k `"$pattern`" -v"
    }
    "custom" {
        $argsStr = $remainingArgs -join " "
        if (-not $argsStr) { Write-Error "Provide custom pytest args (quoted)." }
        Run-Cmd "uv run pytest $argsStr"
    }
    "fe-build" {
        Run-Cmd "cd web_app; npm run build"
    }
    "fe-lint" {
        Run-Cmd "cd web_app; npm run lint"
    }
    "fe-custom" {
        $script = $remainingArgs[0]
        if (-not $script) { Write-Error "Provide npm script name." }
        Run-Cmd "cd web_app; npm run $script"
    }
    "quit" {
        Write-Host "Bye."
    }
    default {
        Write-Error "Invalid choice/suite. Use: .\run.ps1 test --help"
    }
}

Write-Host "Log saved to $LogFile" -ForegroundColor Green
