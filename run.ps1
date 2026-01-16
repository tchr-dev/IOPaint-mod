#!/usr/bin/env pwsh
# IOPaint CLI - Unified development workflow for Windows.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LibDir = Join-Path $ScriptDir "scripts\lib"

. "$LibDir\common.ps1"

function Show-Usage {
    @"
IOPaint CLI - Unified development workflow.

Usage:
  .\run.ps1 <command> [args]

Commands:
  dev           Start backend + Vite dev server
  prod          Build frontend, copy assets, start backend
  build         Build frontend only
  test          Run tests (interactive or named suite)

Run:
  .\run.ps1 <command> --help
"@
}

$cmd = $args[0]
$remainingArgs = $args[1..$args.Count]

switch ($cmd) {
    { $_ -in @("-h", "--help", "help", $null) } {
        Show-Usage
        exit 0
    }
    { $_ -in @("dev", "prod", "build", "test") } {
        $root = Set-ProjectRoot
        $scriptPath = Join-Path $ScriptDir "scripts\windows\$cmd.ps1"
        if (Test-Path $scriptPath) {
            & $scriptPath @remainingArgs
        }
        else {
            Write-Error "Command script not found: $scriptPath"
        }
    }
    default {
        Write-Error "Unknown command: $cmd. Run --help."
    }
}
