#!/usr/bin/env python3
"""
IOPaint Unified Management CLI
"""
import sys
import os
import subprocess
import signal
from pathlib import Path
from typing import Optional, List, Tuple

# Third-party imports (will be checked/installed via uv)
try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    import questionary
except ImportError:
    print("Installing required dependencies (typer, rich, questionary)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "typer", "rich", "questionary"], check=True)
    import typer
    from rich.console import Console
    from rich.panel import Panel
    import questionary

app = typer.Typer(help="IOPaint Unified Management CLI")
console = Console()

import shutil
import time

# --- Helpers ---
def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None, capture_output: bool = False) -> int:
    """Run a command in a subprocess."""
    cmd_str = ' '.join(cmd)
    console.print(f"[bold blue]Running:[/bold blue] {cmd_str}")
    try:
        if capture_output:
             result = subprocess.run(cmd, cwd=cwd, env=env, check=False, capture_output=True, text=True)
             return result
        else:
             result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
             return result.returncode
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Error running command:[/bold red] {e}")
        return 1

def check_port(port: int) -> bool:
    """Check if a port is in use (simple implementation)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_port(port: int):
    """Kill process listening on a port (Unix/Mac implementation for now)."""
    # Try lsof
    try:
        res = subprocess.run(["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            pids = res.stdout.strip().split('\n')
            console.print(f"[yellow]Killing processes on port {port}: {', '.join(pids)}[/yellow]")
            subprocess.run(["kill", "-9"] + pids, check=False)
    except FileNotFoundError:
        console.print("[red]lsof not found, cannot kill by port automatically.[/red]")

# --- Commands ---

@app.command()
def stop(
    backend_port: int = typer.Option(8080, help="Backend port"),
    frontend_port: int = typer.Option(5173, help="Frontend port"),
):
    """Stop running services."""
    kill_port(backend_port)
    kill_port(frontend_port)
    console.print("[green]Services stopped.[/green]")

@app.command()
def dev(
    model: str = typer.Option("openai-compat", help="Model name"),
    port: int = typer.Option(8080, help="Backend port"),
    frontend_port: int = typer.Option(5173, help="Frontend port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    no_sync: bool = typer.Option(False, "--no-sync", help="Skip python sync"),
    no_npm: bool = typer.Option(False, "--no-npm", help="Skip npm install"),
):
    """Start backend + Vite dev server."""
    
    # 1. Sync Python
    if not no_sync:
        if run_cmd(["uv", "sync"]) != 0:
            raise typer.Exit(1)
            
    # 2. NPM Install
    web_app_dir = Path("web_app")
    if not no_npm:
        if (web_app_dir / "package.json").exists():
             if not (web_app_dir / "node_modules").exists():
                 console.print("[yellow]Installing frontend dependencies...[/yellow]")
                 if run_cmd(["npm", "install"], cwd=web_app_dir) != 0:
                     raise typer.Exit(1)

    # 3. Start Servers
    env = os.environ.copy()
    if verbose:
        env["IOPAINT_VERBOSE"] = "1"
        
    backend_cmd = ["uv", "run", "python", "main.py", "start", "--model", model, "--port", str(port)]
    frontend_cmd = ["npm", "run", "dev"]
    
    console.print(Panel(f"Starting IOPaint Dev\nBackend: {port}\nFrontend: {frontend_port}", style="bold green"))
    
    try:
        # Start Backend
        p_backend = subprocess.Popen(backend_cmd, env=env)
        
        # Start Frontend
        p_frontend = subprocess.Popen(frontend_cmd, cwd=web_app_dir, env=env)
        
        # Wait for either to exit
        while True:
            if p_backend.poll() is not None:
                console.print("[red]Backend exited![/red]")
                break
            if p_frontend.poll() is not None:
                console.print("[red]Frontend exited![/red]")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping servers...[/yellow]")
    finally:
        p_backend.terminate()
        p_frontend.terminate()
        try:
            p_backend.wait(timeout=2)
            p_frontend.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p_backend.kill()
            p_frontend.kill()

@app.command()
def prod(
    model: str = typer.Option("openai-compat", help="Model name"),
    port: int = typer.Option(8080, help="Backend port"),
    no_sync: bool = typer.Option(False, help="Skip python sync"),
    no_npm: bool = typer.Option(False, help="Skip npm install"),
):
    """Build frontend and start backend."""
    # 1. Sync & Install
    if not no_sync and run_cmd(["uv", "sync"]) != 0: raise typer.Exit(1)
    
    web_app_dir = Path("web_app")
    if not no_npm and run_cmd(["npm", "install"], cwd=web_app_dir) != 0: raise typer.Exit(1)
    
    # 2. Build Frontend
    console.print("[bold]Building frontend...[/bold]")
    shutil.rmtree(web_app_dir / "dist", ignore_errors=True)
    if run_cmd(["npm", "run", "build"], cwd=web_app_dir) != 0: raise typer.Exit(1)
    
    # 3. Copy Assets
    console.print("[bold]Copying assets...[/bold]")
    dest_dir = Path("iopaint/web_app")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in dest_dir.iterdir():
        if item.name != ".gitkeep": # prevent deleting gitkeep if exists, optional
             if item.is_dir(): shutil.rmtree(item)
             else: item.unlink()
             
    src_dist = web_app_dir / "dist"
    if src_dist.exists():
        shutil.copytree(src_dist, dest_dir, dirs_exist_ok=True)
    else:
        console.print("[red]Frontend build failed? dist folder missing.[/red]")
        raise typer.Exit(1)
        
    # 4. Start Backend
    # 4. Start Backend
    run_cmd(["uv", "run", "python", "main.py", "start", "--model", model, "--port", str(port)])

def npm_install_if_needed(app_dir: Path, force: bool):
    """Check and run npm install."""
    if not app_dir.exists():
        console.print(f"[red]Directory not found: {app_dir}[/red]")
        raise typer.Exit(1)
        
    if not force and (app_dir / "node_modules").exists():
        return
        
    console.print(f"[bold]Installing dependencies in {app_dir}...[/bold]")
    if (app_dir / "package-lock.json").exists():
        cmd = ["npm", "ci"]
    else:
        cmd = ["npm", "install"]
        
    if run_cmd(cmd, cwd=app_dir) != 0:
        raise typer.Exit(1)

@app.command()
def build(
    npm_force: bool = typer.Option(False, help="Force npm install"),
):
    """Build frontend only."""
    web_app_dir = Path("web_app")
    npm_install_if_needed(web_app_dir, npm_force)
    
    console.print("[bold]Building frontend...[/bold]")
    shutil.rmtree(web_app_dir / "dist", ignore_errors=True)
    if run_cmd(["npm", "run", "build"], cwd=web_app_dir) != 0: raise typer.Exit(1)
    console.print(f"[green]Frontend built successfully ({web_app_dir}/dist)[/green]")

@app.command()
def publish(
    skip_frontend: bool = typer.Option(False, help="Skip frontend build"),
    clean: bool = typer.Option(False, help="Clean web_app/dist"),
):
    """Build frontend assets + python sdist/wheel."""
    if not skip_frontend:
        web_app_dir = Path("web_app")
        if clean:
            shutil.rmtree(web_app_dir / "dist", ignore_errors=True)
            shutil.rmtree("iopaint/web_app", ignore_errors=True)
            
        npm_install_if_needed(web_app_dir, False)
        
        console.print("[bold]Building frontend...[/bold]")
        shutil.rmtree(web_app_dir / "dist", ignore_errors=True)
        if run_cmd(["npm", "run", "build"], cwd=web_app_dir) != 0: raise typer.Exit(1)
        
        console.print("[bold]Copying assets...[/bold]")
        dest_dir = Path("iopaint/web_app")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(dest_dir, ignore_errors=True)
        shutil.copytree(web_app_dir / "dist", dest_dir)
        
    console.print("[bold]Building python distributions...[/bold]")
    shutil.rmtree("dist", ignore_errors=True)
    if run_cmd(["python3", "setup.py", "sdist", "bdist_wheel"]) != 0: raise typer.Exit(1)
    console.print("[green]Done. Artifacts in ./dist[/green]")

@app.command()
def docker(
    subcommand: str = typer.Argument(..., help="Subcommand: build"),
    tag: str = typer.Option(..., help="Image tag (version)"),
    cpu: bool = typer.Option(False, help="Build CPU image"),
    gpu: bool = typer.Option(False, help="Build GPU image"),
    repo: str = typer.Option("cwq1913/lama-cleaner", help="Image repo"),
    platform: str = typer.Option("linux/amd64", help="Docker platform"),
    push: bool = typer.Option(False, help="Push image"),
    load: bool = typer.Option(False, help="Load image"),
):
    """Docker utilities."""
    if subcommand != "build":
        console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
        raise typer.Exit(1)
        
    if not cpu and not gpu:
        cpu = True
        gpu = True
        
    mode = "--push" if push else "--load" if load else ""
    desc = "Image inpainting tool powered by SOTA AI Model"
    repo_url = os.environ.get("GIT_REPO_URL", "https://github.com/Sanster/lama-cleaner")
    
    common_args = [
        "docker", "buildx", "build",
        "--platform", platform,
        "--label", "org.opencontainers.image.title=lama-cleaner",
        "--label", f"org.opencontainers.image.description={desc}",
        "--label", f"org.opencontainers.image.url={repo_url}",
        "--label", f"org.opencontainers.image.source={repo_url}",
        "--label", f"org.opencontainers.image.version={tag}",
        "--build-arg", f"version={tag}",
    ]
    
    if mode:
        common_args.append(mode)
        
    if cpu:
        console.print(f"[bold]Building CPU image: {repo}:cpu-{tag}[/bold]")
        cmd = common_args + [
            "--file", "./docker/CPUDockerfile",
            "--tag", f"{repo}:cpu-{tag}",
            "."
        ]
        run_cmd(cmd)

    if gpu:
        console.print(f"[bold]Building GPU image: {repo}:gpu-{tag}[/bold]")
        cmd = common_args + [
             "--file", "./docker/GPUDockerfile",
             "--tag", f"{repo}:gpu-{tag}",
             "."
        ]
        run_cmd(cmd)

@app.command()
def jobs(
    action: str = typer.Argument(..., help="Action: cancel"),
    url: str = typer.Option("http://127.0.0.1:8080", help="IOPaint server URL"),
    db: Path = typer.Option(Path(os.environ.get("HOME", "")) / ".iopaint/data/budget.db", help="Path to budget.db"),
    dry_run: bool = typer.Option(False, help="Dry run"),
):
    """Job utilities."""
    if action != "cancel":
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)
        
    console.print(f"Cancelling stuck jobs on {url}...")
    
    if not db.exists():
        console.print(f"[red]Database not found at {db}[/red]")
        raise typer.Exit(1)
        
    import sqlite3
    import httpx
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT id, session_id FROM generation_jobs WHERE status IN ('running','queued')")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        console.print("No stuck jobs found.")
        return
        
    console.print(f"Found {len(rows)} stuck job(s).")
    
    for job_id, session_id in rows:
        console.print(f"Cancelling job: {job_id} (session: {session_id[:8]}...)")
        
        if dry_run:
            console.print(f"  (dry-run) POST {url}/api/v1/openai/jobs/{job_id}/cancel")
            continue
            
        try:
            resp = httpx.post(
                f"{url}/api/v1/openai/jobs/{job_id}/cancel",
                headers={"X-Session-Id": session_id},
                timeout=5.0
            )
            data = resp.json()
            status = data.get("status", "error")
            if status == "cancelled":
                console.print("  Cancelled")
            else:
                console.print(f"  [yellow]Status: {status}[/yellow]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            
    console.print("[bold yellow]IMPORTANT: Clear browser state to fix stuck 'Generating...'[/bold yellow]")
    console.print("Run in browser console: localStorage.removeItem('ZUSTAND_STATE'); location.reload()")

# --- Test Helpers ---
def extract_description(filename: str) -> str:
    """Extract test description from filename."""
    # Map common test file patterns to descriptions
    name_map = {
        'adjust_mask': 'adjust mask',
        'anytext': 'anytext',
        'api_error_handling': 'api error handling',
        'brushnet': 'brushnet',
        'budget_guard': 'budget guard',
        'budget_limits': 'budget limits',
        'controlnet': 'controlnet',
        'dedupe_fingerprint': 'dedupe fingerprint',
        'external_services_config': 'external services config',
        'history_snapshots_api': 'history snapshots api',
        'instruct_pix2pix': 'instruct pix2pix',
        'load_img': 'load img',
        'low_mem': 'low mem',
        'match_histograms': 'match histograms',
        'model_md5': 'model md5',
        'model_switch': 'model switch',
        'model': 'model',
        'models_cache_api': 'models cache api',
        'openai_capabilities': 'openai capabilities',
        'openai_client': 'openai client',
        'openai_errors': 'openai errors',
        'openai_protocol_integration': 'openai protocol integration',
        'openai_tools_api': 'openai tools api',
        'outpainting': 'outpainting',
        'paint_by_example': 'paint by example',
        'plugins': 'plugins',
        'save_exif': 'save exif',
        'save_quality': 'save quality',
        'sd_model': 'sd model',
        'sdxl': 'sdxl',
    }
    base_name = filename.replace('test_', '').replace('.py', '')
    return name_map.get(base_name, base_name.replace('_', ' '))

def categorize_test(filename: str) -> str:
    """Categorize a test file."""
    if 'api' in filename or 'history_snapshots' in filename:
        return 'API'
    elif 'model' in filename or 'sd' in filename or 'adjust_mask' in filename or 'load_img' in filename or 'save' in filename:
        return 'Models'
    elif 'openai' in filename:
        return 'OpenAI'
    elif 'budget' in filename:
        return 'Budget'
    elif 'plugins' in filename:
        return 'Plugins'
    else:
        return 'Other'

def get_test_files() -> List[Tuple[str, str]]:
    """Scan for backend test files."""
    test_dir = Path("iopaint/tests")
    if not test_dir.exists():
        return []
    test_files = []
    for file_path in sorted(test_dir.glob("test_*.py")):
        filename = file_path.name
        description = extract_description(filename)
        test_files.append((filename, description))
    return test_files

def get_frontend_test_files() -> List[Tuple[str, str]]:
    """Scan for frontend test files."""
    test_dir = Path("web_app/src")
    if not test_dir.exists():
        return []
    test_files = []
    for file_path in sorted(test_dir.rglob("*.test.tsx")):
        rel_path = file_path.relative_to(test_dir)
        filename = str(rel_path)
        description = filename.replace('.test.tsx', '').replace('/', ' > ')
        test_files.append((filename, description))
    return test_files

@app.command()
def test(
    suite: str = typer.Argument(None, help="Test suite to run (smoke, full, file, fe-test, etc.)"),
    file: str = typer.Option(None, help="Specific file for 'file' or 'fe-test' suite"),
    k: str = typer.Option(None, "-k", help="Pytest k pattern"),
    args: str = typer.Option(None, help="Custom args"),
):
    """Run tests."""
    
    # Non-interactive modes
    if suite == "smoke":
        run_cmd(["uv", "run", "pytest", "iopaint/tests/test_model.py", "-v"])
        return
    elif suite == "full":
        run_cmd(["uv", "run", "pytest", "-v"])
        return
    elif suite == "file":
        if not file:
             console.print("[red]--file is required for 'file' suite[/red]")
             raise typer.Exit(1)
        run_cmd(["uv", "run", "pytest", file, "-v"])
        return
    elif suite == "k":
        if not k:
             console.print("[red]-k is required for 'k' suite[/red]")
             raise typer.Exit(1)
        run_cmd(["uv", "run", "pytest", "-k", k, "-v"])
        return
    elif suite == "custom":
        if not args:
             console.print("[red]--args is required for 'custom' suite[/red]")
             raise typer.Exit(1)
        # Split args string into list safely? For now just use shell=True logic equivalent via bash wrapper if complex,
        # but pure python subprocess array is better. 
        # But args string might contain quotes.
        # Simple split is risky. Let's just pass as is if we can, or split by space.
        run_cmd(["uv", "run", "pytest"] + args.split())
        return
    elif suite == "test-lint":
        run_cmd(["uv", "run", "ruff", "check", "iopaint/tests/"])
        run_cmd(["uv", "run", "ruff", "format", "--check", "iopaint/tests/"])
        return
        
    # Frontend non-interactive
    elif suite == "fe-test":
        cmd = ["npm", "run", "test"]
        web_app_dir = Path("web_app")
        if file:
            cmd.extend(["--", file])
        elif k:
             cmd.extend(["--", "-k", k])
        
        run_cmd(cmd, cwd=web_app_dir)
        return
    elif suite == "fe-lint":
        run_cmd(["npm", "run", "lint"], cwd=Path("web_app"))
        return
    elif suite == "fe-build":
        run_cmd(["npm", "run", "build"], cwd=Path("web_app"))
        return
        
    # Interactive Menu
    options = [
        "📋 List all test files",
        "📋 List frontend test files",
        "🚀 Backend smoke",
        "🧪 Backend full",
        "🧹 Test lint",
        "🏗️  Frontend build",
        "🔎 Frontend lint",
        "← Back"
    ]
    
    choice = questionary.select(
        "Testing Options:",
        choices=options,
        use_shortcuts=True
    ).ask()
    
    if choice is None or "Back" in choice:
        return
        
    if "List all test files" in choice:
        # File sub-menu
        test_files = get_test_files()
        if not test_files:
            console.print("[red]No tests found.[/red]")
            return
            
        # Group by category logic... for brevity let's just list them sorted
        # Or re-implement the categories.
        # Let's use questionary
        file_choices = []
        for f, d in test_files:
            file_choices.append(f"{f} - {d}")
        file_choices.append("Back")
        
        f_choice = questionary.select("Select test file:", choices=file_choices).ask()
        if f_choice and "Back" not in f_choice:
            filename = f_choice.split(" - ")[0].strip()
            run_cmd(["uv", "run", "pytest", f"iopaint/tests/{filename}", "-v"])
            
    elif "List frontend test files" in choice:
        fs = get_frontend_test_files()
        if not fs:
            console.print("[red]No frontend tests found.[/red]")
            return
        f_choices = [f"{f} - {d}" for f, d in fs] + ["Back"]
        f_choice = questionary.select("Select frontend test:", choices=f_choices).ask()
        if f_choice and "Back" not in f_choice:
            fname = f_choice.split(" - ")[0].strip()
            run_cmd(["npm", "run", "test", "--", fname], cwd=Path("web_app"))
            
    elif "Backend smoke" in choice:
        test(suite="smoke")
    elif "Backend full" in choice:
        test(suite="full")
    elif "Test lint" in choice:
        test(suite="test-lint")
    elif "Frontend build" in choice:
        test(suite="fe-build")
    elif "Frontend lint" in choice:
        test(suite="fe-lint")

@app.command()
def interact():
    """Show interactive main menu."""
    options = [
        "🚀 Development",
        "🧪 Testing",
        "🌐 Production",
        "🏗️  Build Frontend",
        "📦 Publish",
        "🐳 Docker",
        "⚙️  Jobs",
        "🛑 Stop Services", 
        "❓ Help",
        "🚪 Quit"
    ]
    
    while True:
        choice = questionary.select(
            "IOPaint Main Menu:",
            choices=options,
            use_shortcuts=True
        ).ask()
        
        if choice is None or "Quit" in choice:
            break
            
        if "Development" in choice:
            dev(model="openai-compat", port=8080, frontend_port=5173, verbose=False, no_sync=False, no_npm=False)
        elif "Testing" in choice:
             test(suite=None)
        elif "Production" in choice:
             prod(model="openai-compat", port=8080, no_sync=False, no_npm=False)
        elif "Build" in choice:
             build(npm_force=False)
        elif "Publish" in choice:
             publish(skip_frontend=False, clean=False)
        elif "Docker" in choice:
             console.print("[yellow]Use CLI for docker build to pass arguments.[/yellow]")
             # TODO: Interactive docker builder?
             run_cmd(["uv", "run", "manage.py", "docker", "--help"])
        elif "Jobs" in choice:
             jobs(action="cancel", url="http://127.0.0.1:8080", db=Path(os.environ.get("HOME", "")) / ".iopaint/data/budget.db", dry_run=False)
        elif "Stop" in choice:
             stop(backend_port=8080, frontend_port=5173)
        elif "Help" in choice:
             run_cmd(["uv", "run", "manage.py", "--help"])


# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) == 1:
        interact()
    else:
        app()
