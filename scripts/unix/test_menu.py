#!/usr/bin/env python3
"""
Interactive CLI for IOPaint test runner using questionary.
Provides professional menu interface with cursor key navigation and search.
"""
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    QUESTIONARY_AVAILABLE = True
    console = Console()
except ImportError:
    QUESTIONARY_AVAILABLE = False
    console = None

def get_test_files() -> List[Tuple[str, str]]:
    """Lazy load test files only when needed."""
    test_dir = Path("iopaint/tests")
    if not test_dir.exists():
        return []

    test_files = []
    for file_path in sorted(test_dir.glob("test_*.py")):
        filename = file_path.name

        # Extract description from filename
        description = extract_description(filename)
        test_files.append((filename, description))

    return test_files

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

def main_menu() -> Optional[str]:
    """Display main interactive menu with all test options."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "📋 List all test files",
        "📋 List frontend test files",
        "🚀 Backend smoke (iopaint/tests/test_model.py)",
        "🧪 Backend full (pytest -v)",
        "📄 Backend single test file",
        "🔍 Backend single test name (-k)",
        "⚙️  Backend custom pytest args",
        "🧹 Test lint (ruff check/format)",
        "🏗️  Frontend build (npm run build)",
        "🔎 Frontend lint (npm run lint)",
        "🔧 Frontend custom npm script",
        "❓ Help (detailed help)",
        "🚪 Quit"
    ]

    choice = questionary.select(
        "Select a test suite:",
        choices=options,
        instruction="(↑/↓ to navigate • Enter to select • Esc to quit)",
    ).ask()

    if choice is None:
        return "quit"

    # Return option number (1-based)
    return str(options.index(choice) + 1)

def test_file_menu() -> Optional[str]:
    """Interactive test file selection with search and number keys."""
    if not QUESTIONARY_AVAILABLE:
        return None

    test_files = get_test_files()
    if not test_files:
        console.print("[red]No test files found![/red]")
        return "back"

    # Create choices with numbers and descriptions
    choices = []
    for i, (filename, description) in enumerate(test_files, 1):
        choices.append(f"{i:2d}) {filename:<35} - {description}")

    choices.append("← Back to main menu")

    choice = questionary.select(
        f"Choose test to run ({len(test_files)} available):",
        choices=choices,
        instruction="(↑/↓ to navigate • Type to search • Enter to run)",
    ).ask()

    if choice is None or "Back" in choice:
        return "back"

    # Extract test file number
    try:
        file_num = int(choice.split(")")[0].strip())
        if 1 <= file_num <= len(test_files):
            return test_files[file_num - 1][0]  # Return filename
    except (ValueError, IndexError):
        pass

    return "invalid"

def is_interactive():
    """Check if running in an interactive terminal."""
    import os
    return os.isatty(0) and os.isatty(1)

def main():
    """Main entry point."""
    if not is_interactive():
        # Not interactive, fall back to bash menu
        sys.exit(1)

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "test-files":
            result = test_file_menu()
        else:
            result = main_menu()

        if result is None:
            # Fallback signal - Python CLI not available
            sys.exit(1)

        print(result, flush=True)
    except KeyboardInterrupt:
        print("quit", flush=True)
        sys.exit(0)
    except Exception as e:
        # On any error, fall back to bash menu
        sys.exit(1)

if __name__ == "__main__":
    main()