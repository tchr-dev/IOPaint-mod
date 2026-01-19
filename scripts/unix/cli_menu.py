#!/usr/bin/env python3
"""
Interactive CLI main menu for IOPaint using questionary.
Provides professional menu interface with icons, shortcuts, and sub-menus.
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

def main_menu() -> Optional[str]:
    """Display main interactive menu with all command categories."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "🚀 Development",
        "🧪 Testing",
        "🌐 Production", 
        "🛠️  Utilities",
        "❓ Help",
        "🚪 Quit"
    ]

    choice = questionary.select(
        "IOPaint Development CLI - Main Menu:",
        choices=options,
        instruction="(↑/↓ to navigate • d/t/p/u/h/q shortcuts • Enter to select • Esc to quit)",
        use_shortcuts=True,
    ).ask()

    if choice is None:
        return "quit"

    # Map to option numbers (1-based)
    choice_map = {
        "🚀 Development": "1",
        "🧪 Testing": "2", 
        "🌐 Production": "3",
        "🛠️  Utilities": "4",
        "❓ Help": "5",
        "🚪 Quit": "6"
    }
    
    return choice_map.get(choice, "invalid")

def development_menu() -> Optional[str]:
    """Development sub-menu."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "🚀 Start dev server",
        "🏗️  Build frontend",
        "🛑 Stop services",
        "← Back to main menu"
    ]

    choice = questionary.select(
        "Development Options:",
        choices=options,
        instruction="(↑/↓ to navigate • Enter to select • Esc to go back)",
    ).ask()

    if choice is None or "Back" in choice:
        return "back"

    choice_map = {
        "🚀 Start dev server": "dev",
        "🏗️  Build frontend": "build", 
        "🛑 Stop services": "stop"
    }
    
    return choice_map.get(choice, "invalid")

def production_menu() -> Optional[str]:
    """Production sub-menu."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "🌐 Start prod server",
        "📦 Publish assets",
        "← Back to main menu"
    ]

    choice = questionary.select(
        "Production Options:",
        choices=options,
        instruction="(↑/↓ to navigate • Enter to select • Esc to go back)",
    ).ask()

    if choice is None or "Back" in choice:
        return "back"

    choice_map = {
        "🌐 Start prod server": "prod",
        "📦 Publish assets": "publish"
    }
    
    return choice_map.get(choice, "invalid")

def utilities_menu() -> Optional[str]:
    """Utilities sub-menu."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "⚙️  Job utilities",
        "🐳 Docker utilities", 
        "← Back to main menu"
    ]

    choice = questionary.select(
        "Utilities:",
        choices=options,
        instruction="(↑/↓ to navigate • Enter to select • Esc to go back)",
    ).ask()

    if choice is None or "Back" in choice:
        return "back"

    choice_map = {
        "⚙️  Job utilities": "jobs",
        "🐳 Docker utilities": "docker"
    }
    
    return choice_map.get(choice, "invalid")

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

def test_main_menu() -> Optional[str]:
    """Display test main menu."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "📋 List all test files",
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
        "← Back to main menu"
    ]

    choice = questionary.select(
        "Testing Options:",
        choices=options,
        instruction="(↑/↓ to navigate • Enter to select • Esc to go back)",
    ).ask()

    if choice is None or "Back" in choice:
        return "back"

    # Return option number (1-based, but adjust for back)
    choice_map = {
        "📋 List all test files": "1",
        "🚀 Backend smoke (iopaint/tests/test_model.py)": "2",
        "🧪 Backend full (pytest -v)": "3",
        "📄 Backend single test file": "4",
        "🔍 Backend single test name (-k)": "5",
        "⚙️  Backend custom pytest args": "6",
        "🧹 Test lint (ruff check/format)": "7",
        "🏗️  Frontend build (npm run build)": "8",
        "🔎 Frontend lint (npm run lint)": "9",
        "🔧 Frontend custom npm script": "10",
        "❓ Help (detailed help)": "11",
    }
    
    return choice_map.get(choice, "invalid")

def test_file_menu() -> Optional[str]:
    """Interactive test file selection."""
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

    choices.append("← Back to test menu")

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
    # Only check stdin since stdout may be captured by command substitution
    return os.isatty(0)

def main():
    """Main entry point."""
    if not is_interactive():
        # Not interactive, fall back to bash menu
        sys.exit(1)

    try:
        menu_type = sys.argv[1] if len(sys.argv) > 1 else "main"
        
        if menu_type == "main":
            result = main_menu()
        elif menu_type == "development":
            result = development_menu()
        elif menu_type == "production":
            result = production_menu()
        elif menu_type == "utilities":
            result = utilities_menu()
        elif menu_type == "test-main":
            result = test_main_menu()
        elif menu_type == "test-files":
            result = test_file_menu()
        else:
            result = "invalid"

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