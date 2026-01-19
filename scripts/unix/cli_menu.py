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

def get_frontend_test_files() -> List[Tuple[str, str]]:
    """Scan for frontend test files."""
    test_dir = Path("web_app/src")
    if not test_dir.exists():
        return []

    test_files = []
    # Recursively find .test.tsx files
    for file_path in sorted(test_dir.rglob("*.test.tsx")):
        # Use relative path from web_app/src (which is effectively project_root/web_app/src)
        # We want path relative to project root? No, rel to src for display?
        # Let's keep it simple: relative to src for display.
        rel_path = file_path.relative_to(test_dir)
        filename = str(rel_path)
        
        # Simple description from filename
        description = filename.replace('.test.tsx', '').replace('/', ' > ')
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

def test_main_menu() -> Optional[str]:
    """Display test main menu."""
    if not QUESTIONARY_AVAILABLE:
        return None

    options = [
        "📋 List all test files",
        "📋 List frontend test files",
        "🚀 Backend smoke (iopaint/tests/test_model.py)",
        "🧪 Backend full (pytest -v)",
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
        "📋 List frontend test files": "2",
        "🚀 Backend smoke (iopaint/tests/test_model.py)": "3",
        "🧪 Backend full (pytest -v)": "4",
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
    """Interactive test file selection with categories."""
    import os

    DEBUG = os.environ.get('DEBUG', '0') == '1'

    if not QUESTIONARY_AVAILABLE:
        return None

    test_files = get_test_files()

    if DEBUG:
        print(f"DEBUG: test_file_menu called, test_files count = {len(test_files)}", flush=True)

    if not test_files:
        console.print("[red]No test files found![/red]")
        return "back"

    # Check if stdin is a terminal
    if not os.isatty(0):
        if DEBUG:
            print("DEBUG: stdin is not a TTY, using numbered input fallback", flush=True)
        # Fall back to numbered input when not interactive
        print(f"Available tests ({len(test_files)} total):")
        print("")
        for i, (filename, description) in enumerate(sorted(test_files), 1):
            print(f"{i:2d}) {filename:<35} - {description}")
        print("")
        print("Enter test number to run (or 'q' to go back): ", end="", flush=True)

        try:
            user_input = input()
        except EOFError:
            return "back"

        if user_input.lower() in ('q', 'quit', 'back', ''):
            return "back"

        try:
            choice_num = int(user_input.strip())
            if 1 <= choice_num <= len(test_files):
                return sorted(test_files)[choice_num - 1][0]
        except ValueError:
            pass

        return "invalid"

    # Group tests by category
    categories = {}
    for filename, description in test_files:
        category = categorize_test(filename)
        if category not in categories:
            categories[category] = []
        categories[category].append((filename, description))

    # Create choices with category headers and numbered tests
    choices = []
    test_index = 1
    test_map = {}

    for category in sorted(categories.keys()):
        if choices:  # Add separator between categories
            choices.append("")  # Empty line
        choices.append(f"📁 {category}:")
        choices.append("")  # Empty line after header

        for filename, description in sorted(categories[category]):
            choices.append(f"{test_index:2d}) {filename:<35} - {description}")
            test_map[test_index] = filename
            test_index += 1

    choices.append("")
    choices.append("← Back to test menu")

    # Filter choices for display
    display_choices = [c for c in choices if c and not c.startswith("📁") and c.strip() != ""]

    if DEBUG:
        print(f"DEBUG: display_choices count = {len(display_choices)}", flush=True)
        print(f"DEBUG: test_map = {test_map}", flush=True)

    choice = questionary.select(
        f"Choose test to run ({len(test_files)} available):",
        choices=display_choices,
        instruction="(↑/↓ to navigate • Enter to run • Esc to go back)",
    ).ask()

    if DEBUG:
        print(f"DEBUG: raw choice = {repr(choice)}", flush=True)

    if choice is None or "Back" in choice:
        return "back"

    if DEBUG:
        print(f"DEBUG: processing choice = {choice}", flush=True)

    # Extract test file
    try:
        if ")" in choice:
            file_num = int(choice.split(")")[0].strip())
            if file_num in test_map:
                result = test_map[file_num]
                if DEBUG:
                    print(f"DEBUG: returning filename = {result}", flush=True)
                return result
    except (ValueError, IndexError) as e:
        if DEBUG:
            print(f"DEBUG: extraction error = {e}", flush=True)
        pass

    return "invalid"

def is_interactive():
    """Check if running in an interactive terminal."""
    import os
    # Check both stdin and stdout for interactivity
    return os.isatty(0) or os.isatty(1)

def main():
    """Main entry point."""
    try:
        menu_type = sys.argv[1] if len(sys.argv) > 1 else "main"
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        # For main and test menus, allow if stdout is a TTY (even if stdin is not)
        if menu_type in ("main", "test-main", "development", "production", "utilities") and not is_interactive():
            # Not interactive, fall back to bash menu
            sys.exit(1)

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
        elif menu_type == "test-files-frontend":
            result = frontend_test_file_menu()
        else:
            result = "invalid"

        if result is None:
            # Fallback signal - Python CLI not available
            sys.exit(1)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
        else:
            print(result, flush=True)
    except KeyboardInterrupt:
        print("quit", flush=True)
        sys.exit(0)
    except Exception as e:
        # On any error, fall back to bash menu
        sys.exit(1)

if __name__ == "__main__":
    main()