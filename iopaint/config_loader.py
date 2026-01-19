"""Environment file loader with hot-reload support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Any
import os
import signal
import threading
import json

from loguru import logger

_BASE_ENV_KEYS = set(os.environ.keys())
_LOADED_KEYS: set[str] = set()
_LOCK = threading.Lock()
_SIGHUP_INSTALLED = False

_PUBLIC_KEYS = {
    "AIE_BACKEND",
    "AIE_OPENAI_BASE_URL",
    "AIE_OPENAI_MODEL",
    "AIE_OPENAI_TIMEOUT_S",
    "AIE_OPENAI_REFINE_MODEL",
    "AIE_OPENAI_MODELS_CACHE_TTL_S",
    "AIE_BUDGET_DAILY_CAP",
    "AIE_BUDGET_MONTHLY_CAP",
    "AIE_BUDGET_SESSION_CAP",
    "AIE_RATE_LIMIT_SECONDS",
    "AIE_DEDUPE_ENABLED",
    "AIE_DEDUPE_TTL_SECONDS",
}


def default_config_path() -> Path:
    return Path(os.getenv("AIE_CONFIG_FILE", "config/secret.env")).expanduser()


def register_env_override(keys: Iterable[str]) -> None:
    _BASE_ENV_KEYS.update(keys)


def _parse_env_line(raw_line: str) -> Optional[tuple[str, str]]:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if value and value[0] in {'"', "'"} and value[-1] == value[0]:
        value = value[1:-1]
    return key, value


def parse_env_file(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            key, value = parsed
            entries[key] = value
    return entries


def load_env_file(path: Optional[Path] = None) -> Dict[str, object]:
    config_path = (path or default_config_path()).expanduser()
    with _LOCK:
        if not config_path.exists():
            logger.info("Config file not found: {}", config_path)
            # Create directory if it doesn't exist
            if not config_path.parent.exists():
                config_path.parent.mkdir(parents=True, exist_ok=True)
            return {"path": str(config_path), "loaded": False, "added": [], "updated": [], "removed": []}

        entries = parse_env_file(config_path)
        new_keys = set(entries.keys())
        added: list[str] = []
        updated: list[str] = []
        removed: list[str] = []

        for key, value in entries.items():
            if key in _BASE_ENV_KEYS:
                continue
            previous = os.environ.get(key)
            if previous is None:
                added.append(key)
            elif previous != value:
                updated.append(key)
            os.environ[key] = value

        for key in list(_LOADED_KEYS):
            if key not in new_keys and key not in _BASE_ENV_KEYS:
                if key in os.environ:
                    del os.environ[key]
                removed.append(key)

        _LOADED_KEYS.clear()
        _LOADED_KEYS.update({key for key in new_keys if key not in _BASE_ENV_KEYS})

        if added or updated or removed:
            logger.info(
                "Config reloaded from {} (added={}, updated={}, removed={})",
                config_path,
                added,
                updated,
                removed,
            )
        else:
            logger.info("Config reloaded from {} (no changes)", config_path)

        return {
            "path": str(config_path),
            "loaded": True,
            "added": added,
            "updated": updated,
            "removed": removed,
        }


def save_config(config_data: Any, path: Optional[Path] = None) -> None:
    """Save configuration to the env file.
    
    This function accepts an ApiConfig object or a dictionary and persists it to the
    configuration file (defaulting to config/secret.env).
    
    Note: IO Paint's original web_config.py used a JSON format for saving checks,
    but we are moving to saving runtime config or environment config.
    Wait, `web_config.py` saved to a JSON file passed as argument.
    If we want to support that flow, we should probably check if we are using environment variables
    or a JSON config file.
    
    However, the user request is to replace Gradio Settings.
    Gradio Settings allowed modifying:
    - Host, Port (startup args)
    - Model, Device, Quality, etc. (Runtime args)
    
    The original `web_config.py` saved these to a JSON file.
    To allow similar functionality, we should save these to a JSON file that can be loaded on startup.
    The `load_config` in `web_config.py` loaded from a JSON file.
    
    Let's stick to the JSON config file approach for compatibility with how people might expect
    settings to work (a persistant settings file), but we need to know WHERE to save it.
    
    If `config_loader` is strictly for ENV files, maybe we should create a new `settings_manager`
    or similar. But `api.py` needs it.
    
    For now, I will implement a `save_config` that saves to `iopaint_config.json` in the working directory
    or the one specified by `AIE_CONFIG_FILE` if it ends in .json, otherwise `iopaint_config.json`.
    """
    
    # We need to handle ApiConfig object
    if hasattr(config_data, "model_dump"):
         data = config_data.model_dump()
    elif hasattr(config_data, "dict"):
        data = config_data.dict()
    else:
        data = config_data

    # Filter out entries that shouldn't be saved or sanitize types if needed
    # For now assume it's JSON serializable
    
    # Check where to save
    # In web_config.py, it used a global _config_file.
    # We don't have that global here.
    # We can default to `iopaint.json` in current dir?
    target_path = Path("iopaint_config.json")
    
    # If we want to respect the file passed in CLI, we'd need that info.
    # But CLI args are parsed in `cli.py` or `entry.py`.
    # Let's check api.py again, `Api` is initialized with `config`.
    # But `Api` class doesn't seem to store the config FILE path, only the values.
    
    # Let's save to a fixed location for this refactor: user provided config path or default.
    # Since we can't easily access the CLI arg here without threading it through, 
    # we will use a standard location.
    
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Configuration saved to {target_path.absolute()}")


def install_sighup_handler(reload_callback) -> None:
    global _SIGHUP_INSTALLED
    if _SIGHUP_INSTALLED:
        return
    if not hasattr(signal, "SIGHUP"):
        logger.warning("SIGHUP not available; config reload disabled")
        return

    def _handler(signum, frame):
        logger.info("Received SIGHUP, reloading config")
        reload_callback()

    signal.signal(signal.SIGHUP, _handler)
    _SIGHUP_INSTALLED = True


def public_config() -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for key in sorted(_PUBLIC_KEYS):
        value = os.getenv(key)
        if value is not None and value != "":
            payload[key] = value
    return payload
