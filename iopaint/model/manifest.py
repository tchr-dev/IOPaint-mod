from typing import Optional, List
from iopaint.schema import ModelManifest
from iopaint.config import get_config, DEFAULT_MODELS

def get_manifest(name: str) -> Optional[ModelManifest]:
    config = get_config()
    for manifest in config.models:
        if manifest.name == name:
            return manifest
    
    # Fallback to defaults if not in config
    for manifest in DEFAULT_MODELS:
        if manifest.name == name:
            return manifest
            
    return None

def get_all_manifests() -> List[ModelManifest]:
    config = get_config()
    # Merge defaults with config models, avoiding duplicates
    manifests = {m.name: m for m in DEFAULT_MODELS}
    for m in config.models:
        manifests[m.name] = m
    return list(manifests.values())

# For backward compatibility if needed, but preferred to use get_manifest
def __getattr__(name: str):
    if name == "LAMA_MANIFEST":
        return get_manifest("lama")
    if name == "ANIME_LAMA_MANIFEST":
        return get_manifest("anime-lama")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
