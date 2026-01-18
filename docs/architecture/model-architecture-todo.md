# Model Architecture Improvements - TODO List

This document tracks all planned improvements to the IOPaint model architecture. Each item includes priority, effort estimate, status, and implementation details.

## Overview

Based on the analysis in [model-architecture.md](model-architecture.md), this TODO list prioritizes fixes that provide the highest impact to users and maintainers.

---

## Priority Legend

| Priority | Description |
|----------|-------------|
| 🔴 P0 | Critical - Breaking issues requiring immediate attention |
| 🟠 P1 | High - Significant user-facing improvements |
| 🟡 P2 | Medium - Quality of life and maintainability |
| 🟢 P3 | Low - Nice to have, future consideration |

---

## Tasks

### 1. Lazy Model Imports

**Issue**: All model classes are imported at startup, loading heavy dependencies.

**Priority**: 🟠 P1  
**Effort**: Medium  
**Status**: Complete

#### Problem Details
Location: `iopaint/model/__init__.py:21-43`

All model classes are imported eagerly, loading torch, diffusers, and other heavy dependencies regardless of whether the model is used.

#### Solution
Use Python module-level `__getattr__` for lazy imports:

```python
# iopaint/model/__init__.py

# Remove eager imports
# from .lama import LaMa, AnimeLaMa
# from .ldm import LDM
# ... etc

_models_cache = {}

def __getattr__(name: str):
    if name == "models":
        if not _models_cache:
            from .lama import LaMa, AnimeLaMa
            from .ldm import LDM
            from .zits import ZITS
            from .mat import MAT
            from .fcf import FcF
            from .manga import Manga
            from .opencv2 import OpenCV2
            from .mi_gan import MIGAN
            from .sd import SD15, SD2, Anything4, RealisticVision14, SD
            from .sdxl import SDXL
            from .paint_by_example import PaintByExample
            from .instruct_pix2pix import InstructPix2Pix
            from .kandinsky import Kandinsky22
            from .power_paint.power_paint import PowerPaint
            from .anytext.anytext_model import AnyText
            from iopaint.openai_compat.model_adapter import OpenAICompatModel

            _models_cache.update({
                LaMa.name: LaMa,
                AnimeLaMa.name: AnimeLaMa,
                LDM.name: LDM,
                ZITS.name: ZITS,
                MAT.name: MAT,
                FcF.name: FcF,
                OpenCV2.name: OpenCV2,
                Manga.name: Manga,
                MIGAN.name: MIGAN,
                SD15.name: SD15,
                Anything4.name: Anything4,
                RealisticVision14.name: RealisticVision14,
                SD2.name: SD2,
                PaintByExample.name: PaintByExample,
                InstructPix2Pix.name: InstructPix2Pix,
                Kandinsky22.name: Kandinsky22,
                SDXL.name: SDXL,
                PowerPaint.name: PowerPaint,
                AnyText.name: AnyText,
                OpenAICompatModel.name: OpenAICompatModel,
            })
        return _models_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# For backwards compatibility
models = None  # Will be populated on first access
```

#### Sub-tasks
- [ ] Modify `iopaint/model/__init__.py` with lazy import pattern
- [ ] Update all imports to use `from iopaint.model import models` (already works)
- [ ] Add `_models_cache` for model instance caching if needed
- [ ] Add tests for lazy import behavior
- [ ] Update AGENTS.md documentation

#### Files to Modify
- `iopaint/model/__init__.py`

---

### 2. Generate AVAILABLE_MODELS Dynamically

**Issue**: Model names exist in two places (`const.py` and `model/__init__.py`), causing drift.

**Priority**: 🟡 P2  
**Effort**: Low  
**Status**: Complete

#### Problem Details
Location: `iopaint/const.py:25`, `iopaint/model/__init__.py`

```python
# const.py - manual list
AVAILABLE_MODELS = ["lama", "ldm", "zits", "mat", "fcf", "manga", "cv2", "migan"]

# model/__init__.py - separate list
models = {
    LaMa.name: LaMa,  # "lama"
    LDM.name: LDM,    # "ldm"
}
```

#### Solution
Remove manual list and generate from model registry:

```python
# iopaint/const.py

def get_available_models() -> List[str]:
    """Get list of all available inpaint model names."""
    from iopaint.model import models
    return [
        name for name, cls in models.items()
        if getattr(cls, "is_erase_model", False)
    ]

# Keep for backwards compatibility
AVAILABLE_MODELS = get_available_models()
```

#### Sub-tasks
- [ ] Modify `iopaint/const.py` to generate list dynamically
- [ ] Update any code using `AVAILABLE_MODELS` directly
- [ ] Add test to verify consistency between `const.py` and model registry

#### Files to Modify
- `iopaint/const.py`

---

### 3. Lightweight Model Type Detection

**Issue**: Full pipeline loading for single-file model type detection is expensive.

**Priority**: 🟠 P1  
**Effort**: Medium  
**Status**: Complete

#### Problem Details
Location: `iopaint/download.py:56-80`

Loading the entire pipeline just to determine model type:
```python
StableDiffusionInpaintPipeline.from_single_file(
    model_abs_path,
    load_safety_checker=False,
    num_in_channels=9,
    original_config_file=get_config_files()["v1"],
)
```

#### Solution Options

**Option A**: Filename-based heuristics
Check filename for "inpaint" or config patterns before loading.

**Option B**: Lightweight config reading
Parse only the needed config without loading weights.

**Option C**: Deferred detection
Only detect type when model is actually selected, not during scan.

#### Implementation (Option C - Recommended)

```python
# iopaint/download.py

def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    # ... existing code until res.append
    
    res.append(
        ModelInfo(
            name=it.name,
            path=model_abs_path,
            model_type=ModelType.UNKNOWN,  # Defer detection
            is_single_file_diffusers=True,
            _deferred_detection=True,  # Flag for lazy loading
        )
    )
    # ... rest of function
```

Then in `ModelInfo`:

```python
class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: ModelType = ModelType.UNKNOWN
    is_single_file_diffusers: bool = False
    _deferred_detection: bool = False
    
    def detect_type(self):
        """Lazy model type detection."""
        if self.model_type != ModelType.UNKNOWN:
            return self.model_type
        
        # Fast check first
        if "inpaint" in self.name.lower():
            self.model_type = ModelType.DIFFUSERS_SD_INPAINT
            return self.model_type
        
        # Only then load pipeline
        self.model_type = get_sd_model_type(self.path)
        return self.model_type
```

#### Sub-tasks
- [ ] Add `ModelType.UNKNOWN` to schema
- [ ] Modify `scan_single_file_diffusion_models` for deferred detection
- [ ] Add `detect_type()` method to `ModelInfo`
- [ ] Update model switching code to trigger detection
- [ ] Improve cache file format with version metadata

#### Files to Modify
- `iopaint/schema.py`
- `iopaint/download.py`
- `iopaint/model_manager.py`

---

### 4. Model Weight Sharing and Incremental Loading

**Issue**: Model weights are not shared between different instances.

**Priority**: 🟢 P3  
**Effort**: High  
**Status**: Pending

#### Problem Details
Location: `iopaint/model_manager.py:144-158`

```python
del self.model
torch_gc()
self.model = self.init_model(new_name, ...)
```

#### Solution
Implement model pooling with weight sharing for similar architectures:

```python
class ModelPool:
    """Pool of loaded models with weight sharing support."""
    
    def __init__(self):
        self._pool: Dict[str, LoadedModel] = {}
        self._shared_components: Dict[str, Any] = {}
    
    def get_or_load(self, model_name: str, device) -> "LoadedModel":
        if model_name in self._pool:
            return self._pool[model_name]
        
        model = self._load_model(model_name, device)
        self._pool[model_name] = model
        return model
    
    def preload_similar(self, model_name: str, device):
        """Preload models with shared components."""
        # Preload SD base if loading SD inpaint variant
        pass
```

#### Sub-tasks
- [ ] Design shared component API (VAE, text encoder, UNet)
- [ ] Implement `ModelPool` class
- [ ] Update `ModelManager` to use pool
- [ ] Add memory pressure handling for pool eviction
- [ ] Add tests for weight sharing correctness

#### Files to Create
- `iopaint/model/pool.py`

#### Files to Modify
- `iopaint/model_manager.py`
- `iopaint/model/sd.py`
- `iopaint/model/sdxl.py`

---

### 5. Device Compatibility via Class Attributes

**Issue**: MPS device handling uses fragile hardcoded list.

**Priority**: 🟡 P2  
**Effort**: Low  
**Status**: Complete

#### Problem Details
Location: `iopaint/const.py:14-22`, `iopaint/model_manager.py:150`

```python
MPS_UNSUPPORT_MODELS = ["lama", "ldm", "zits", "mat", "fcf", "cv2", "manga"]

# In switch_mps_device()
device = switch_mps_device(new_name, self.device)
```

#### Solution
Add `supported_devices` class attribute:

```python
# iopaint/model/base.py
from typing import List

class InpaintModel:
    name: str
    pad_mod: int
    is_erase_model: bool = False
    supported_devices: List[str] = ["cuda", "mps", "cpu"]  # Default all
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Validate device list format
        for device in cls.supported_devices:
            if device not in ["cuda", "mps", "cpu"]:
                raise ValueError(f"Invalid device: {device}")

# iopaint/model/lama.py
class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8
    is_erase_model = True
    supported_devices = ["cuda", "cpu"]  # No MPS
```

Then update `switch_mps_device`:

```python
# iopaint/helper.py
def switch_mps_device(model_name: str, device: torch.device) -> torch.device:
    if device.type != "mps":
        return device
    
    from iopaint.model import models
    model_cls = models.get(model_name)
    if model_cls and "mps" not in getattr(model_cls, "supported_devices", ["cuda", "mps", "cpu"]):
        logger.info(f"Model {model_name} doesn't support MPS, switching to CPU")
        return torch.device("cpu")
    
    return device
```

#### Sub-tasks
- [x] Add `supported_devices` to base class
- [x] Update all model classes with appropriate device lists
- [x] Remove `MPS_UNSUPPORT_MODELS` from const.py
- [x] Update `switch_mps_device` helper
- [ ] Add runtime device capability detection (future enhancement)

#### Files to Modify
- `iopaint/model/base.py`
- `iopaint/model/lama.py` (and all other models)
- `iopaint/const.py`
- `iopaint/helper.py`

---

### 6. Unified Error Handling in Scanners

**Issue**: Inconsistent error handling across model scanners.

**Priority**: 🟢 P3  
**Effort**: Medium  
**Status**: Pending

#### Problem Details
Location: `iopaint/download.py` multiple locations

Some errors are logged, others silently ignored.

#### Solution
Create exception hierarchy:

```python
# iopaint/download/exceptions.py

class ModelScanError(Exception):
    """Base exception for model scanning errors."""
    pass

class CorruptedModelError(ModelScanError):
    """Model file is corrupted or unreadable."""
    pass

class UnsupportedFormatError(ModelScanError):
    """Model format is not supported."""
    pass

class CacheReadError(ModelScanError):
    """Failed to read cache file."""
    pass

class ModelIncompatibleError(ModelScanError):
    """Model is incompatible with current system."""
    pass
```

Update scanners to use specific exceptions:

```python
def scan_diffusers_models() -> List[ModelInfo]:
    # ...
    for it in model_index_files:
        try:
            with open(it, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise CacheReadError(f"Failed to read {it}: {e}") from e
        
        if "_class_name" not in data:
            raise CorruptedModelError(f"Missing _class_name in {it}")
```

#### Sub-tasks
- [ ] Create exception module
- [ ] Update all scanner functions with specific exceptions
- [ ] Update `scan_models()` to collect and report all errors
- [ ] Add error reporting to API response
- [ ] Add frontend display of scan errors

#### Files to Create
- `iopaint/download/exceptions.py`

#### Files to Modify
- `iopaint/download.py`
- `iopaint/api.py`
- `web_app/src/components/Settings.tsx`

---

### 7. Cache Versioning and Invalidation

**Issue**: Cache files are never invalidated on model updates.

**Priority**: 🟢 P3  
**Effort**: Low  
**Status**: Pending

#### Problem Details
Location: `iopaint/download.py:120-126`

```python
if cache_file.exists():
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            model_type_cache = json.load(f)
    except:
        pass
```

#### Solution
Add version metadata to cache:

```python
# iopaint/download.py

CACHE_VERSION = 1

def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    cache_file = stable_diffusion_dir / "iopaint_cache.json"
    
    model_type_cache = {}
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check version
                if data.get("_version") == CACHE_VERSION:
                    model_type_cache = data.get("_models", {})
                else:
                    logger.info("Cache version mismatch, clearing")
                    model_type_cache = {}
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            model_type_cache = {}
    
    # ... process models
    
    # Write back with version
    if stable_diffusion_dir.exists():
        cache_data = {
            "_version": CACHE_VERSION,
            "_models": model_type_cache,
            "_updated": time.time(),
        }
        with open(cache_file, "w", encoding="utf-8") as fw:
            json.dump(cache_data, fw, indent=2, ensure_ascii=False)
```

Add TTL-based invalidation:

```python
import time
from datetime import timedelta

CACHE_TTL = timedelta(days=7)  # 1 week TTL

def is_cache_valid(cache_data) -> bool:
    """Check if cache is still valid based on TTL."""
    if "_updated" not in cache_data:
        return False
    updated = datetime.fromtimestamp(cache_data["_updated"])
    return datetime.now() - updated < CACHE_TTL
```

#### Sub-tasks
- [ ] Add `CACHE_VERSION` constant
- [ ] Update cache read/write logic
- [ ] Add TTL checking
- [ ] Add cache clear CLI command
- [ ] Document cache behavior

#### Files to Modify
- `iopaint/download.py`
- `iopaint/cli.py`

---

### 8. Refactor OpenAI-Compat as Standard Adapter

**Issue**: OpenAI-Compat is a special case that doesn't follow model patterns.

**Priority**: 🟢 P3  
**Effort**: Medium  
**Status**: Pending

#### Problem Details
Location: `iopaint/download.py:322-327`

```python
openai_model = ModelInfo(
    name=OPENAI_COMPAT_NAME,
    path="openai-api",
    model_type=ModelType.OPENAI_COMPAT,
)
```

#### Solution
Create proper adapter pattern:

```python
# iopaint/model/openai_compat/adapter.py

class OpenAICompatAdapter:
    """Adapter making OpenAI-compatible API follow model pattern."""
    
    name = "openai-compat"
    model_type = ModelType.OPENAI_COMPAT
    is_erase_model = False
    
    @staticmethod
    def is_downloaded() -> bool:
        # Always "available" - no local files needed
        return True
    
    @staticmethod
    def download():
        # No download needed
        pass
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAIClient(config)
```

Update model registry:

```python
# iopaint/model/__init__.py

# Import at module level (lazy loading still applies)
def __getattr__(name: str):
    if name == "models":
        # ... existing models ...
        
        # Add OpenAI adapter
        from iopaint.model.openai_compat.adapter import OpenAICompatAdapter
        _models_cache[OpenAICompatAdapter.name] = OpenAICompatAdapter
        
        return _models_cache
```

#### Sub-tasks
- [ ] Create `iopaint/model/openai_compat/` module
- [ ] Refactor `OpenAICompatModel` to adapter pattern
- [ ] Update `scan_models()` to use adapter
- [ ] Remove special case handling in download.py
- [ ] Ensure backward compatibility

#### Files to Create
- `iopaint/model/openai_compat/__init__.py`
- `iopaint/model/openai_compat/adapter.py`

#### Files to Modify
- `iopaint/model/__init__.py`
- `iopaint/download.py`
- `iopaint/openai_compat/model_adapter.py`

---

### 9. Unified Plugin Model Discovery

**Issue**: Plugin models are not integrated into main model discovery.

**Priority**: 🟢 P3  
**Effort**: Medium  
**Status**: Pending

#### Problem Details
Location: `iopaint/plugins/` - each plugin downloads its own models inline.

#### Solution
Extend `scan_models()` to include plugin models:

```python
# iopaint/download.py

def scan_plugin_models() -> List[ModelInfo]:
    """Scan models provided by plugins."""
    from iopaint import plugins
    
    res = []
    for plugin_name, plugin in plugins.items():
        if hasattr(plugin, "available_models"):
            for model_info in plugin.available_models:
                res.append(ModelInfo(
                    name=model_info["name"],
                    path=model_info["path"],
                    model_type=ModelType.PLUGIN,
                    is_plugin_model=True,
                    plugin_name=plugin_name,
                ))
    return res

def scan_models() -> List[ModelInfo]:
    # ... existing scans
    available_models.extend(scan_plugin_models())
    # ... rest
```

Plugin implementation:

```python
# iopaint/plugins/realesrgan.py

class RealESRGANPlugin:
    name = "realesrgan"
    
    @property
    def available_models(self) -> List[Dict]:
        return [
            {
                "name": "RealESRGAN_x4plus",
                "path": self._get_model_path("realesrgan_x4plus"),
            },
            # ...
        ]
```

#### Sub-tasks
- [ ] Define plugin model interface
- [ ] Update all plugins to implement model registry
- [ ] Extend `scan_models()` for plugins
- [ ] Update frontend to display plugin models
- [ ] Add plugin model selection to Settings

#### Files to Modify
- `iopaint/download.py`
- `iopaint/plugins/realesrgan.py`
- `iopaint/plugins/gfpgan_plugin.py`
- `iopaint/plugins/interactive_seg.py`
- `web_app/src/components/Settings.tsx`

---

### 10. Model Version Checking

**Issue**: No way to detect when model updates are available.

**Priority**: 🟢 P3  
**Effort**: Medium  
**Status**: Pending

#### Problem Details
Model URLs and MD5s are hardcoded without version metadata.

#### Solution
Add version metadata to model classes:

```python
# iopaint/model/lama.py

class LaMa(InpaintModel):
    name = "lama"
    is_erase_model = True
    
    # Version metadata
    VERSION = "1.0.0"
    VERSION_URL = "https://api.github.com/repos/Sanster/models/releases/latest"
    
    @classmethod
    def get_remote_version(cls) -> Optional[str]:
        """Fetch latest version from remote."""
        import requests
        try:
            resp = requests.get(cls.VERSION_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("tag_name", "").lstrip("v")
        except Exception:
            pass
        return None
    
    @classmethod
    def check_for_updates(cls) -> bool:
        """Check if newer version is available."""
        remote = cls.get_remote_version()
        if remote and remote != cls.VERSION:
            logger.info(f"New version available for {cls.name}: {remote}")
            return True
        return False
```

CLI command for updates:

```python
# iopaint/cli.py

@app.command()
def check_updates():
    """Check if newer model versions are available."""
    from iopaint.model import models
    
    for name, cls in models.items():
        if hasattr(cls, "check_for_updates"):
            try:
                if cls.check_for_updates():
                    print(f"Update available: {name}")
            except Exception as e:
                print(f"Error checking {name}: {e}")
```

#### Sub-tasks
- [ ] Add version metadata to base model class
- [ ] Implement remote version checking for all models
- [ ] Add CLI command for update checking
- [ ] Add update notification in Settings UI
- [ ] Add `--update-models` CLI flag

#### Files to Modify
- `iopaint/model/base.py`
- `iopaint/model/lama.py`
- `iopaint/cli.py`
- `web_app/src/components/Settings.tsx`

---

## Progress Tracking

### Completed

None yet - this is the initial version.

### Overall Status

| Metric | Count |
|--------|-------|
| Total Tasks | 10 |
| P0 Critical | 0 |
| P1 High | 2 |
| P2 Medium | 3 |
| P3 Low | 5 |

---

## Dependencies Between Tasks

```
Task 1 (Lazy Imports)     ──┬──> Task 2 (AVAILABLE_MODELS)
                           │
Task 3 (Type Detection)    ──┤
                           │
Task 4 (Weight Sharing)    ──┤──> Task 6 (Error Handling)
                           │
Task 5 (Device Support)    ──┘
                              
Task 8 (OpenAI Adapter)    ──> Task 9 (Plugin Models)
                              
Task 7 (Cache Versioning)  ──> Task 10 (Version Checking)
```

---

## Suggested Implementation Order

1. **Task 1: Lazy Imports** - Foundational change, enables other improvements
2. **Task 5: Device Support** - Low effort, high user impact
3. **Task 2: AVAILABLE_MODELS** - Quick win, prevents drift
4. **Task 3: Type Detection** - Performance improvement
5. **Task 4: Weight Sharing** - Complex, may be deferred
6. **Task 6: Error Handling** - Debugging improvement
7. **Task 7: Cache Versioning** - Reliability improvement
8. **Task 8: OpenAI Adapter** - Code cleanup
9. **Task 9: Plugin Models** - Feature integration
10. **Task 10: Version Checking** - Future feature

---

## Testing Requirements

Each task should include:

- [ ] Unit tests for new functionality
- [ ] Integration tests with existing model system
- [ ] Error case handling tests
- [ ] Performance benchmarks (before/after)
- [ ] Backward compatibility verification

---

## References

- [Model Architecture Overview](model-architecture.md)
- [Original Issue Analysis](link-to-issue-tracker)
- [Related PRs](link-to-prs)

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-18 | 1.0 | Initial TODO list created |
