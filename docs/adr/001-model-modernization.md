# ADR-001: Model Architecture Modernization

## Status
Accepted

## Date
2026-01-18

## Context
The IOPaint model system needed modernization to address several issues:
- Duplicate model entries in discovery mechanisms
- No unified approach for plugin model integration
- No memory optimization for shared model components
- Complex model switching with high memory overhead

## Decision

Implemented a comprehensive model architecture refactoring with the following key changes:

### 1. Unified Model Discovery

**Before:**
```python
# scan_models() added models manually
def scan_models():
    models = get_core_models()
    models += get_openai_models()  # Duplicate OpenAI entries
    return models
```

**After:**
```python
def scan_models():
    models = get_core_models()
    models += scan_plugin_models()  # Unified plugin integration
    return models
```

### 2. Plugin Model Integration

Added `available_models` property to `BasePlugin`:

```python
class BasePlugin:
    @property
    def available_models(self) -> List[ModelInfo]:
        """Override to expose plugin models in UI."""
        return []

class RealESRGANPlugin(BasePlugin):
    @property
    def available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(name="RealESRGAN_4x", type=ModelType.PLUGIN, plugin_name="realesrgan"),
            # ... more variants
        ]
```

### 3. Model Pool for Component Sharing

Created `ModelPool` for memory-efficient weight sharing:

```python
class ModelPool:
    def get_shared_components(self, model_type: Type[InpaintModel]) -> Optional[SharedComponents]:
        """Return cached components if available."""
        return self._pool.get(model_type)

    def release(self, model_type: Type[InpaintModel]) -> None:
        """Release components when model is no longer needed."""
        # Automatic cleanup via reference counting
```

### 4. Updated Model Types

Added `ModelType.PLUGIN` enum for plugin model classification:

```python
class ModelType(Enum):
    INPAINT = "inpaint"
    DIFFUSERS_SD = "diffusers_sd"
    DIFFUSERS_SDXL = "diffusers_sdxl"
    PLUGIN = "plugin"  # NEW
```

## Consequences

### Positive
- **Memory Reduction**: Shared VAE and text encoder components between compatible models
- **Simplified Discovery**: Single `scan_models()` function handles all model types
- **Plugin Integration**: Consistent model exposure across all plugins (15+ new models)
- **No Breaking Changes**: All existing APIs preserved

### Negative
- **Migration Complexity**: Required updating all plugin implementations
- **Testing Overhead**: Additional tests for ModelPool and plugin discovery

## Technical Details

### Files Modified
- `iopaint/model/pool.py` - NEW: ModelPool implementation
- `iopaint/schema.py` - Added ModelType.PLUGIN
- `iopaint/download.py` - Added scan_plugin_models()
- `iopaint/model_manager.py` - Integrated ModelPool
- All plugin files - Added available_models property

### Statistics
- **16 files modified**
- **666 insertions, 96 deletions**
- **38 model tests passing**
- **15+ plugin models now discoverable**

## Related ADRs
- None (first ADR)

## References
- Commit: `0166971` - "feat: complete P3 model architecture tasks"
- Architecture Documentation: `docs/architecture/models.md`
