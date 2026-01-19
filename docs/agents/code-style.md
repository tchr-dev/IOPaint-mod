# Code Style Guidelines

Code style conventions extracted from `AGENTS.md` for quick reference.

## Python (Backend)

### Imports
Standard library → third-party → local (alphabetical)

```python
import os
from typing import List, Optional
import cv2
import torch
from loguru import logger
from iopaint.schema import InpaintRequest
```

### Types
Use `typing` module type hints; Pydantic for data models

```python
def forward(self, image, mask, config: InpaintRequest) -> np.ndarray:
    """Input and output images have same size [H, W, C] RGB."""
    ...
```

### Naming
- `snake_case` for functions/variables
- `PascalCase` for classes

```python
class LaMa(InpaintModel):
    name = "lama"
    LAMA_MODEL_URL = "https://..."
    def forward(self, image, mask): ...
```

### Error Handling
Use specific exceptions; log with `loguru`

```python
try:
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200: ...
except Exception as e:
    logger.debug(f"Failed to fetch: {e}")
```

### Abstract Methods
Use `@abc.abstractmethod` for required implementations

---

## TypeScript/React (Frontend)

### Imports
External libraries first, then local with `@/` alias

```typescript
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
```

### Naming
- `camelCase` for variables/functions
- `PascalCase` for components

```typescript
const [isOpen, setIsOpen] = useState(false)
export function SettingsDialog() { ... }
```

### Components
Use `forwardRef` for components accepting ref

```typescript
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, ...props }, ref) => { ... }
)
Button.displayName = "Button"
export { Button }
```

### Variants
Use `class-variance-authority (cva)` for component variants

```typescript
const buttonVariants = cva("base-classes", {
  variants: { variant: { default: "...", ghost: "..." } }
})
```

### className Merging
Always use `cn()` utility (tailwind-merge + clsx)

```typescript
<Comp className={cn(buttonVariants({ size }), "additional-class")} />
```

### Type Safety
Enable `strict: true` in tsconfig; prefer proper types over `any`

---

## Documentation

### File Naming
- Use **lowercase kebab-case**: `ui.md`
- No spaces: Use hyphens instead
- Numbered when sequential: `001-topic.md`

### Frontmatter
Use YAML frontmatter for metadata:

```yaml
---
status: backlog | active | completed
priority: high | medium | low
created: YYYY-MM-DD
---
```

### Links
Use relative links within the docs folder:
```markdown
[Architecture Overview](../architecture/overview.md)
```
