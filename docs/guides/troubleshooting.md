# Troubleshooting Guide

Common issues and their solutions.

## Build Failures

### TypeScript Compilation Errors

**Problem**: `Module '"react"' has no exported member 'act'`

**Solution**: The `act` function is in `react/test-utils`, not directly from `react`:
```typescript
import { act } from "react/test-utils"
```

**Problem**: `globalThis` has no index signature

**Solution**: Add type declaration:
```typescript
declare global {
  interface GlobalThis {
    IS_REACT_ACT_ENVIRONMENT?: boolean
  }
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true
```

### Budget Status Type Errors

**Problem**: `status: string` not assignable to `"ok" | "warning" | "blocked"`

**Solution**: Use explicit type assertions:
```typescript
const mockBudgetStatus: BudgetStatus = {
  status: "ok" as const,
  // ...
}
```

## Runtime Errors

### Empty Models List

**Problem**: Models list is empty in the UI

**Solution**:
1. Verify `AIE_OPENAI_API_KEY` and base URL are set
2. Check backend logs for OpenAI API errors
3. Ensure the provider is correctly configured in settings

### Requests Failing

**Problem**: API requests fail with errors

**Solution**:
1. Check backend logs for OpenAI error details
2. Verify API key has correct permissions
3. Confirm base URL is accessible

### Budget Blocked

**Problem**: UI shows `blocked_budget`

**Solution**:
1. Check budget environment variables (`AIE_BUDGET_DAILY`, etc.)
2. Increase budget caps or disable them temporarily
3. Review budget configuration in `config/secret.env`

## Performance Issues

### Slow Image Generation

- Ensure GPU is being used (check backend logs)
- Reduce image size for testing
- Check model-specific optimizations

### High Memory Usage

- Restart the backend periodically
- Consider using CPU inference for large batches
- Check for memory leaks in custom plugins

## See Also

- [Development Guide](development.md)
- [Architecture Overview](../architecture/overview.md)
- [Agents Guide](../agents/README.md)
