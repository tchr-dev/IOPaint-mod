# Terminal Statuses

Understanding job status messages in the IOPaint system.

## Job Lifecycle States

### Non-Terminal States (In-Progress)

These indicate the job is still running and should continue polling:

| Status | Meaning |
|--------|---------|
| `queued` | Job accepted, not yet started |
| `pending` | Alias of queued |
| `running` | Model is executing |
| `processing` | Same as running |
| `in_progress` | Same as running |

**Rule**: Keep polling with exponential backoff.

### Terminal States (Complete)

These indicate the job has finished and will not change:

#### Success
| Status | Meaning |
|--------|---------|
| `succeeded` | Completed successfully |
| `completed` | Alias used by some APIs |

#### Failure
| Status | Meaning |
|--------|---------|
| `failed` | Job failed with error |
| `error` | Terminal error state |
| `cancelled` | User cancelled the job |
| `canceled` | Alternative spelling |

## Status Flow

```
queued → running → succeeded
     → cancelled (user action)
     → failed (error)
```

## Implementation Guidelines

1. **Polling**: Use exponential backoff starting at 1s, max 30s
2. **Timeouts**: Set reasonable timeouts (e.g., 5 minutes for image generation)
3. **Cancellation**: Support both user-initiated and server-side cancellation
4. **Retries**: Only retry on transient errors (network, rate limits)

## Related

- [TypeScript Resolver](typescript-resolver.md)
- [Code Style](code-style.md)
- [Architecture Overview](../architecture/overview.md)
