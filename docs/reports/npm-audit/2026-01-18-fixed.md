# NPM Audit Fix Report

**Date:** 2026-01-18

## Summary

All npm security vulnerabilities in the web_app frontend have been successfully resolved.

## Before Fix

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 8 |
| Moderate | 7 |
| Low | 1 |
| **Total** | **18** |

## After Fix

**0 vulnerabilities found**

## Commands Executed

```bash
cd web_app
npm install --package-lock-only
npm audit fix
npm audit fix --force  # For esbuild/vite update
npm run lint
npm run build
npm run test
```

## Package Updates

| Package | Before | After |
|---------|--------|-------|
| axios | 1.6.2 | 1.13.2 |
| vite | 5.0.0 | 7.3.1 |
| vitest | 1.6.0 | 4.0.17 |

## Additional Fix

Fixed TypeScript error in `src/lib/states.ts:711`:
- Added missing `rendersCountBeforeInpaint` property to `AppState` type

## Verification

- Lint: ✓ Passed
- Build: ✓ Passed
- Tests: ✓ Passed (5 tests)

## Files Modified

- `web_app/package-lock.json` - Regenerated
- `web_app/src/lib/states.ts` - TypeScript fix
