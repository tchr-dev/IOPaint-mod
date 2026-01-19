# NPM Audit History

Security audit reports for the web application dependencies.

## Latest Status

- **Date**: 2026-01-18
- **Vulnerabilities**: 18 total (1 low, 7 moderate, 8 high, 2 critical)
- **Status**: Fixed
- **Report**: [2026-01-18-fixed.md](2026-01-18-fixed.md)

## History

| Date | Vulnerabilities | Status | Report |
|------|----------------|--------|---------|
| 2026-01-18 | 18 (1L/7M/8H/2C) | Fixed | [Link](2026-01-18-fixed.md) |
| 2026-01-18 | 18 (1L/7M/8H/2C) | Initial | [Link](2026-01-18-initial.txt) |

## Audit Commands

```bash
cd web_app
npm audit              # Check for vulnerabilities
npm audit --json       # Get detailed JSON output
npm audit fix          # Apply automatic fixes
npm audit fix --force  # Apply breaking fixes
```

## Best Practices

1. Run `npm audit` regularly (weekly recommended)
2. Address high/critical vulnerabilities immediately
3. Review moderate vulnerabilities and fix when feasible
4. Document any accepted risks (vulnerabilities not fixed)

## Related

- [Security Policy](../../.github/SECURITY.md) - Security reporting guidelines
- [NPM Audit Documentation](https://docs.npmjs.com/cli/v8/commands/npm-audit) - Official NPM docs
