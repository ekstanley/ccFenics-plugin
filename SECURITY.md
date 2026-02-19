# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.10.x  | Yes       |
| < 0.10  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Use [GitHub Security Advisories](https://github.com/ekstanley/ccFenics-plugin/security/advisories/new) to report privately
3. Include steps to reproduce, impact assessment, and any suggested fixes

We will acknowledge your report within 48 hours and work with you on a fix.

## Security Model

This server executes UFL expressions (Python syntax) within a Docker container. The defense layers are:

1. **Token blocklist** (`_check_forbidden()`) blocks dangerous tokens (`import`, `__`, `exec`, `os.`, `subprocess`, etc.)
2. **Empty `__builtins__`** in expression namespaces
3. **Docker isolation** (`--network none`, non-root user, `--rm`)

The `run_custom_code` tool intentionally bypasses the blocklist for full Python access. Docker container isolation is the only security boundary for that tool.
