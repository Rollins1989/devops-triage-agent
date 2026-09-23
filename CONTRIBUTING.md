# Contributing

Thanks for contributing to DevOps Triage Agent.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the tests before opening a pull request:

```bash
python -m pytest
```

For a local demo:

```bash
python -m main --scenario crashloop
```

Keep remediation behavior conservative and preserve the default dry-run safety rail.

## Pull requests

Please keep changes focused, include tests for behavior changes, and update the README when user-facing behavior changes.
