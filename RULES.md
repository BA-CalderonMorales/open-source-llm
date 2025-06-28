# Repository Workflow Rules

These rules keep development consistent across the project. The document is intentionally brief so it can be referenced often.

## General Principles

- Follow Test-Driven Development. Write tests before production code and keep changes small.
- Prefer immutable data structures and avoid side effects.
- When looking for solutions, consult **context7** and the guidance in **MEMORY.md**. Do not copy text from MEMORY.md into this file.

## Local Workflow

Development spans Python and Rust. Set up a virtual environment and install Python dependencies with `pip install -r inference/requirements.txt`. Run `pytest` for Python tests.

Rust code lives under `mobile/` and `inference-re/`. Use `cargo test --manifest-path <crate>/Cargo.toml` to run tests and `cargo build --release --manifest-path <crate>/Cargo.toml` to build release artifacts.

Run all tests and builds locally before pushing changes.

## Commit Standards

Commits must use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Examples:

```
feat: add dark mode toggle
fix: handle null todo values
chore: update dependencies
```

## Pull Requests

Prefix PR titles to show intent:

- **Feature:** … → merge into `develop`
- **Bugfix:** … → merge into `develop`
- **Cleanup:** … → merge into `develop`
- **Pipeline:** … → merge into `develop`
- **Hotfix:** … → merge directly to `main`

Include a **Codex CI** section summarising `install`, `build`, `typecheck`, and `test` results.

After merging into `develop`, automatically open a PR that merges `develop` into `main` so changes can be tested against the main branch.

## Continuous Integration

Continuous integration runs Rust unit tests and builds on every pull request using `.github/workflows/rust.yml`. Python tests run with `pytest` when present.

If the linter reports issues, fix them incrementally without breaking existing functionality.
