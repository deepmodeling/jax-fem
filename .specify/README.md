# Spec Kit Bootstrap Status

This repository currently contains a **Spec Kit-compatible artifact bootstrap**,
not a fully initialized or executable GitHub Spec Kit installation.

## Available Now

- `.specify/memory/constitution.md`
- `specs/001-kaess-paper-reproduction/spec.md`
- draft `plan.md`, `tasks.md`, research, data model, contracts, and checklists
- manual structure, JSON-syntax, and cross-artifact review

## Not Installed

The `specify` CLI is not installed on this workstation. The repository therefore
does not yet contain the CLI-generated:

- `.specify/scripts/<variant>/`
- `.specify/templates/`
- Codex Spec Kit command/skill integration

Until those components are explicitly installed, `/speckit.*` or
`$speckit-*` commands MUST NOT be presented as runnable project commands.
The current plan and tasks are review drafts, not completed Spec Kit lifecycle
phases.

## Optional Full Initialization

Full initialization is a separate, user-approved tooling change because it
installs software and may add or update shared repository files. The official
shape for this Windows/WSL + Codex project would be based on:

```bash
uv tool install specify-cli
specify init --here \
  --integration codex \
  --integration-options="--skills" \
  --script sh
```

Do not run `specify init --here --force` without reviewing its changes. The
existing `specs/` artifacts must be preserved, and the project constitution must
be reconciled rather than silently replaced.

Official documentation: <https://github.github.com/spec-kit/>
