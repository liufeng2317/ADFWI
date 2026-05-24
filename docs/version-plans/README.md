# ADFWI Version Plans

This directory records ADFWI branch and release planning notes. The structure is
version-first, then topic-ordered, so later bv1.2 optimization work can keep a
clear trace without turning the folder into a flat list of notes.

## Directory Layout

```text
docs/version-plans/
  README.md
  v1.1/
    code-graph.md
  bv1.2/
    00-development-plan.md
    01-device-backend-design.md
    02-misfit-backend-audit.md
    03-data-transform-pipeline.md
    04-lowpass-filter-comparison.md
```

## Version Index

| Version | Status | Purpose | Documents |
| --- | --- | --- | --- |
| `bv1.1` | Stable baseline | Current synchronized baseline branch. Keep it stable and use it mainly for necessary fixes. | [Code graph](./v1.1/code-graph.md) |
| `v1.1-freeze` | Frozen tag | Reproducible anchor before later `bv1.2` development. | [Code graph](./v1.1/code-graph.md) |
| `bv1.2` | Active development | Framework cleanup, backend/device unification, transforms, tests, and benchmark preparation. | [Index](./bv1.2/README.md) |

## Naming Rules

- Put documents under a version directory, such as `bv1.2/`.
- Use two-digit numeric prefixes inside active development folders to preserve
  reading order.
- Use concise topic names after the prefix, for example
  `03-data-transform-pipeline.md`.
- Keep historical records in place. When direction changes, add a dated section
  or a follow-up document instead of rewriting the earlier rationale.
- Keep generated notebooks, figures, and experiment outputs out of version-plan
  commits unless the file is itself a documentation artifact.

## Maintenance Rules

1. Keep the top-level `README.md` as the navigation entry point.
2. Keep one overview file per active branch, such as
   `bv1.2/00-development-plan.md`.
3. Link detailed design, audit, and comparison notes from the branch index.
4. Keep stable branches conservative. Put structural changes into the next
   development branch.
