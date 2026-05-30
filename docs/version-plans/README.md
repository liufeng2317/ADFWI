# ADFWI Version Plans

This directory records ADFWI branch and release planning notes. The structure is
version-first, then topic-ordered, so later bv1.2 optimization work can keep a
clear trace without turning the folder into a flat list of notes.

Before starting a new optimization round, read
[ADFWI Optimization Skill](./optimization-skill.md). It defines the default
scope, validation, record, commit, and stop rules for future optimization work.

## Directory Layout

```text
docs/version-plans/
  README.md
  bv1.1/
    code-graph.md
  bv1.2/
    bv1.2-fwi/
      00-development-plan.md
      optimization-chain.md
    bv1.2-model/
      00-model-optimization-outline.md
      model-optimization-map.md
    bv1.2-survey/
      00-survey-optimization-outline.md
      survey-optimization-map.md
    bv1.2-propagator/
      00-propagator-optimization-outline.md
      propagator-optimization-map.md
    bv1.2-utils/
      00-utils-optimization-outline.md
      utils-optimization-map.md
    bv1.2-view/
      00-view-optimization-outline.md
      view-optimization-map.md
  bv1.2-example-sync/
    00-example-sync-outline.md
    example-sync-map.md
```

## Version Index

| Version | Status | Purpose | Documents |
| --- | --- | --- | --- |
| `bv1.1` | Stable baseline | Current synchronized baseline branch. Keep it stable and use it mainly for necessary fixes. | [Code graph](./bv1.1/code-graph.md) |
| `v1.1-freeze` | Frozen tag | Reproducible anchor before later `bv1.2` development. | [Code graph](./bv1.1/code-graph.md) |
| `bv1.2-fwi` | Archived closeout | FWI framework cleanup, backend/device unification, transforms, validation, and archive records. | [Index](./bv1.2/bv1.2-fwi/README.md) |
| `bv1.2-model` | Archived closeout | Bounded optimization and validation records for `ADFWI/model` ownership and parameter contracts. | [Index](./bv1.2/bv1.2-model/README.md) |
| `bv1.2-survey` | Archived closeout | Bounded optimization and validation records for survey geometry/data contracts. | [Index](./bv1.2/bv1.2-survey/README.md) |
| `bv1.2-propagator` | Archived closeout | Bounded non-kernel cleanup and validation records for propagator contracts. | [Index](./bv1.2/bv1.2-propagator/README.md) |
| `bv1.2-utils` | Archived closeout | Bounded optimization records for shared utility helpers, model data helpers, wavelets, mutes, metrics, and conversion contracts. | [Index](./bv1.2/bv1.2-utils/README.md) |
| `bv1.2-view` | Archived closeout | Bounded optimization records for plotting helpers and validation figure checks. | [Index](./bv1.2/bv1.2-view/README.md) |
| `bv1.2-example-sync` | Active branch | Synchronize public examples with the bv1.2 framework while preserving original notebook workflows. | [Index](./bv1.2-example-sync/README.md) |

## Naming Rules

- Put documents under a version directory, such as `bv1.2/bv1.2-fwi/` or
  `bv1.2/bv1.2-model/`.
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
   `bv1.2-model/00-model-optimization-outline.md`.
3. Link detailed design, audit, and comparison notes from the branch index.
4. Keep stable branches conservative. Put structural changes into the next
   development branch.
