# ADFWI Version Plans

This directory records the development plan for each ADFWI branch or release.
It is intended to keep the stable version fixed while making future extensions
traceable and easy to discuss.

## Version Index

| Version | Status | Purpose |
| --- | --- | --- |
| `bv1.1` | Stable baseline | Current synchronized baseline branch. Keep it stable and use it mainly for necessary fixes. Code graph recorded in `v1.1-code-graph.md`. |
| `v1.1-freeze` | Frozen tag | Reproducible anchor before later `bv1.2` development. |
| `bv1.2` | Active development | Framework cleanup, extensibility improvements, tests, documentation, and benchmark preparation. |

## Maintenance Rules

1. Keep one plan file per major branch or release.
2. Use plan files to record goals, non-goals, implementation phases, and acceptance criteria.
3. Avoid editing old plans to rewrite history. Add notes or follow-up plans when direction changes.
4. Keep stable branches conservative. Put structural changes into the next development branch.

## Current Plans

- [v1.1 Code Graph](./v1.1-code-graph.md)
- [bv1.2 Development Plan](./bv1.2-development-plan.md)
- [bv1.2 Device Backend Interface Design](./bv1.2-device-backend-design.md)
