# Propagator Performance Test Guidelines

This file defines which tests to run during future performance work. The goal
is to avoid rerunning the full baseline matrix for every small change.

## 1. Core Baselines

| Role | Case | When to run |
| --- | --- | --- |
| default development gate | `3 shots x 3 iterations x checkpoint_segments=10` | every meaningful code change |
| speed upper-bound reference | `3 shots x 3 iterations x checkpoint_segments=1` | when a change claims speed improvement |
| real-case promotion gate | `40 shots x 10 iterations x checkpoint_segments=10` | before accepting a phase-level optimization |

`checkpoint_segments=5` is not a core baseline for now. The measured Marmousi2
matrix shows it is slower than checkpoint=10 and uses more memory.

## 2. Required Metrics

Every performance comparison must report:

| Metric | Purpose |
| --- | --- |
| steady seconds/iteration | avoid first-iteration setup noise |
| peak allocated memory | guard against hidden memory inflation |
| loss trajectory finite | catch broken FWI loops |
| raw/processed gradient finite | protect autograd behavior |

When the propagator output or gradient path changes, numerical parity must also
be checked against the production path.

## 3. Minimal Test Selection

| Optimization stage | Required tests |
| --- | --- |
| early prototype | `3 shot x 3 iter x ckpt=10` |
| promising speed change | `3 shot x 3 iter x ckpt=10` and `ckpt=1` |
| memory-sensitive change | add `40 shot x 3 iter x ckpt=10` |
| phase closeout | `40 shot x 10 iter x ckpt=10` |

Do not run the full `1/3/40 shot x 3/10 iter x checkpoint 1/5/10` matrix unless
the optimization changes checkpoint/storage behavior itself.

## 4. Decision Rule

A change is worth continuing only if it satisfies at least one of these:

| Condition | Meaning |
| --- | --- |
| faster than checkpoint=10 with similar memory | practical production candidate |
| close to checkpoint=1 speed with controlled memory growth | strong custom-operator candidate |
| no speed gain but large memory reduction | useful storage-policy candidate |

If a change is slower and uses more memory than checkpoint=10, stop that path.
