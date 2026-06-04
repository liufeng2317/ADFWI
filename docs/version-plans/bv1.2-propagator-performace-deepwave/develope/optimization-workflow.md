# Optimization Workflow

Use this workflow for each future propagator-performance update. The purpose is
to keep the work focused and prevent open-ended optimization loops.

## 1. Draw The Overall Target

Start from the current top-to-bottom optimization map. State which branch of
the map this round belongs to, such as custom acoustic operator, storage policy,
checkpoint behavior, or validation infrastructure.

## 2. Select The Current Focus

Pick one concrete focus for the round and state the boundary. Examples:

| Focus | Boundary |
| --- | --- |
| forward pressure parity | no production kernel changes |
| raw `vp.grad` parity | tiny/reduced cases before full FWI |
| storage policy | compare speed and peak memory together |
| production integration | opt-in only until full validation passes |

Also state what is intentionally out of scope.

## 3. Implement The Update

Keep the update small enough that its effect can be explained. Prototype code
must stay outside the default production path until numerical and FWI gates
pass.

## 4. Run Layered Tests

Choose tests from `../optimization-test-guidelines.md`:

| Stage | Default test |
| --- | --- |
| early prototype | `3 shot x 3 iter x checkpoint=10` or smaller parity smoke |
| speed claim | add `3 shot x 3 iter x checkpoint=1` |
| memory-sensitive change | add `40 shot x 3 iter x checkpoint=10` |
| phase closeout | `40 shot x 10 iter x checkpoint=10` |

If the output or gradient path changes, include output/loss/gradient parity.

Testing-framework work is capped at three gate definitions for this branch:

| Gate | Status |
| --- | --- |
| baseline/checkpoint and Phase B parity gates | complete |
| short-loop `3 shot x 3 iter` gate | current |
| full `40 shot x 10 iter` promotion gate | only after a concrete algorithmic gain |

After the short-loop gate is recorded, further work must focus on algorithm or
operator implementation. Do not add new gate categories unless a clear bug makes
one of these gates invalid.

## 5. Summarize The Result

Each round summary must answer:

| Question | Required answer |
| --- | --- |
| What changed? | file/function-level change |
| What improved? | timing and/or memory result |
| What stayed equivalent? | output/loss/gradient parity result |
| Is it worth keeping? | keep, revise, or stop |

## 6. Choose The Next Step

End with one next recommendation and its boundary. Do not start a new direction
without saying whether the current direction continues, pauses, or stops.
