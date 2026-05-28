# ADFWI Optimization Skill

This is the default rule for future ADFWI optimization work. Keep it practical:
each round should be bounded, validated, recorded briefly, and then stopped.

## 1. Start With A Boundary

Before editing, define only four things:

```text
Goal:
Scope:
Validation:
Stop:
```

- `Goal`: what problem is being solved.
- `Scope`: what files or workflow may change.
- `Validation`: the smallest test or comparison that proves the change.
- `Stop`: the condition for ending this round.

If these four lines are unclear, do not start coding yet.

## 2. Keep One Round Small

Default rule:

- one subsystem or one example workflow per round;
- one concise record per round, not many small records;
- one commit per coherent change;
- no "while here" cleanup.

Do not expand from a small improvement into adjacent restructuring. If another
issue is found, write it down as a possible next task and continue only if it is
needed for the current goal.

## 3. Use The Right Validation

Use the lightest validation that proves the change:

| Change type | Validation |
| --- | --- |
| docs/comments/import readability | link check, import check, or focused unit test |
| example or script workflow | run the script/notebook path or compare saved outputs |
| performance | before/after timing on the same case |
| FWI numerical path | unit test plus numerical precision comparison |

FWI numerical paths include loss formulas, transform order, receiver/shot
selection, gradient processing, regularization, optimizer order, propagator
inputs/outputs, and model constraints.

Do not run full Marmousi2 or other heavy cases for pure documentation,
formatting, import, or comment changes.

## 4. Record Only What Matters

A record should answer:

```text
What changed?
How was it validated?
What is the result?
Is there a next bounded task?
```

Avoid creating many records for one investigation. Summarize repeated tests in
one table or one paragraph.

## 5. Stop Aggressively

Stop when the original goal is handled and validation has passed. Do not keep
optimizing because the code is still imperfect.

Stop and ask for direction if:

- the task starts touching unrelated subsystems;
- a numerical difference appears outside the planned validation;
- the change requires broad example migration;
- the next step is only aesthetic cleanup.

## 6. Default Workflow

1. Define `Goal / Scope / Validation / Stop`.
2. Inspect only the target area and direct callers.
3. Make the smallest coherent change.
4. Run the planned validation.
5. Update one record if the change is meaningful.
6. Commit and push when appropriate.
7. State that the task is closed, or name one next bounded task.

## Anti-Patterns

- broad cleanup without a user or release need;
- repeated module reshuffling;
- compatibility shims when an old compatibility branch is already retained;
- replacing legacy numerical behavior without drift comparison;
- using full-case tests to justify non-numerical edits;
- treating archived records as an active task queue.
