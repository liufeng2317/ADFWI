# ADFWI Optimization Skill

This document defines the default optimization workflow for future ADFWI work.
It exists because the `bv1.2` internal cleanup expanded into more than one
hundred iterations. Future optimization should be bounded, testable, and
closed deliberately.

## Core Principle

One optimization task must have one explicit target, one validation plan, one
record, and one stop condition.

Do not start broad cleanup just because code can be improved. Start only when
there is a concrete reason:

- a user-facing workflow is unclear or broken;
- a measurable performance bottleneck exists;
- a numerical path needs validation or stabilization;
- a public API needs release preparation;
- a failing test or reproducibility issue exists;
- a small readability issue blocks understanding of an active module.

## Required Start Checklist

Before editing code, write down the optimization path in plain terms:

```text
Target:
Scope:
Non-scope:
Expected behavior change:
Validation:
Stop condition:
```

Definitions:

- `Target`: the exact module, script, notebook, or workflow being changed.
- `Scope`: files or behavior allowed to change.
- `Non-scope`: related but excluded work.
- `Expected behavior change`: use `none` for pure refactor/readability.
- `Validation`: the smallest reliable test or comparison.
- `Stop condition`: the point where the task is considered complete.

If these cannot be written clearly, do not start implementation yet.

## Optimization Classes

Use one class per task.

| Class | Examples | Required Validation |
| --- | --- | --- |
| Readability/API cleanup | docstrings, imports, constants, module ownership | import checks, lint/diff checks, focused unit tests |
| User workflow update | examples, notebooks, scripts, backend usage | script/notebook parity or runnable smoke test |
| Numerical FWI change | loss, transform order, receiver selection, gradient processing, propagator inputs | focused unit test plus numerical precision comparison |
| Performance change | batching, checkpointing, tensor movement, NPU/CPU path | before/after timing with same case and saved metrics |
| Validation/documentation | records, release notes, case summaries | link checks, command validity, no code tests unless needed |

Do not mix classes unless the task explicitly requires it. For example, do not
combine a readability cleanup with a performance rewrite.

## Scope Limits

Default limits for one optimization round:

- code changes should normally stay within one subsystem;
- avoid touching more than five source files unless the task is a mechanical
  API migration with a clear search pattern;
- avoid changing examples and core code in the same commit unless validating a
  new public API;
- do not refactor modules that are only placeholders or unused legacy code;
- do not continue from one small improvement into adjacent cleanups without a
  new start checklist.

Large migrations must be split into explicit phases:

1. audit and plan;
2. first narrow implementation;
3. validation;
4. documentation;
5. stop or open a new bounded task.

## Numerical Safety Rules

FWI core changes require numerical validation. Treat these as core paths:

- propagator inputs or outputs;
- loss and misfit formulas;
- waveform transform order or formulas;
- receiver/shot selection semantics;
- gradient processing;
- regularization formulas;
- optimizer update order;
- model constraints or parameter bounds.

Minimum numerical report:

```text
Case:
Device:
Dtype:
Seed:
Metric before:
Metric after:
Max absolute error:
Max relative error:
Tolerance:
Conclusion:
```

If a numerical difference is expected, document why it is acceptable. If no
reference exists, create one before changing the algorithm.

## Validation Ladder

Use the smallest validation that proves the task.

1. Import or syntax check for documentation/readability changes.
2. Focused unit test for helper-level behavior.
3. Smoke test for public workflow changes.
4. Script/notebook parity for example changes.
5. Reduced real-case gate for FWI loop changes.
6. Full real-case gate only for core numerical or release-critical changes.

Do not run expensive real-case tests just to validate comments, imports, or
documentation.

## Record Policy

Each optimization round should produce one concise record when it changes code,
behavior, validation assets, or release state.

Recommended record shape:

```markdown
# N - Short Title

## Target
## Change
## Validation
## Result
## Next Boundary
```

Rules:

- one record per meaningful optimization, not one record per tiny edit;
- summarize repeated experiments in one table instead of many files;
- link to saved artifacts instead of embedding long logs;
- if a task is documentation-only, say that no numerical validation was needed;
- if the next step is not necessary, write `No immediate follow-up`.

## Commit Policy

One commit should represent one coherent task.

Before committing:

- check `git status --short`;
- stage only files related to the task;
- do not stage unrelated notebooks or generated outputs;
- run `git diff --check`;
- run the planned validation;
- confirm the record file names the validation.

Commit message format:

```text
<verb> <bounded target>
```

Examples:

- `Clarify FWI iteration loss ownership`
- `Validate Marmousi2 forward script parity`
- `Document bv1.2 closeout policy`

## Stop Conditions

Stop the task when all are true:

- the original target is handled;
- planned validation passed or the blocker is documented;
- related docs/records are updated;
- code is committed and pushed when appropriate;
- the next step is either explicitly scoped or deferred.

Stop immediately and ask for direction if:

- the task requires changing more subsystems than planned;
- validation reveals a numerical drift not covered by the task;
- the change would require rewriting public examples broadly;
- the improvement is only aesthetic and not tied to a release/user need.

## Anti-Patterns

Avoid these patterns:

- "while here" cleanup;
- repeatedly renaming helper modules without a user-facing benefit;
- adding compatibility shims after a compatibility branch is already retained;
- replacing legacy numerical behavior without drift comparison;
- running full cases to justify non-numerical edits;
- creating many one-off records for the same investigation;
- treating an archive of old records as an active task queue;
- continuing optimization because there is still imperfect code.

## Recommended Default Workflow

For future ADFWI optimization requests:

1. Classify the request into one optimization class.
2. Write the start checklist.
3. Inspect only the target subsystem and immediate callers.
4. Make the smallest coherent change.
5. Run the validation ladder at the right level.
6. Write or update one record.
7. Commit and push.
8. State whether the task is closed or what the next bounded task is.

## Example

```text
Target:
examples/validation/marmousi2_acoustic_bv12/scripts

Scope:
Make script output match the manually verified notebooks.

Non-scope:
Do not migrate all examples.
Do not change FWI core internals.

Expected behavior change:
None; script should reproduce notebook results.

Validation:
Run forward and inversion scripts, compare saved arrays against notebook outputs.

Stop condition:
Script/notebook metrics match and one record documents the comparison.
```

This is the intended scale for a normal optimization round.
