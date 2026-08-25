# Rapid DAgger Diagnostic Benchmark — Private Scoring Guide

> Evaluator-only material. Do not copy this file or its answer oracle into the frozen
> question repository at `/Users/chengyuxuan/Downloads/UniLab`.

Use [`EVALUATE.md`](EVALUATE.md) as the single evaluation entry point. It defines the
submission layout, process-evidence policy, evaluation order, and required report
format. This file remains the authoritative rubric and reference answer.

## Benchmark Boundary

- Question repository: `/Users/chengyuxuan/Downloads/UniLab`
- Candidate-facing prompt: its root `TASK.md`
- Reference implementation and answer evidence: this latest UniLab repository
- Primary answer oracle:
  [`note/distill_native_corruption/ownerfix_r4_resolution.md`](note/distill_native_corruption/ownerfix_r4_resolution.md)

The benchmark measures rapid diagnosis, not exact historical-patch reproduction. Score
semantic equivalence and causal discipline rather than keyword overlap.

## Dual-Score Contract

Effective 2026-08-14, every evaluation reports two independent scores:

1. **One-shot submission score** grades only what the submitted answer and
   verifiable process evidence actually demonstrate under the frozen resource budget.
   It remains the benchmark-comparison score governed by the rubric, caps, and
   penalties below.
2. **Collaboration potential score** estimates the agent's likely diagnostic ceiling
   when tool-count/time/process limits are ignored and the agent is later given one
   bounded live test plus ordinary human collaboration. It evaluates whether the
   current causal model is directionally correct, whether missing pieces are local
   refinements rather than a conceptual rewrite, and whether the agent can use future
   evidence without overclaiming.

The two scores must never be averaged or substituted for one another. Potential does
not turn an absent invariant, test, or runtime result into completed evidence. State
both the current gap and why it appears locally repairable—or why it requires a new
diagnosis.

## Reference Answer

The exception sites were **moving victims**, not proven corruption writers. The
best-supported practical diagnosis is a long-lived persistent-process ownership
failure spanning repeated DAgger iterations and cold serialization boundaries.

The verified repair class was:

```text
Role/scenario artifact identity
-> exact aggregate source identity
-> dataset atomic save
-> checkpoint CPU-owned atomic save
-> DAgger learner checkpoint reload without optimizer state
```

Concretely, the repaired implementation:

1. made artifact metadata the owner truth for `workflow_scenario` and failed closed
   on caller/metadata conflicts;
2. recorded the exact aggregate source snapshot;
3. wrote datasets through a same-directory temporary file plus atomic replacement;
4. serialized CPU-owned checkpoint payloads through atomic replacement; and
5. carried student weights, but not optimizer state, across DAgger iteration
   checkpoints because optimizer state is bound to parameter/update lifecycle.

This repair completed an eight-iteration `persistent_async` DAgger run and produced a
loadable final checkpoint. It did **not** prove the exact PyTorch/CUDA/native first
invalid operation. An answer that preserves this uncertainty can receive full credit.

## One-Shot Scoring Rubric (100 points)

### 1. Moving-victim reasoning — 20 points

- **8**: Recognizes that failure locations/types move across runs or lifecycle stages.
- **6**: Explicitly distinguishes detector/victim, trigger, owner, and possible
  corrupter instead of equating the last traceback line with root cause.
- **6**: Does not claim that the exact first invalid native operation is known.

### 2. Owner-boundary diagnosis — 30 points

- **10**: Centers the persistent process and repeated cross-iteration lifecycle.
- **8**: Identifies role/scenario or aggregate artifact identity/provenance as an
  owner-controlled invariant.
- **7**: Identifies dataset/checkpoint serialization as a cold ownership boundary,
  including partial-write or live-device-state risks.
- **5**: Connects optimizer state to parameter identity and repeated learner reloads.

Equivalent causal decompositions receive the same credit. Merely listing suspicious
files without explaining how state crosses iterations receives at most half credit in
this category.

### 3. Coherent repair class — 25 points

- **6**: Metadata owner truth with fail-closed identity conflict handling.
- **4**: Exact aggregate-source provenance/snapshot.
- **5**: Same-directory temporary dataset write plus atomic replacement, or an
  equivalent no-partial-reader guarantee.
- **5**: CPU-owned atomic checkpoint serialization, or an equivalent cold-artifact
  ownership guarantee.
- **5**: Weights-only DAgger iteration reload/save, excluding optimizer state while
  preserving deployable student semantics.

Do not require the exact historical APIs or filenames. Award points for equivalent
invariants placed at their owning layers.

### 4. High-information validation plan — 15 points

- **5**: Focused tests for artifact metadata ownership, missing identity, and conflict
  rejection.
- **4**: Dataset/checkpoint atomic-write and CPU round-trip checks.
- **3**: A checkpoint test proving optimizer state is neither required nor persisted
  for the DAgger iteration boundary.
- **3**: Reserves one fresh eight-iteration live run and checkpoint load for later
  evaluator confirmation, not diagnosis.

### 5. Efficiency and epistemic rigor — 10 points

- **4**: Respects the 15-minute, 15-tool-call, read-only, and response-length limits.
- **3**: Uses focused source/test inspection rather than broad scans or speculative
  experiments.
- **3**: Separates repository evidence, inference, runtime confirmation, and remaining
  native uncertainty concisely.

## Collaboration Potential Rubric (100 points)

Use the same first four category maxima so the two scores remain interpretable, but
score future convergence rather than pretending missing work is already complete.

### 1. Moving-victim reasoning potential — 20 points

Score the current causal level. Full credit requires that later feedback can refine
the diagnosis without returning to traceback-site patching or claiming the exact
native first-invalid operation prematurely.

### 2. Owner-boundary convergence potential — 30 points

Score how close the current ownership model is to a coherent cross-iteration chain.
Give high credit when missing role/scenario, aggregate, checkpoint, or optimizer
details are local additions to an already correct persistent-lifecycle model. Give
substantially lower credit when human feedback would need to replace the proposed
root-cause class.

### 3. Repair-trajectory potential — 25 points

Score whether the proposed changes form a safe owner-layer trajectory that can absorb
human corrections. Missing exact APIs or one owner invariant may retain high
potential; workarounds, traceback patches, or repairs that violate preserved behavior
do not. Explicitly name which missing invariants still prevent full credit.

### 4. Validation and live-test leverage — 15 points

Assume one later bounded live test and normal human collaboration are available.
Score whether the agent has proposed high-information offline gates, can turn human
feedback into discriminating checks, and reserves the live run for confirmation.
Do not award full credit merely because a future live run is promised.

### 5. Collaboration and epistemic readiness — 10 points

- **4**: Exposes decision-relevant unknowns and can accept human corrections without
  defending an obsolete branch or silently changing the task.
- **3**: Separates current evidence, inference, proposed validation, and future live
  confirmation.
- **3**: Communicates a concise, inspectable next reasoning step rather than requiring
  blind retries or a complete restart.

For this potential score, ignore wall-clock time, tool-call count, internet/process
metadata, repository-mutation evidence, and response-length penalties. These remain
fully active for the one-shot score.

Potential is not a reward for optimism. A high score requires evidence that the
agent's present causal hierarchy makes the remaining gaps locally correctable.

## One-Shot Score Caps and Penalties

Apply the lowest relevant cap after assigning rubric points:

| Submission behavior | Maximum total |
| --- | ---: |
| Treats the final traceback site as the proven root cause | 35 |
| Proposes retries, swallowed exceptions, reduced scale/iterations, or legacy mode as the repair | 25 |
| Gives only generic advice such as restart, add logging, or inspect CUDA | 45 |
| Claims the exact native first-invalid operation was proven | 70 |
| Claims the repository is confirmed fixed without the evaluator live run | 60 |
| Produces code edits, runs GPU training, or exceeds 15 tool calls | 80 |

For a response over the length limit, subtract 5 points. A modest time overrun should
be recorded separately when the evaluator can measure it; a deliberate or substantial
overrun caps the score at 80.

## Collaboration Potential Caps

Apply these only to the collaboration potential score:

| Current reasoning behavior | Maximum potential |
| --- | ---: |
| Treats the final traceback site as the proven root cause | 40 |
| Relies on retries, reduced scale/iterations, swallowed exceptions, or legacy mode | 30 |
| Gives only generic restart, logging, or CUDA advice | 50 |
| Claims the exact native first-invalid operation was proven | 75 |
| Claims no later live or human evidence is needed | 70 |

Do not apply one-shot process or response-length caps to the collaboration potential
score.

## Interpretation

| Score | Interpretation |
| ---: | --- |
| 85–100 | Strong rapid diagnosis: unified lifecycle/ownership model, coherent repair, calibrated uncertainty |
| 65–84 | Useful diagnosis: finds checkpoint/optimizer or artifact boundary but misses part of the chain |
| 40–64 | Recognizes a Heisenbug or lifecycle issue but remains generic or traceback-led |
| 0–39 | Primarily patches the last exception site or relies on prohibited workarounds |

Use the same numerical bands for both scores, but label them explicitly as one-shot
and collaboration potential.

## Grading Procedure

1. Record wall-clock time, tool-call count, prohibited actions, and response length.
2. Score the five one-shot categories using quoted evidence from the candidate
   response.
3. Apply the one-shot caps and penalties.
4. Independently score collaboration potential under the stated future live-test and
   human-collaboration assumptions; do not import one-shot process penalties.
5. Apply the collaboration potential caps and state which gaps are local refinements,
   which require validation, and which would require a conceptual rewrite.
6. Record one short justification for each category and for both final uncertainty
   assessments.
7. For a one-shot tie, prefer fewer tool calls and less unsupported specificity. For a
   potential tie, prefer the clearer evidence-to-feedback loop and fewer required
   conceptual rewrites.

Do not award points merely for naming the known fix. The answer must explain why the
proposed boundaries form one cross-iteration ownership repair and what they still do
not prove.
