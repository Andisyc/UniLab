# Rapid DAgger Diagnostic Benchmark — Evaluation Entry Point

This is the single entry point for grading a submission to the rapid DAgger
diagnostic benchmark. It is evaluator-only material and must not be copied into the
frozen question repository.

## Invocation

The human should provide either:

1. a submission directory containing the files described below; or
2. the candidate's final answer directly in the evaluation request.

Recommended request:

```text
请读取 /Users/chengyuxuan/ArtiIntComVis/UniLab/EVALUATE.md，
评估 /absolute/path/to/submission-directory。
只输出评分报告，不修改任何文件。
```

## Authoritative Files

Read these files in this order:

1. Question: `/Users/chengyuxuan/Downloads/UniLab/TASK.md`
2. Candidate answer: `<submission-directory>/submission.md`, or the answer supplied
   directly by the human
3. Process metadata, when present: `<submission-directory>/run_meta.json`
4. Transcript, when present: `<submission-directory>/transcript.md`
5. Rubric and reference answer:
   `/Users/chengyuxuan/ArtiIntComVis/UniLab/SCORING.md`
6. Answer oracle only when clarification is needed:
   `/Users/chengyuxuan/ArtiIntComVis/UniLab/note/distill_native_corruption/ownerfix_r4_resolution.md`

Do not inspect unrelated source files merely to rediscover the known answer. Do not
modify either repository or the submission directory.

## Submission Directory

The recommended layout is:

```text
submission-directory/
├── submission.md    # required: candidate's final answer
├── run_meta.json    # recommended: timing and declared process facts
└── transcript.md    # strongly recommended: conversation and tool events
```

Minimal `run_meta.json` schema:

```json
{
  "started_at": "2026-08-14T14:00:00+08:00",
  "finished_at": "2026-08-14T14:12:30+08:00",
  "tool_calls": 11,
  "internet_used": false,
  "subagents_used": false,
  "repository_modified": false,
  "transcript": "transcript.md"
}
```

Additional usage or harness fields may be retained. Do not reject a submission merely
because optional fields are absent.

## Process-Evidence Policy

Use the following evidence precedence:

```text
verifiable transcript or harness record
> platform-generated metadata
> candidate/human self-report
> no evidence
```

- Treat a process fact as **verified** only when the transcript, harness, or
  platform-generated record supports it.
- Treat an uncorroborated `run_meta.json` value as **self-reported**, not verified.
- Treat a conflict between metadata and the transcript in favor of the transcript and
  report the conflict.
- If process evidence is absent, mark the affected field **unverified**. Do not invent
  compliance or a violation.
- Content can still be scored when process evidence is missing, but unverified
  process-compliance points cannot receive full credit.

### What Counts as a Tool Call

Count each agent-initiated external tool action that reads, searches, executes, tests,
edits, browses, or delegates. If an orchestration wrapper visibly launches several
nested tool actions, count the nested actions. A single shell-tool invocation counts
as one tool call even if its command contains several shell subcommands. Pure assistant
messages do not count.

Record at least these process fields:

| Field | How to verify |
| --- | --- |
| Wall-clock time | Difference between trustworthy start and finish timestamps |
| Tool calls | Count visible tool events using the rule above |
| Internet use | Browser/search/network tool events or network commands |
| Sub-agent use | Delegation/spawn events |
| Repository mutation | Patch/write commands or before/after repository status; creating the required `submission.md` is exempt |
| Final-answer length | Count the submitted final answer, not the transcript |

If the platform hides reasoning-token usage, do not estimate it. The benchmark uses
wall-clock time, tool calls, prohibited actions, and final-answer length as observable
efficiency measures.

## Evaluation Procedure

1. Confirm that the candidate answered the frozen `TASK.md` rather than a different
   task.
2. Read the candidate answer before consulting the reference answer.
3. Extract concise candidate claims about diagnosis, evidence chain, repair boundary,
   validation, and remaining uncertainty.
4. Audit process compliance using the evidence policy above.
5. Apply every category in `SCORING.md` and quote or precisely paraphrase supporting
   candidate evidence.
6. Apply the lowest relevant score cap and any length penalty.
7. Independently assign the collaboration potential score using `SCORING.md`. Ignore
   one-shot tool/time/process/length penalties and assume one later bounded live test
   plus ordinary human collaboration.
8. For the potential score, distinguish details that are locally addable, evidence
   that still requires validation, and conceptual errors that require replacing the
   diagnosis. Do not count future collaboration or live testing as completed evidence.
9. Give semantic-equivalence credit. Do not require historical filenames or exact
   patch reproduction.
10. Never infer that a keyword demonstrates causal understanding; require the answer
   to connect the proposed boundaries across DAgger iterations.

For the four process-compliance points in scoring category 5:

- **4/4**: all applicable limits are verified compliant;
- **2/4 maximum**: compliance is only self-reported or partially verified;
- **0/4**: process evidence is absent, materially contradictory, or verifies a
  prohibited action.

Other content-related points in category 5 remain independently available.

The required creation of `submission.md` is not a prohibited repository mutation.
Any source, configuration, test, task, or other existing-file write remains a
violation.

## Required Output

Return exactly one Markdown scoring report with this structure:

```markdown
# Evaluation Report

## Result
- One-shot score: XX/100
- One-shot ability band: ...
- Collaboration potential score: YY/100
- Potential ability band: ...
- Process status: verified / partially verified / unverified / violated
- Grading confidence: high / medium / low

## Process Audit
| Item | Observed | Evidence status | Decision |
| --- | --- | --- | --- |
| Wall-clock time | ... | ... | ... |
| Tool calls | ... | ... | ... |
| Internet use | ... | ... | ... |
| Sub-agents | ... | ... | ... |
| Repository mutation | ... | ... | ... |
| Answer length | ... | ... | ... |

## One-Shot Rubric
| Category | Score | Maximum | Evidence and reasoning |
| --- | ---: | ---: | --- |
| Moving-victim reasoning | ... | 20 | ... |
| Owner-boundary diagnosis | ... | 30 | ... |
| Coherent repair class | ... | 25 | ... |
| Validation plan | ... | 15 | ... |
| Efficiency and epistemic rigor | ... | 10 | ... |

## Collaboration Potential Rubric
| Category | Score | Maximum | Evidence and reasoning |
| --- | ---: | ---: | --- |
| Moving-victim reasoning potential | ... | 20 | ... |
| Owner-boundary convergence potential | ... | 30 | ... |
| Repair-trajectory potential | ... | 25 | ... |
| Validation and live-test leverage | ... | 15 | ... |
| Collaboration and epistemic readiness | ... | 10 | ... |

## Caps and Penalties
- One-shot applied: ...
- Potential applied: ...
- Reason: ...

## Main Strength
...

## Main Gap
...

## Potential Assessment
Explain why the remaining gaps are locally repairable or require a conceptual rewrite,
and how one bounded live test plus human feedback would change confidence.

## Final Judgment
Two to four sentences explaining both the demonstrated one-shot diagnostic level and
the credible collaboration ceiling, while naming what remains unverified.
```

Do not write a replacement answer for the candidate. Do not reveal more of the hidden
reference answer than is necessary to justify awarded or missing points.
