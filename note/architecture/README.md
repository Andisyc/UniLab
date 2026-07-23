# UniLab Architecture Atlas

This folder stores human-readable architecture maps for the UniLab runtime and repository structure.

The maps use one shared rule:

```text
same Code Block ID
  -> same concept name
  -> same color
  -> same code location
```

## Current Maps

- `runtime/01_unilab_runtime_atlas.data.json`: formal UniLab runtime reading cards.
- `architecture/02_g1_distillation_method_to_code.data.json`: distillation design-point-to-owner-code reading atlas.
- `concept/03_g1_multiteacher_distillation_method.data.json`: active stand/walk multi-teacher distillation Concept Figure.
- `concept/04_amp_walk_async_method.data.json`: active Phase 1 walk-only AMP Concept Figure.
- `auxiliary/atlas_app/`: helper viewer, local server, static renderer, checks, and JS dependencies.

## Folder Contract

```text
note/architecture/
  concept/        method Concept Figure and design-point map
  architecture/   repo/file/block mind map
  runtime/        module interface contract map
  auxiliary/      helper app files kept out of the map folders
  index.html      clean entry page
```

## Map Lifecycle

Temporary maps are allowed, including in the main entry page, while they are
actively guiding a change. After the change lands, a temporary map must be
either deleted or integrated into one of the active maps.

The main entry contains exactly four active maps in numbered order. Superseded
supporting maps stay in Git history rather than the active Atlas surface.

## VSCode Workflow

```bash
cd note/architecture
node auxiliary/atlas_app/serve_architecture.mjs
```

Open one of these URLs on the right side of VSCode:

```text
http://127.0.0.1:8766/
http://127.0.0.1:8766/auxiliary/atlas_app/architecture_atlas.html?data=../../runtime/01_unilab_runtime_atlas.data.json
http://127.0.0.1:8766/auxiliary/atlas_app/architecture_atlas.html?data=../../architecture/02_g1_distillation_method_to_code.data.json
http://127.0.0.1:8766/auxiliary/atlas_app/architecture_atlas.html?data=../../concept/03_g1_multiteacher_distillation_method.data.json
http://127.0.0.1:8766/auxiliary/atlas_app/architecture_atlas.html?data=../../concept/04_amp_walk_async_method.data.json
```

Open the matching `*.data.json` on the left. Saving the JSON refreshes the graph
automatically. The atlas page also polls the current JSON file, so it still
updates even if an older server process is running.

Viewer controls:

- The built-in JSON editor is hidden by default so the graph uses the full page.
- `Show Editor` opens the built-in JSON editor when quick in-browser edits are useful.
- `+`, `-`, `Fit Width`, and `100%` control graph zoom.
- `Fit Width` also restores auto-fit behavior after manual zooming.
- Drag the graph canvas to pan. Trackpad horizontal scroll also works on large maps.
- `Ctrl`/`Cmd` + wheel zooms around the pointer.
- Blue source rows in Runtime and Method-to-Code open the exact repository file and line through the local server.

## Human / LLM Governance

The distillation atlas is the visual half of the governed document system in
`note/distillation/`. The control loop is:

```text
human intent
  -> Concept Figure and active contract
  -> Method-to-Code owner map
  -> runtime branch and current checklist
  -> code change and focused evidence
  -> contract/evidence update before the next decision
```

Start at `../distillation/README.md`. Active method commitments live only under
`../distillation/contracts/active/`; ideas not yet accepted belong under
`../distillation/plans/`. Runtime findings belong in `evidence/` or `logs/`, not
inside the Concept Figure.

The AMP Concept Figure is governed independently by `note/amp/contracts/`.
Its active v003 method semantics are Phase 1 fixed-forward walking with task-
owned minimum physical viability, including the source-parity symmetric full-
body self-collision cost, while AMP alone owns human-like posture/style. The
four Concept Figure blocks are unchanged. The backend sensor-history interface
and AMP-only reward connector are now implemented and evidenced in
`../amp/evidence/2026-07-23-self-collision-steps1-2.md`. They do not add a new
method block; a broader Method-to-Code or Runtime Atlas expansion remains
unnecessary for this isolated owner path.

## Validation

```bash
cd note/architecture/auxiliary/atlas_app
npm run check
```

This validates the viewer, map schemas, contract/design IDs, owner source paths,
runtime branches, and the local source-navigation route.

## HTML Design Contract

The current atlas uses one reusable HTML viewer:

```text
auxiliary/atlas_app/architecture_atlas.html
  -> loads one *.data.json through ?data=...
  -> chooses renderer by data.layout
  -> draws rough SVG cards with shared colors, IDs, zoom, pan, editor, and live reload
```

The active and supporting pages are data variants, not separate applications:

- UniLab Runtime uses `layout: "repository_reading_atlas"`.
  - Source: `runtime/01_unilab_runtime_atlas.data.json`.
  - Purpose: runtime-ordered owner cards from CLI through playback.
  - Main schema: `systems[].modules[]`, `runtimeOrder[]`, internal B-routes.

- Distillation Method-to-Code uses `layout: "repository_reading_atlas"`.
  - Source: `architecture/02_g1_distillation_method_to_code.data.json`.
  - Purpose: runtime-ordered owner cards with navigable source locations.
  - Main schema: `modules[]`, `runtimeOrder[]`, `supportOrder[]`, internal routes.

- Distillation Concept Figure uses `layout: "method_figure"`.
  - Source: `concept/03_g1_multiteacher_distillation_method.data.json`.
  - Purpose: human-controlled method intent and causal closure.
  - Main schema: `nodes[]`, `edges[]`, stable design/contract mappings.
- AMP Walk Concept Figure uses `layout: "method_figure"`.
  - Source: `concept/04_amp_walk_async_method.data.json`.
  - Purpose: human-controlled Phase 1 AMP method intent and causal closure.
  - Main schema: four design-point nodes, four causal interactions, and stable
    `AMP-WALK` contract mappings.

## Reuse Contract

For another LLM Agent: this atlas is meant to be reused by copying the whole
folder, not by copying a single HTML file. The folder is a small self-contained
viewer plus JSON map sources.

Copy this directory into the new project:

```text
note/architecture/
```

The copied folder should keep this shape:

```text
note/architecture/
  index.html
  README.md
  architecture/
    *.data.json
  runtime/
    *.data.json
  concept/
    *.data.json
  auxiliary/atlas_app/
    architecture_atlas.html
    serve_architecture.mjs
    render_rough_arch_svg.mjs
    package.json
    package-lock.json
```

In the new project, start the viewer from the copied folder:

```bash
cd note/architecture
npm --prefix auxiliary/atlas_app install
node auxiliary/atlas_app/serve_architecture.mjs
```

Then open:

```text
http://127.0.0.1:8766/
```

To reuse the current HTML page for a specific map, create or edit a
`*.data.json` file and open:

```text
http://127.0.0.1:8766/auxiliary/atlas_app/architecture_atlas.html?data=../../PATH/TO/MAP.data.json
```

Choose the `layout` field by the thinking task:

- Use `method_figure` for the human-controlled method idea and causal closure.
- Use `repository_reading_atlas` for runtime-ordered module-family reading cards.
- Use `repo_tree` when the question is "which file owns which code block?".
- Use `flow_tree` when the question is "what enters a module, what does it own, what exits, and what is forbidden?".
- Omit `layout` or use `tabs` when the question is conceptual taxonomy rather than code ownership.

Reusable parts:

- Page shell: header, hidden editor, status, live reload, zoom, fit-width, pan.
- Drawing helpers: `drawHeader`, `drawLegend`, `drawCard`, `wrapText`, `conceptColor`.
- Shared visual grammar: Code Block IDs, concept color IDs, rough SVG cards, Chinese explanatory text with stable English names.
- Data-driven rendering: a new map should usually require only a new JSON file and an `index.html` link.

Non-reusable parts without refactoring:

- The renderer functions are currently embedded in `architecture_atlas.html`, not exported as a JS library.
- Adding a fourth layout still requires editing `architecture_atlas.html`.
- Cross-file automatic consistency checks are not built into the viewer; consistency is maintained by the JSON contract and review.

New-project adaptation checklist for another LLM Agent:

- Keep `auxiliary/atlas_app/architecture_atlas.html` unchanged at first.
- Replace the example JSON content with the new project's architecture data.
- Update `index.html` links so they point to the new JSON files.
- Keep stable English names in `title` / module labels when they identify code concepts.
- Put explanations, roles, risks, and diagnostics in Chinese if the project owner reads Chinese.
- Preserve Code Block IDs and concept color IDs across maps when the same concept appears in multiple diagrams.
- Do not split the HTML into a JS library unless the viewer itself becomes difficult to maintain.

If the atlas grows further, the next engineering step should be to split the
embedded script into:

```text
viewer_shell.js       shared loading, editor, status, zoom, pan
render_helpers.js    SVG text, cards, colors, wrapping
layouts/             method_figure.js, repository_reading_atlas.js,
                     repo_tree.js, flow_tree.js, tabs.js
```

Do not do this split merely because one map changes. Do it only when the HTML
itself becomes a maintenance bottleneck.

## Static SVG

```bash
node note/architecture/auxiliary/atlas_app/render_rough_arch_svg.mjs
```

## ID Convention

- `P-*`: real problem layer.
- `C-*`: concept variable layer.
- `M-*`: engineering owner/module layer.
- `R-*`: runner code block.
- `A-*`: algorithm code block.
- `S-*`: storage contract block.
- `D-*`: diagnostics block.
- `DR-*`: DR curriculum / GMT frontier block.
- `F-*`: executable floor block.
- `AL-*`: state alpha block.
- `RH-*`: structured rho block.
- `G-*`: diagnostics block.
