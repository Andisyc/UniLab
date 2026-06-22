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

- `architecture/01_unilab_repo_architecture.data.json`: editable source data for the UniLab repo ownership map.
- `runtime/02_unilab_runtime_flow.data.json`: editable source data for the UniLab runtime contract map.
- `auxiliary/atlas_app/`: helper viewer, local server, static renderer, checks, and JS dependencies.

## Folder Contract

```text
note/architecture/
  architecture/   repo/file/block mind map
  runtime/        module interface contract map
  auxiliary/      helper app files kept out of the map folders
  index.html      clean entry page
```

## Map Lifecycle

Temporary maps are allowed, including in the main entry page, while they are
actively guiding a change. After the change lands, a temporary map must be
either deleted or integrated into one of the active maps.

The main entry should stay small: UniLab repo ownership map and UniLab runtime
contract map.

## VSCode Workflow

```bash
cd note/architecture
node auxiliary/atlas_app/serve_architecture.mjs
```

Open one of these URLs on the right side of VSCode:

```text
http://127.0.0.1:8765/
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../architecture/01_unilab_repo_architecture.data.json
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../runtime/02_unilab_runtime_flow.data.json
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

## HTML Design Contract

The current atlas uses one reusable HTML viewer:

```text
auxiliary/atlas_app/architecture_atlas.html
  -> loads one *.data.json through ?data=...
  -> chooses renderer by data.layout
  -> draws rough SVG cards with shared colors, IDs, zoom, pan, editor, and live reload
```

The three main pages are data variants, not separate applications:

- Repo Architecture uses `layout: "repo_tree"`.
  - Source: `architecture/01_unilab_repo_architecture.data.json`.
  - Purpose: file tree -> code block ownership.
  - Main schema: `title`, `subtitle`, `layout`, `root`, `concepts`, `files[]`.
  - Each file has `group`, `path`, `color`, `blocks[]`.
  - Each block has `id`, `role`, `lines`, `concept`.

- Runtime uses `layout: "flow_tree"`.
  - Source: `runtime/02_unilab_runtime_flow.data.json`.
  - Purpose: interface boundary -> input / ownership / output / forbidden freedom / diagnostic proof.
  - Main schema: `title`, `subtitle`, `layout`, `concepts`, `nodes[]`.
  - Each node has `id`, `title`, `role`, `input`, `output`, `forbidden`, `diagnostic`, `concept`, optional `children[]`.

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
http://127.0.0.1:8765/
```

To reuse the current HTML page for a specific map, create or edit a
`*.data.json` file and open:

```text
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../PATH/TO/MAP.data.json
```

Choose the `layout` field by the thinking task:

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
layouts/             repo_tree.js, flow_tree.js, tabs.js
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
