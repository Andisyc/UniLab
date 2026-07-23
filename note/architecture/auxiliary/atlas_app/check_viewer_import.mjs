import fs from "node:fs";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";

const html = fs.readFileSync("architecture_atlas.html", "utf8");
const indexHtml = fs.readFileSync("../../index.html", "utf8");
const runtimeAtlas = JSON.parse(
  fs.readFileSync("../../runtime/01_unilab_runtime_atlas.data.json", "utf8"),
);
const methodToCode = JSON.parse(
  fs.readFileSync("../../architecture/02_g1_distillation_method_to_code.data.json", "utf8"),
);
const methodFigure = JSON.parse(
  fs.readFileSync("../../concept/03_g1_multiteacher_distillation_method.data.json", "utf8"),
);
const ampMethodFigure = JSON.parse(
  fs.readFileSync("../../concept/04_amp_walk_async_method.data.json", "utf8"),
);

if (typeof rough.svg !== "function") {
  throw new Error("roughjs import succeeded but rough.svg is missing");
}
if (!html.includes('import rough from "./node_modules/roughjs/bundled/rough.esm.js";')) {
  throw new Error("architecture_atlas.html does not import local roughjs");
}
if (!html.includes('new EventSource("/events")')) {
  throw new Error("architecture_atlas.html is not wired to the auto-refresh event stream");
}
if (!html.includes('<main id="layout" class="editor-hidden">')) {
  throw new Error("architecture_atlas.html should hide the editor sidebar by default");
}
if (!html.includes('<button id="toggle-editor">Show Editor</button>')) {
  throw new Error("architecture_atlas.html default toggle label should be Show Editor");
}
if (!indexHtml.includes('window.location.origin !== "http://127.0.0.1:8766"')) {
  throw new Error("index.html must canonicalize non-Atlas origins to the local Atlas server");
}
if (!indexHtml.includes('window.location.replace("http://127.0.0.1:8766/")')) {
  throw new Error("index.html redirect must target the UniLab Atlas server on port 8766");
}
if (!html.includes('architecture_atlas.html${window.location.search}')) {
  throw new Error("direct viewer access must preserve its data query while canonicalizing the Atlas origin");
}
if (!html.includes('fetch(href, { method: "POST" })')) {
  throw new Error("reading-card source links must use the FEMR 04 POST interaction contract");
}
for (const renderer of [
  "function renderTabs",
  "function renderRepoTree",
  "function renderFlowTree",
  "function renderMethodFigure",
  "function renderRepositoryReadingAtlas",
]) {
  if (!html.includes(renderer)) throw new Error(`viewer missing ${renderer}`);
}
for (const requiredId of [
  'id="toggle-editor"', 'id="zoom-out"', 'id="zoom-in"',
  'id="zoom-fit"', 'id="zoom-reset"', 'id="stage"',
]) {
  if (!html.includes(requiredId)) throw new Error(`viewer missing control ${requiredId}`);
}
if (!html.includes("../../concept/03_g1_multiteacher_distillation_method.data.json")) {
  throw new Error("viewer default data path must point to the active distillation Concept Figure");
}
if (runtimeAtlas.layout !== "repository_reading_atlas") {
  throw new Error("01 UniLab Runtime Atlas must use repository_reading_atlas");
}
if (methodToCode.layout !== "repository_reading_atlas") {
  throw new Error("02 Method-to-Code Atlas must use repository_reading_atlas");
}
if (methodFigure.layout !== "method_figure") {
  throw new Error("distillation Concept Figure must use method_figure");
}
if (ampMethodFigure.layout !== "method_figure") {
  throw new Error("AMP Walk Concept Figure must use method_figure");
}
if (!indexHtml.includes("../../concept/04_amp_walk_async_method.data.json")) {
  throw new Error("Atlas index must expose the AMP Walk Concept Figure");
}

console.log("roughjs viewer import and UniLab atlas data contracts OK");
