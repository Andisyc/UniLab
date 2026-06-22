import fs from "node:fs";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";

const html = fs.readFileSync("architecture_atlas.html", "utf8");
const repoMap = JSON.parse(fs.readFileSync("../../architecture/01_repo_architecture.data.json", "utf8"));
const flowMap = JSON.parse(fs.readFileSync("../../runtime/02_frontres_flow.data.json", "utf8"));
const conceptTabs = JSON.parse(fs.readFileSync("../../concept/03_frontres_concept_tabs.data.json", "utf8"));

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

if (!html.includes("let autoFitWidth = true;")) {
  throw new Error("architecture_atlas.html should auto-fit width by default");
}

for (const requiredId of [
  'id="toggle-editor"',
  'id="zoom-out"',
  'id="zoom-in"',
  'id="zoom-fit"',
  'id="zoom-reset"',
  'id="stage"',
]) {
  if (!html.includes(requiredId)) {
    throw new Error(`architecture_atlas.html is missing viewer control ${requiredId}`);
  }
}

for (const requiredHandler of [
  'stage.addEventListener("wheel"',
  'stage.addEventListener("mousedown"',
  'window.addEventListener("resize"',
  'window.addEventListener("mousemove"',
  'window.addEventListener("mouseup"',
]) {
  if (!html.includes(requiredHandler)) {
    throw new Error(`architecture_atlas.html is missing interaction handler ${requiredHandler}`);
  }
}

if (!html.includes("../../concept/03_frontres_concept_tabs.data.json")) {
  throw new Error("architecture_atlas.html default data path must point to ../../concept/");
}

if (repoMap.layout !== "repo_tree") {
  throw new Error("architecture/01_repo_architecture.data.json must use layout=repo_tree");
}

if (flowMap.layout !== "flow_tree") {
  throw new Error("runtime/02_frontres_flow.data.json must use layout=flow_tree");
}

if (!Array.isArray(conceptTabs.tabs)) {
  throw new Error("concept/03_frontres_concept_tabs.data.json must keep tabs[]");
}

console.log("roughjs atlas import and data contracts ok");
