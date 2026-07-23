import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const atlasRoot = path.resolve(here, "../..");
const repoRoot = path.resolve(atlasRoot, "../..");
const readJson = (relative) => JSON.parse(fs.readFileSync(path.join(atlasRoot, relative), "utf8"));

const runtime = readJson("runtime/01_unilab_runtime_atlas.data.json");
const methodToCode = readJson("architecture/02_g1_distillation_method_to_code.data.json");
const concept = readJson("concept/03_g1_multiteacher_distillation_method.data.json");
const contract = fs.readFileSync(
  path.join(repoRoot, "note/distillation/contracts/active/method/DISTILL-METHOD-v001.md"),
  "utf8",
);
const viewer = fs.readFileSync(path.join(here, "architecture_atlas.html"), "utf8");
const server = fs.readFileSync(path.join(here, "serve_architecture.mjs"), "utf8");
const index = fs.readFileSync(path.join(atlasRoot, "index.html"), "utf8");

const indexRedirectMatch = index.match(/<script>([\s\S]*?)<\/script>/);
if (!indexRedirectMatch) throw new Error("Atlas index redirect script missing");
let indexRedirectUrl = null;
new vm.Script(indexRedirectMatch[1]).runInNewContext({
  window: {
    location: {
      origin: "null",
      replace: (url) => { indexRedirectUrl = url; },
    },
  },
});
if (indexRedirectUrl !== "http://127.0.0.1:8766/") {
  throw new Error(`Atlas index redirect mismatch: ${indexRedirectUrl}`);
}

const viewerRedirectMatch = viewer.match(/<script>([\s\S]*?)<\/script>/);
if (!viewerRedirectMatch) throw new Error("Atlas viewer redirect script missing");
let viewerRedirectUrl = null;
new vm.Script(viewerRedirectMatch[1]).runInNewContext({
  window: {
    location: {
      origin: "http://preview.invalid",
      search: "?data=../../runtime/01_unilab_runtime_atlas.data.json",
      replace: (url) => { viewerRedirectUrl = url; },
    },
  },
});
if (viewerRedirectUrl !== "http://127.0.0.1:8766/auxiliary/atlas_app/architecture_atlas.html?data=../../runtime/01_unilab_runtime_atlas.data.json") {
  throw new Error(`Atlas viewer redirect mismatch: ${viewerRedirectUrl}`);
}

const scriptMatch = viewer.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("viewer module script missing");
const parseable = scriptMatch[1].replace(/^\s*import rough[^;]+;\s*$/m, "const rough = {};");
new vm.Script(parseable, { filename: "architecture_atlas.inline.mjs" });
for (const required of [
  "function renderMethodFigure",
  "function renderRepositoryReadingAtlas",
  "function addSourceTextLink",
  "/open-source?path=",
]) {
  if (!viewer.includes(required)) throw new Error(`viewer missing ${required}`);
}
if (!server.includes('requestUrl.pathname === "/open-source"')) {
  throw new Error("atlas server missing /open-source owner");
}
if (!server.includes('res.writeHead(204, { "Cache-Control": "no-store" })')) {
  throw new Error("atlas server must acknowledge FEMR 04 source opens with HTTP 204");
}
if (server.includes('open -a "Visual Studio Code"')) {
  throw new Error("atlas server must use VS Code CLI --goto, not macOS open -a");
}

if (concept.layout !== "method_figure") throw new Error("concept must use method_figure");
if ((concept.zones || []).length || (concept.callouts || []).length || (concept.acceptance || []).length) {
  throw new Error("Concept Figure contains forbidden framing metadata");
}
const conceptNodes = new Map();
for (const node of concept.nodes || []) {
  if (conceptNodes.has(node.id)) throw new Error(`duplicate concept node ${node.id}`);
  conceptNodes.set(node.id, node);
  for (const field of ["owner", "status", "codeRefs"]) {
    if (field in node) throw new Error(`${node.id} contains forbidden field ${field}`);
  }
  if (!node.design_id || !node.contract_id || !node.contract_anchor) {
    throw new Error(`${node.id} missing contract mapping`);
  }
  if (/[。.!！?？；;]$/.test(String(node.summary || "").trim())) {
    throw new Error(`${node.id} summary ends with punctuation`);
  }
  if (!contract.includes(node.design_id) || !contract.includes(node.id)) {
    throw new Error(`${node.id} mapping absent from active contract`);
  }
}
const topLevelDesignIds = new Set(
  [...conceptNodes.values()].filter((node) => node.kind !== "external").map((node) => node.design_id),
);
for (const expected of [
  "DISTILL-DP-01", "DISTILL-DP-02", "DISTILL-DP-03",
  "DISTILL-DP-04", "DISTILL-DP-05",
]) {
  if (!topLevelDesignIds.has(expected)) throw new Error(`Concept Figure missing ${expected}`);
}
if (conceptNodes.has("DT-M-06") || [...conceptNodes.values()].some((node) => node.title === "Single Policy")) {
  throw new Error("Single Policy must remain an output property of MoE Student, not a Concept Figure block");
}

function conceptAnchor(node, anchor) {
  const cx = node.x + node.w / 2;
  const cy = node.y + node.h / 2;
  if (anchor === "left") return [node.x, cy];
  if (anchor === "right") return [node.x + node.w, cy];
  if (anchor === "top") return [cx, node.y];
  if (anchor === "bottom") return [cx, node.y + node.h];
  throw new Error(`${node.id} has invalid anchor ${anchor}`);
}

function segmentIntersectsRect(a, b, rect) {
  const minX = Math.min(a[0], b[0]);
  const maxX = Math.max(a[0], b[0]);
  const minY = Math.min(a[1], b[1]);
  const maxY = Math.max(a[1], b[1]);
  if (a[0] === b[0]) {
    return a[0] >= rect.left && a[0] <= rect.right
      && maxY >= rect.top && minY <= rect.bottom;
  }
  if (a[1] === b[1]) {
    return a[1] >= rect.top && a[1] <= rect.bottom
      && maxX >= rect.left && minX <= rect.right;
  }
  throw new Error(`Concept Figure connector segment is not orthogonal: ${JSON.stringify([a, b])}`);
}

const requiredConceptEdges = new Set([
  "DT-M-02->DT-M-01",
  "DT-M-01->DT-M-03",
  "DT-M-03->DT-M-04",
  "DT-M-04->DT-X-01",
  "DT-X-01->DT-M-05",
  "DT-M-05->DT-M-03",
  "DT-M-02->DT-M-04",
]);
const expectedConceptRoutes = new Map([
  ["DT-M-02->DT-M-01", "spine"],
  ["DT-M-01->DT-M-03", "spine"],
  ["DT-M-03->DT-M-04", "spine"],
  ["DT-M-04->DT-X-01", "spine"],
  ["DT-M-02->DT-M-04", "upper"],
  ["DT-X-01->DT-M-05", "lower"],
  ["DT-M-05->DT-M-03", "lower"],
]);
const actualConceptEdges = new Set((concept.edges || []).map((edge) => `${edge.from}->${edge.to}`));
for (const expected of requiredConceptEdges) {
  if (!actualConceptEdges.has(expected)) throw new Error(`Concept Figure missing interaction ${expected}`);
}
if (actualConceptEdges.size !== requiredConceptEdges.size) {
  throw new Error(`Concept Figure must contain exactly ${requiredConceptEdges.size} approved interactions`);
}
if ((concept.edges || []).length !== requiredConceptEdges.size) {
  throw new Error("Concept Figure contains duplicate or extra interactions");
}

const connectorClearance = concept.connectorClearance ?? 18;
const spineNodes = ["DT-M-02", "DT-M-01", "DT-M-03", "DT-M-04", "DT-X-01"]
  .map((id) => conceptNodes.get(id));
const spineCenterY = spineNodes[0].y + spineNodes[0].h / 2;
const spineTop = Math.min(...spineNodes.map((node) => node.y));
const spineBottom = Math.max(...spineNodes.map((node) => node.y + node.h));
for (const edge of concept.edges || []) {
  if (!conceptNodes.has(edge.from) || !conceptNodes.has(edge.to)) {
    throw new Error(`edge references missing node ${edge.from}->${edge.to}`);
  }
  if (!edge.fromAnchor || !edge.toAnchor) {
    throw new Error(`edge ${edge.from}->${edge.to} requires explicit anchors`);
  }
  if (edge.label && /[。.!！?？；;]$/.test(String(edge.label).trim())) {
    throw new Error(`edge ${edge.from}->${edge.to} label ends with punctuation`);
  }
  const from = conceptNodes.get(edge.from);
  const to = conceptNodes.get(edge.to);
  const edgeKey = `${edge.from}->${edge.to}`;
  if (edge.route !== expectedConceptRoutes.get(edgeKey)) {
    throw new Error(`edge ${edgeKey} must use route=${expectedConceptRoutes.get(edgeKey)}`);
  }
  const points = [
    conceptAnchor(from, edge.fromAnchor),
    ...(edge.via || []),
    conceptAnchor(to, edge.toAnchor),
  ];
  if (edge.route === "spine" && (edge.via?.length || points.some((point) => point[1] !== spineCenterY))) {
    throw new Error(`edge ${edgeKey} must remain on the horizontal causal spine`);
  }
  if (edge.route === "upper" && (edge.via || []).some((point) => point[1] > spineTop - connectorClearance)) {
    throw new Error(`edge ${edgeKey} upper route entered the spine corridor`);
  }
  if (edge.route === "lower" && (edge.via || []).some((point) => point[1] < spineBottom + connectorClearance)) {
    throw new Error(`edge ${edgeKey} lower route entered the spine corridor`);
  }
  for (let index = 1; index < points.length; index += 1) {
    const a = points[index - 1];
    const b = points[index];
    if (a[0] !== b[0] && a[1] !== b[1]) {
      throw new Error(`edge ${edge.from}->${edge.to} segment ${index} is not orthogonal`);
    }
    for (const node of conceptNodes.values()) {
      if (node.id === edge.from || node.id === edge.to) continue;
      const rect = {
        left: node.x - connectorClearance,
        right: node.x + node.w + connectorClearance,
        top: node.y - connectorClearance,
        bottom: node.y + node.h + connectorClearance,
      };
      if (segmentIntersectsRect(a, b, rect)) {
        throw new Error(`edge ${edge.from}->${edge.to} intersects non-endpoint ${node.id}`);
      }
    }
  }
}

const daggerIncoming = (concept.edges || []).find(
  (edge) => edge.from === "DT-X-01" && edge.to === "DT-M-05",
);
const daggerOutgoing = (concept.edges || []).find(
  (edge) => edge.from === "DT-M-05" && edge.to === "DT-M-03",
);
if (daggerIncoming?.toAnchor !== "right" || daggerOutgoing?.fromAnchor !== "left") {
  throw new Error("Student-State DAgger feedback must enter right-center and leave left-center");
}
const daggerNode = conceptNodes.get("DT-M-05");
const moeNode = conceptNodes.get("DT-M-04");
if (daggerNode.x + daggerNode.w / 2 !== moeNode.x + moeNode.w / 2) {
  throw new Error("Student-State DAgger must be vertically center-aligned with MoE Student");
}
const daggerIncomingY = conceptAnchor(daggerNode, daggerIncoming.toAnchor)[1];
const daggerOutgoingY = conceptAnchor(daggerNode, daggerOutgoing.fromAnchor)[1];
if (daggerIncomingY !== daggerOutgoingY) {
  throw new Error("Student-State DAgger feedback anchors must share one horizontal centerline");
}
const daggerIncomingLastVia = daggerIncoming.via?.at(-1);
if (
  daggerIncoming.via?.length !== 1
  || !daggerIncomingLastVia
  || daggerIncomingLastVia[1] !== daggerIncomingY
  || daggerIncomingLastVia[0] < daggerNode.x + daggerNode.w + connectorClearance
) {
  throw new Error("Student-State DAgger right feedback must use one bend and finish horizontally inward");
}
const daggerOutgoingFirstVia = daggerOutgoing.via?.[0];
if (
  !daggerOutgoingFirstVia
  || daggerOutgoingFirstVia[1] !== daggerOutgoingY
  || daggerOutgoingFirstVia[0] > daggerNode.x - connectorClearance
) {
  throw new Error("Student-State DAgger left feedback must start with a horizontal outward segment");
}

if (methodToCode.layout !== "repository_reading_atlas") {
  throw new Error("method-to-code must use repository_reading_atlas");
}
const moduleIds = new Set();
for (const system of methodToCode.systems || []) {
  if (!system.id || !system.title || !system.summary || !system.color) {
    throw new Error(`incomplete system ${system.id || "<missing>"}`);
  }
  for (const module of system.modules || []) {
    if (moduleIds.has(module.id)) throw new Error(`duplicate module ${module.id}`);
    moduleIds.add(module.id);
    for (const field of ["summary", "owns", "mustNot", "gap"]) {
      if (!module[field]) throw new Error(`${module.id} missing ${field}`);
    }
    if (!Array.isArray(module.mainRoute) || module.mainRoute.length < 2) {
      throw new Error(`${module.id} missing multi-step mainRoute`);
    }
    if (module.mainRouteTitles?.length !== module.mainRoute.length) {
      throw new Error(`${module.id} route titles mismatch`);
    }
    module.mainRoute.forEach((step, index) => {
      if (!String(step).startsWith(`B${index + 1} `) || !String(step).includes(" -> ")) {
        throw new Error(`${module.id} invalid route step ${index + 1}`);
      }
    });
    for (const file of module.files || []) {
      const absolute = path.resolve(repoRoot, file.path);
      if (!absolute.startsWith(`${repoRoot}${path.sep}`) || !fs.existsSync(absolute)) {
        throw new Error(`${module.id} source path missing or unsafe: ${file.path}`);
      }
      if (!Number.isInteger(file.sourceLine) || file.sourceLine < 1) {
        throw new Error(`${module.id} invalid sourceLine for ${file.path}`);
      }
    }
  }
}
const ordered = [...(methodToCode.runtimeOrder || []), ...(methodToCode.supportOrder || [])];
if (new Set(ordered).size !== ordered.length) throw new Error("reading order contains duplicates");
for (const id of moduleIds) if (!ordered.includes(id)) throw new Error(`unordered module ${id}`);
for (const id of ordered) if (!moduleIds.has(id)) throw new Error(`reading order missing module ${id}`);

if (runtime.layout !== "repository_reading_atlas") {
  throw new Error("01 UniLab Runtime Atlas must use repository_reading_atlas");
}
if ((runtime.supportOrder || []).length) {
  throw new Error("01 UniLab Runtime Atlas must not contain supporting cards");
}
const runtimeIds = new Set();
for (const system of runtime.systems || []) {
  if (!system.id || !system.title || !system.summary || !system.color) {
    throw new Error(`incomplete runtime system ${system.id || "<missing>"}`);
  }
  for (const module of system.modules || []) {
    if (runtimeIds.has(module.id)) throw new Error(`duplicate runtime module ${module.id}`);
    runtimeIds.add(module.id);
    for (const field of ["summary", "owns", "mustNot", "gap"]) {
      if (!module[field]) throw new Error(`${module.id} missing ${field}`);
    }
    if (!Array.isArray(module.files) || !module.files.length) {
      throw new Error(`${module.id} missing read-first files`);
    }
    if (!Array.isArray(module.mainRoute) || module.mainRoute.length < 2) {
      throw new Error(`${module.id} missing multi-step mainRoute`);
    }
    if (module.mainRouteTitles?.length !== module.mainRoute.length) {
      throw new Error(`${module.id} route titles mismatch`);
    }
    module.mainRoute.forEach((step, index) => {
      if (!String(step).startsWith(`B${index + 1} `) || !String(step).includes(" -> ")) {
        throw new Error(`${module.id} invalid route step ${index + 1}`);
      }
    });
    for (const file of module.files) {
      const absolute = path.resolve(repoRoot, file.path);
      if (!absolute.startsWith(`${repoRoot}${path.sep}`) || !fs.existsSync(absolute)) {
        throw new Error(`${module.id} source path missing or unsafe: ${file.path}`);
      }
      if (!Number.isInteger(file.sourceLine) || file.sourceLine < 1) {
        throw new Error(`${module.id} invalid sourceLine for ${file.path}`);
      }
    }
  }
}
const runtimeOrder = runtime.runtimeOrder || [];
if (runtimeOrder.length !== runtimeIds.size || new Set(runtimeOrder).size !== runtimeOrder.length) {
  throw new Error("01 runtimeOrder must contain every runtime module exactly once");
}
for (const id of runtimeOrder) if (!runtimeIds.has(id)) throw new Error(`runtimeOrder missing module ${id}`);
for (const expected of [
  "U-RT-01", "U-RT-02", "U-RT-03", "U-RT-04", "U-RT-05",
  "U-RT-06", "U-RT-07", "U-RT-08", "U-RT-09",
]) {
  if (!runtimeIds.has(expected)) throw new Error(`01 Runtime Atlas missing ${expected}`);
}

const stalePerformanceGapPatterns = [
  /(?:reset\/resource|MuJoCo|live)\s+timing[^;,.]*(?:尚缺|尚未|未连接|absent)/i,
  /A\/B[^;,.]*(?:尚缺|尚未|未执行|absent)/i,
  /(?:尚缺|尚未|未连接|未执行|absent)[^;,.]*(?:A\/B|timing)/i,
];
for (const atlas of [runtime, methodToCode]) {
  for (const system of atlas.systems || []) {
    for (const mod of system.modules || []) {
      for (const pattern of stalePerformanceGapPatterns) {
        if (pattern.test(String(mod.gap || ""))) {
          throw new Error(`${mod.id} contains stale timing/A/B gap: ${mod.gap}`);
        }
      }
    }
  }
}
for (const requiredId of ["U-RT-06", "U-RT-08"]) {
  const runtimeModule = (runtime.systems || [])
    .flatMap((system) => system.modules || [])
    .find((mod) => mod.id === requiredId);
  for (const requiredFact of ["E67", "NO_STABLE_SPEEDUP", "HP-6"]) {
    if (!String(runtimeModule?.gap || "").includes(requiredFact)) {
      throw new Error(`${requiredId} gap missing current fact ${requiredFact}`);
    }
  }
}

const indexAtlasLinks = [...index.matchAll(/architecture_atlas\.html\?data=/g)];
if (indexAtlasLinks.length !== 4) throw new Error("Atlas index must expose exactly four maps");
for (const forbidden of ["Supporting:", "Distillation Runtime", "Distillation Control Room"]) {
  if (index.includes(forbidden)) throw new Error(`Atlas index contains forbidden entry ${forbidden}`);
}
for (const required of [
  "01 UniLab Runtime Atlas", "02 Method-to-Code Atlas", "03 Concept Figure",
  "04 AMP Walk Concept Figure",
]) {
  if (!index.includes(required)) throw new Error(`Atlas index missing ${required}`);
}

console.log(
  `atlas OK runtime_modules=${runtimeIds.size} method_modules=${moduleIds.size} concept_nodes=${conceptNodes.size}`,
);
