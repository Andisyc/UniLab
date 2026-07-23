import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const atlasRoot = path.resolve(here, "../..");
const repoRoot = path.resolve(atlasRoot, "../..");

const read = (relative) => fs.readFileSync(path.join(repoRoot, relative), "utf8");
const concept = JSON.parse(read("note/architecture/concept/04_amp_walk_async_method.data.json"));
const methodContract = read(
  "note/amp/contracts/active/method/AMP-WALK-METHOD-v002.md",
);
const trainingContract = read(
  "note/amp/contracts/active/training/AMP-WALK-TRAIN-v003.md",
);
const registry = read("note/amp/contracts/README.md");
const plan = read("note/amp/plans/current_engineering_plan.md");
const checklist = read("note/amp/checklists/current.md");
const architectureReadme = read("note/architecture/README.md");
const index = read("note/architecture/index.html");

if (concept.layout !== "method_figure") {
  throw new Error("AMP Concept Figure must use method_figure");
}
if ((concept.zones || []).length || (concept.callouts || []).length || (concept.acceptance || []).length) {
  throw new Error("AMP Concept Figure contains forbidden framing metadata");
}

const expectedMappings = new Map([
  ["AW-M-01", ["AMP-WALK-DP-01", "walk-expert-transitions"]],
  ["AW-M-02", ["AMP-WALK-DP-02", "policy-walk-transitions"]],
  ["AW-M-03", ["AMP-WALK-DP-03", "amp-style-discriminator"]],
  ["AW-M-04", ["AMP-WALK-DP-04", "amp-regularized-walking-policy"]],
]);
const nodes = new Map();
const designIds = new Set();
for (const node of concept.nodes || []) {
  if (nodes.has(node.id)) throw new Error(`duplicate AMP concept node ${node.id}`);
  nodes.set(node.id, node);
  for (const field of ["owner", "status", "codeRefs", "evidence", "acceptance"]) {
    if (field in node) throw new Error(`${node.id} contains forbidden field ${field}`);
  }
  const expected = expectedMappings.get(node.id);
  if (!expected) throw new Error(`unexpected AMP concept node ${node.id}`);
  if (node.design_id !== expected[0] || node.contract_anchor !== expected[1]) {
    throw new Error(`${node.id} contract mapping mismatch`);
  }
  if (node.contract_id !== "AMP-WALK-METHOD-v002") {
    throw new Error(`${node.id} maps to the wrong contract`);
  }
  if (designIds.has(node.design_id)) {
    throw new Error(`design point ${node.design_id} maps to multiple visible blocks`);
  }
  designIds.add(node.design_id);
  if (/[。.!！?？；;]$/.test(String(node.summary || "").trim())) {
    throw new Error(`${node.id} summary ends with punctuation`);
  }
  if (
    !methodContract.includes(node.design_id)
    || !methodContract.includes(node.id)
    || !methodContract.includes(`## ${node.title}`)
  ) {
    throw new Error(`${node.id} mapping absent from active method contract`);
  }
}
if (nodes.size !== expectedMappings.size) {
  throw new Error(`AMP Concept Figure must contain exactly ${expectedMappings.size} blocks`);
}
for (const id of expectedMappings.keys()) {
  if (!nodes.has(id)) throw new Error(`AMP Concept Figure missing ${id}`);
}

const expectedEdges = new Set([
  "AW-M-01->AW-M-03",
  "AW-M-02->AW-M-03",
  "AW-M-03->AW-M-04",
  "AW-M-04->AW-M-02",
]);
const actualEdges = new Set((concept.edges || []).map((edge) => `${edge.from}->${edge.to}`));
if ((concept.edges || []).length !== expectedEdges.size || actualEdges.size !== expectedEdges.size) {
  throw new Error(`AMP Concept Figure must contain exactly ${expectedEdges.size} interactions`);
}
for (const edge of expectedEdges) {
  if (!actualEdges.has(edge)) throw new Error(`AMP Concept Figure missing interaction ${edge}`);
}

function anchor(node, side) {
  const cx = node.x + node.w / 2;
  const cy = node.y + node.h / 2;
  if (side === "left") return [node.x, cy];
  if (side === "right") return [node.x + node.w, cy];
  if (side === "top") return [cx, node.y];
  if (side === "bottom") return [cx, node.y + node.h];
  throw new Error(`${node.id} has invalid anchor ${side}`);
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
  throw new Error(`non-orthogonal segment ${JSON.stringify([a, b])}`);
}

const clearance = concept.connectorClearance ?? 18;
for (const edge of concept.edges || []) {
  const from = nodes.get(edge.from);
  const to = nodes.get(edge.to);
  if (!from || !to) throw new Error(`edge references missing node ${edge.from}->${edge.to}`);
  if (!edge.fromAnchor || !edge.toAnchor) {
    throw new Error(`edge ${edge.from}->${edge.to} requires explicit anchors`);
  }
  if (edge.label && /[。.!！?？；;]$/.test(String(edge.label).trim())) {
    throw new Error(`edge ${edge.from}->${edge.to} label ends with punctuation`);
  }
  const points = [
    anchor(from, edge.fromAnchor),
    ...(edge.via || []),
    anchor(to, edge.toAnchor),
  ];
  for (let index = 1; index < points.length; index += 1) {
    const a = points[index - 1];
    const b = points[index];
    if (a[0] !== b[0] && a[1] !== b[1]) {
      throw new Error(`edge ${edge.from}->${edge.to} segment ${index} is not orthogonal`);
    }
    for (const node of nodes.values()) {
      if (node.id === edge.from || node.id === edge.to) continue;
      const rect = {
        left: node.x - clearance,
        right: node.x + node.w + clearance,
        top: node.y - clearance,
        bottom: node.y + node.h + clearance,
      };
      if (segmentIntersectsRect(a, b, rect)) {
        throw new Error(`edge ${edge.from}->${edge.to} intersects ${node.id}`);
      }
    }
  }
}

for (const required of [
  "AMP-WALK-METHOD-v002",
  "AMP-WALK-TRAIN-v003",
  "history/method/AMP-WALK-METHOD-v001.md",
  "history/training/AMP-WALK-TRAIN-v002.md",
  "history/training/AMP-WALK-TRAIN-v001.md",
]) {
  if (!registry.includes(required)) throw new Error(`AMP registry missing ${required}`);
}
if (!methodContract.includes("note/architecture/concept/04_amp_walk_async_method.data.json")) {
  throw new Error("AMP method contract missing Concept Figure path");
}
for (const required of [
  "AMP-WALK-METHOD-v002",
  "learner freezes `D_k`",
  "implementation_status: style_authority_recovery",
]) {
  if (!trainingContract.includes(required)) {
    throw new Error(`AMP training contract missing ${required}`);
  }
}
for (let step = 1; step <= 3; step += 1) {
  if (!plan.includes(`Step ${step} / 3`)) throw new Error(`recovery plan missing Step ${step}`);
  if (!checklist.includes(`| ${step} / 3 |`)) {
    throw new Error(`recovery checklist missing Step ${step}`);
  }
}
for (const required of [
  "04 AMP Walk Concept Figure",
  "../../concept/04_amp_walk_async_method.data.json",
]) {
  if (!index.includes(required)) throw new Error(`Atlas index missing ${required}`);
}
if (!architectureReadme.includes("concept/04_amp_walk_async_method.data.json")) {
  throw new Error("Architecture README missing AMP Concept Figure");
}

console.log(
  `AMP atlas OK design_points=${designIds.size} concept_nodes=${nodes.size} interactions=${actualEdges.size}`,
);
