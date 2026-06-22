import fs from "node:fs";
import path from "node:path";

const input = process.argv[2] || "../../concept/03_frontres_concept_tabs.data.json";
const output = process.argv[3] || "../../concept/03_frontres_concept_tabs.svg";
const cwd = path.dirname(new URL(import.meta.url).pathname);
const data = JSON.parse(fs.readFileSync(path.resolve(cwd, input), "utf8"));

const width = 1800;
const margin = 56;
const top = 92;
const tabW = 500;
const gap = 48;
const cardW = 430;
const cardH = 112;
const cardGap = 32;
const tabH = 54;
const font = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Helvetica, Arial";

function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function jitter(seed, amp = 4) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * amp;
}

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wrap(text, max = 45) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (next.length > max && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 3);
}

function roughRect(x, y, w, h, fill, stroke = "#161616", id = "") {
  const seed = hash(id);
  const r = 22;
  const paths = [];
  for (let pass = 0; pass < 2; pass++) {
    const a = 2 + pass * 1.4;
    const x0 = x + jitter(seed + pass + 1, a);
    const y0 = y + jitter(seed + pass + 2, a);
    const x1 = x + w + jitter(seed + pass + 3, a);
    const y1 = y + h + jitter(seed + pass + 4, a);
    paths.push(
      `<path d="M ${x0 + r} ${y0} C ${x0 + 4} ${y0} ${x0} ${y0 + 4} ${x0} ${y0 + r} L ${x0} ${y1 - r} C ${x0} ${y1 - 5} ${x0 + 5} ${y1} ${x0 + r} ${y1} L ${x1 - r} ${y1} C ${x1 - 5} ${y1} ${x1} ${y1 - 5} ${x1} ${y1 - r} L ${x1} ${y0 + r} C ${x1} ${y0 + 5} ${x1 - 5} ${y0} ${x1 - r} ${y0} Z" fill="${pass === 0 ? fill : "none"}" stroke="${stroke}" stroke-width="${pass === 0 ? 2.2 : 1.3}" opacity="${pass === 0 ? 0.92 : 0.55}"/>`
    );
  }
  return paths.join("\n");
}

function roughLine(x1, y1, x2, y2, color = "#333", id = "") {
  const seed = hash(id);
  const mx = (x1 + x2) / 2 + jitter(seed + 2, 24);
  const my = (y1 + y2) / 2 + jitter(seed + 3, 18);
  const p1 = `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
  const p2 = `M ${x1 + jitter(seed + 4, 3)} ${y1 + jitter(seed + 5, 3)} Q ${mx + jitter(seed + 6, 5)} ${my + jitter(seed + 7, 5)} ${x2 + jitter(seed + 8, 3)} ${y2 + jitter(seed + 9, 3)}`;
  return `<path d="${p1}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round"/><path d="${p2}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linecap="round" opacity="0.55"/>`;
}

const positions = new Map();
let maxCards = 0;
for (const tab of data.tabs) maxCards = Math.max(maxCards, tab.cards.length);
const height = top + tabH + maxCards * (cardH + cardGap) + 90;

const chunks = [];
chunks.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
chunks.push(`<rect width="100%" height="100%" fill="#fbfaf5"/>`);
chunks.push(`<text x="${margin}" y="48" font-family="${font}" font-size="34" font-weight="800" fill="#111">${esc(data.title)}</text>`);
chunks.push(`<text x="${margin}" y="76" font-family="${font}" font-size="16" fill="#555">Shared IDs connect this concept map with repo mind map and training flow tree.</text>`);

data.tabs.forEach((tab, ti) => {
  const x = margin + ti * (tabW + gap);
  const y = top;
  chunks.push(roughRect(x, y, tabW, tabH, tab.color, "#151515", `${tab.id}-tab`));
  chunks.push(`<text x="${x + 24}" y="${y + 36}" font-family="${font}" font-size="24" font-weight="800" fill="#111">${esc(tab.id)} ${esc(tab.title)}</text>`);
  tab.cards.forEach((card, ci) => {
    const cx = x + 34;
    const cy = y + tabH + 36 + ci * (cardH + cardGap);
    positions.set(card.id, { x: cx, y: cy, w: cardW, h: cardH, color: tab.color });
    chunks.push(roughRect(cx, cy, cardW, cardH, "#fffaf0", tab.color, card.id));
    chunks.push(`<text x="${cx + 20}" y="${cy + 34}" font-family="${font}" font-size="18" font-weight="800" fill="#111">${esc(card.id)} ${esc(card.title)}</text>`);
    wrap(card.body).forEach((line, li) => {
      chunks.push(`<text x="${cx + 20}" y="${cy + 62 + li * 19}" font-family="${font}" font-size="14" fill="#3a3a3a">${esc(line)}</text>`);
    });
  });
});

for (const [from, to] of data.links) {
  const a = positions.get(from);
  const b = positions.get(to);
  if (!a || !b) continue;
  chunks.push(roughLine(a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2, "#333", `${from}-${to}`));
}

chunks.push(`</svg>`);
fs.writeFileSync(path.resolve(cwd, output), chunks.join("\n"));
console.log(path.resolve(cwd, output));
