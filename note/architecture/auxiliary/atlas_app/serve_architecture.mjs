import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const atlasRoot = path.resolve(__dirname, "../..");
const port = Number(process.env.PORT || 8765);
const clients = new Set();

const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
]);

function sendEvent(eventName) {
  for (const client of clients) {
    client.write(`event: ${eventName}\n`);
    client.write(`data: ${Date.now()}\n\n`);
  }
}

function watchDataFiles() {
  for (const dir of [
    path.join(atlasRoot, "architecture"),
    path.join(atlasRoot, "runtime"),
    path.join(atlasRoot, "concept"),
  ]) {
    if (!fs.existsSync(dir)) continue;
    fs.watch(dir, { persistent: true }, (_eventType, filename) => {
      if (filename && filename.endsWith(".data.json")) {
        sendEvent("architecture-data");
      }
    });
  }
}

function safeResolve(urlPath) {
  const cleanPath = decodeURIComponent(urlPath.split("?")[0]);
  const relativePath = cleanPath === "/" ? "index.html" : cleanPath.slice(1);
  const resolved = path.resolve(atlasRoot, relativePath);
  if (!resolved.startsWith(atlasRoot)) return null;
  return resolved;
}

const server = http.createServer((req, res) => {
  if (req.url === "/events") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    res.write("\n");
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }

  const filePath = safeResolve(req.url || "/");
  if (!filePath) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(error.code === "ENOENT" ? 404 : 500);
      res.end(error.code === "ENOENT" ? "Not found" : String(error));
      return;
    }
    res.writeHead(200, {
      "Content-Type": mimeTypes.get(path.extname(filePath)) || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(data);
  });
});

watchDataFiles();

server.listen(port, "127.0.0.1", () => {
  console.log(`MOSAIC architecture atlas: http://127.0.0.1:${port}/`);
  console.log(`Repo map: http://127.0.0.1:${port}/auxiliary/atlas_app/architecture_atlas.html?data=../../architecture/01_repo_architecture.data.json`);
  console.log(`Interface map: http://127.0.0.1:${port}/auxiliary/atlas_app/architecture_atlas.html?data=../../runtime/02_frontres_flow.data.json`);
  console.log(`Concept tabs: http://127.0.0.1:${port}/auxiliary/atlas_app/architecture_atlas.html?data=../../concept/03_frontres_concept_tabs.data.json`);
  console.log(`Watching data folders: architecture/, runtime/, concept/`);
});
