/**
 * Lanzador de Python multiplataforma: usa el intérprete del venv si existe,
 * si no cae en `python`/`python3` del sistema. Pasa los argumentos tal cual.
 *
 *   node tools/python.mjs train.py --module participation
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const candidates =
  process.platform === "win32"
    ? [".venv\\Scripts\\python.exe", ".venv\\Scripts\\python"]
    : [".venv/bin/python3", ".venv/bin/python"];

const python = candidates.find(existsSync) ?? (process.platform === "win32" ? "python" : "python3");

const result = spawnSync(python, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
  console.error(`No se pudo ejecutar "${python}": ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
