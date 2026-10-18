#!/usr/bin/env node
/**
 * Prueba los módulos del clasificador: para cada frase muestra la etiqueta, la
 * probabilidad (%), el umbral y el tiempo que tarda en responder.
 *
 *   node typescript/playground.ts                 # menú interactivo
 *   node typescript/playground.ts "busco ranked"  # una frase contra todos los módulos
 *   node typescript/playground.ts -m participation "me apunto"
 *   node typescript/playground.ts -t 0.5 "puedo participar?"
 */
import { createInterface } from "node:readline";
import { statSync } from "node:fs";
import { join } from "node:path";
import { styleText } from "node:util";
import { DetectorSet, type SetPrediction, type SplitMetrics } from "./detector.ts";

type Fmt = Parameters<typeof styleText>[0];
const c = (fmt: Fmt, value: string | number): string => styleText(fmt, String(value));

interface Args {
  dir: string;
  modules: string[];
  threshold?: number;
  phrases: string[];
}

const pct = (x: number): string => `${(100 * x).toFixed(1)}%`;

function errorColor(rel: number): Fmt {
  if (rel < 0.1) return "green";
  if (rel < 0.2) return "yellow";
  return "red";
}

function gradeColor(value: number, good: number, ok: number): Fmt {
  if (value >= good) return "green";
  if (value >= ok) return "yellow";
  return "red";
}

function banner(title: string, subtitle = ""): void {
  const line = "─".repeat(title.length + 2);
  console.log(`\n${c("cyan", `╭${line}╮`)}`);
  console.log(`${c("cyan", "│")} ${c(["bold", "cyan"], title)} ${c("cyan", "│")}${subtitle ? `  ${c("dim", subtitle)}` : ""}`);
  console.log(`${c("cyan", `╰${line}╯`)}`);
}

const GLOSSARY: Array<[string, string]> = [
  ["probabilidad", "confianza del modelo en la etiqueta positiva (0–100%)"],
  ["umbral", "probabilidad mínima para marcar la etiqueta positiva"],
  ["precisión", "de lo que marcó positivo, qué % acertó (evita falsas alarmas)"],
  ["recall", "de los positivos reales, qué % encontró (evita dejarse casos)"],
  ["F1", "media de precisión y recall en un solo número"],
  ["aciertos", "porcentaje total de clasificaciones correctas"],
  ["error abs", "número de fallos (falsos positivos + falsos negativos)"],
  ["error rel", "esos fallos como % del total"],
];

function printGlossary(only?: string[]): void {
  for (const [term, text] of GLOSSARY) {
    if (only && !only.includes(term)) continue;
    console.log(`  ${c("bold", term.padEnd(12))} ${c("dim", text)}`);
  }
}

function parseArgs(argv: string[]): Args {
  const args: Args = { dir: "model", modules: [], phrases: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dir" || a === "-d") args.dir = argv[++i];
    else if (a === "--module" || a === "-m") args.modules.push(...argv[++i].split(",").map((s) => s.trim()).filter(Boolean));
    else if (a === "--threshold" || a === "-t") args.threshold = Number(argv[++i]);
    else if (a === "--help" || a === "-h") {
      console.log(`Uso: node typescript/playground.ts [opciones] [frases...]

Opciones:
  -d, --dir <ruta>        carpeta de modelos (por defecto: model)
  -m, --module <nombre>   limita a uno o varios módulos (separados por coma)
  -t, --threshold <0..1>  sobrescribe el umbral de todos los módulos

Si pasas frases, se evalúan y sale. Si no, abre el menú interactivo.`);
      process.exit(0);
    } else {
      args.phrases.push(a);
    }
  }
  return args;
}

class Session {
  readonly set: DetectorSet;
  readonly baseModules: string[];
  active: string[];
  threshold?: number;
  last = "";
  private readonly dir: string;
  private stats = new Map<string, { pos: number; total: number; ms: number }>();
  private readonly nameW: number;

  constructor(set: DetectorSet, args: Args) {
    this.set = set;
    this.dir = args.dir;
    this.baseModules = args.modules.length ? args.modules : set.names();
    for (const m of this.baseModules) {
      if (!set.get(m)) throw new Error(`Módulo desconocido: ${m}. Disponibles: ${set.names().join(", ")}`);
    }
    this.active = [...this.baseModules];
    this.threshold = args.threshold;
    this.nameW = Math.max(6, ...set.names().map((n) => n.length));
  }

  private fileMb(module: string): number {
    try {
      return statSync(join(this.dir, `${module}.json`)).size / 1e6;
    } catch {
      return 0;
    }
  }

  runPhrase(text: string): void {
    for (const name of this.active) {
      const t0 = process.hrtime.bigint();
      const pred = this.set.predict(name, text, this.threshold);
      const msValue = Number(process.hrtime.bigint() - t0) / 1e6;

      const s = this.stats.get(name) ?? { pos: 0, total: 0, ms: 0 };
      s.total++;
      if (pred.decision === this.set.get(name)!.positive) s.pos++;
      s.ms += msValue;
      this.stats.set(name, s);

      this.printPrediction(pred, msValue);
    }
    this.last = text;
  }

  private printPrediction(pred: SetPrediction, msValue: number): void {
    const positive = pred.score >= pred.threshold;
    const decisionColor: Fmt = positive ? ["bold", "green"] : ["bold", "red"];
    const head = `  ${c(["bold", "cyan"], pred.module.padEnd(this.nameW))}  ${c("dim", "→")}  ${c(decisionColor, pred.decision)}`;
    const manual = this.threshold !== undefined ? c("magenta", " (umbral manual)") : "";
    console.log(head + manual);
    console.log(
      `    ${c("dim", "probabilidad")}  ${c(decisionColor, pct(pred.score).padStart(5))}   ` +
        `${c("dim", "umbral")} ${pct(pred.threshold).padStart(5)}   ${c(decisionColor, positive ? "(supera)" : "(no llega)")}`,
    );
    console.log(`    ${c("dim", "tiempo")}        ${c("dim", `${msValue.toFixed(2)} ms`)}`);
  }

  private qualityRow(name: string, m: SplitMetrics, width: number): string {
    const prec = c(gradeColor(m.precision, 0.92, 0.85), pct(m.precision).padStart(9));
    const rec = c(gradeColor(m.recall, 0.85, 0.7), pct(m.recall).padStart(6));
    const err = c(errorColor(m.error_rel), pct(m.error_rel).padStart(6));
    return (
      `  ${c("cyan", name.padEnd(width))}  ${String(this.set.get(name)!.metrics!.n_examples).padStart(8)}  ` +
      `${this.set.get(name)!.threshold.toFixed(3).padStart(6)}  ${prec}  ${rec}  ` +
      `${err}  ${this.fileMb(name).toFixed(2).padStart(5)} MB`
    );
  }

  qualityTable(): void {
    const names = this.set.names();
    if (!names.every((n) => this.set.get(n)!.metrics)) return;
    const width = Math.max(6, ...names.map((n) => n.length));
    console.log(`\n${c(["bold", "cyan"], "Calidad (validación) · tamaño")}`);
    console.log(
      c("bold", `  ${"módulo".padEnd(width)}  ${"ejemplos".padStart(8)}  ${"umbral".padStart(6)}  ` +
        `${"precisión".padStart(9)}  ${"recall".padStart(6)}  ${"error".padStart(6)}  ${"tamaño".padStart(8)}`),
    );
    for (const name of names) console.log(this.qualityRow(name, this.set.get(name)!.metrics!.validation, width));
    console.log(`  ${c("dim", "verde = bien · amarillo = aceptable · rojo = a revisar  ·  detalle en :info")}`);
  }

  info(module: string): void {
    const det = this.set.get(module);
    if (!det) {
      console.log(`Desconocido: ${module}. Disponibles: ${this.baseModules.join(", ")}`);
      return;
    }
    console.log(`\n${c(["bold", "cyan"], module)}  ${c("dim", `(${det.positive} / ${det.negative})`)}`);
    if (!det.metrics) {
      console.log(`  ${c("bold", "modelo")}  ${module}.json · ${this.fileMb(module).toFixed(2)} MB · umbral ${det.threshold}`);
      console.log(`  ${c("dim", "(sin métricas en el modelo)")}`);
    } else {
      const m = det.metrics;
      console.log(
        `  ${c("bold", "datos")}     ${m.n_examples} ejemplos · ${module}.json · ` +
          `${this.fileMb(module).toFixed(2)} MB · umbral ${det.threshold} · entrenado ${m.date}`,
      );
      const center = (text: string): string => text.padStart(12);
      console.log(`  ${c("bold", "métrica".padEnd(10))} ${c("bold", center("validación"))} ${c("bold", center("prueba"))}`);
      const rows: Array<[string, (s: SplitMetrics) => number, (x: number) => string, [number, number]]> = [
        ["precisión", (s) => s.precision, pct, [0.92, 0.85]],
        ["recall", (s) => s.recall, pct, [0.85, 0.7]],
        ["F1", (s) => s.f1, pct, [0.88, 0.75]],
        ["aciertos", (s) => s.accuracy, pct, [0.9, 0.8]],
        ["error abs", (s) => s.error_abs, (x) => String(x), [0, 0]],
        ["error rel", (s) => s.error_rel, pct, [0, 0]],
      ];
      for (const [label, get, fmt, [good, ok]] of rows) {
        const colorOf = (s: SplitMetrics) =>
          label.startsWith("error") ? errorColor(s.error_rel) : gradeColor(get(s), good, ok);
        const v = c(colorOf(m.validation), center(fmt(get(m.validation))));
        const t = c(colorOf(m.test), center(fmt(get(m.test))));
        console.log(`  ${label.padEnd(10)} ${v} ${t}`);
      }
    }
    console.log(`\n  ${c("bold", "Qué significa cada dato")}`);
    printGlossary();
  }

  summary(): void {
    if (!this.stats.size) {
      console.log("Sin consultas todavía.");
      return;
    }
    console.log("\nResumen de la sesión:");
    for (const [name, s] of this.stats) {
      console.log(
        `  ${c("cyan", name.padEnd(this.nameW))}  positivos ${c("bold", `${s.pos}/${s.total}`)}  ` +
          `tiempo medio ${(s.ms / s.total).toFixed(2)} ms`,
      );
    }
  }

  status(): string {
    const thr = this.threshold === undefined ? "automático por módulo" : pct(this.threshold);
    return `${this.active.length} módulo(s): ${this.active.join(", ")} · umbral: ${thr}`;
  }
}

function help(): void {
  console.log(`
Escribe una frase y se prueba en los módulos activos.
Muestra, por cada módulo: la etiqueta, la probabilidad y el tiempo.

Comandos:
  :info [módulo]   calidad del modelo, tamaño y qué significa cada dato
  :use <a,b|all>   elegir módulos activos
  :thr <0..1|off>  fijar umbral manual (off = usar el del modelo)
  :stats           resumen de la sesión
  :help            esta ayuda
  :q               salir
`);
}

const args = parseArgs(process.argv.slice(2));

let set: DetectorSet;
try {
  set = DetectorSet.fromDir(args.dir);
} catch (err) {
  console.error(`No se pudieron cargar los modelos de "${args.dir}": ${(err as Error).message}`);
  console.error("Ejecuta primero:  python train.py");
  process.exit(1);
}
if (!set.names().length) {
  console.error(`No hay módulos en "${args.dir}". Ejecuta primero:  python train.py`);
  process.exit(1);
}

const session = new Session(set, args);

if (args.phrases.length) {
  session.qualityTable();
  console.log(`\n${c("dim", "probabilidad = confianza en la etiqueta positiva · umbral = mínimo para marcarla")}`);
  for (const phrase of args.phrases) {
    console.log(`\n${c("bold", `"${phrase}"`)}`);
    session.runPhrase(phrase);
  }
  process.exit(0);
}

banner("Clasificador de texto", session.status());
session.qualityTable();
console.log(`\n${c("dim", "probabilidad = confianza en la etiqueta positiva · umbral = mínimo para marcarla")}`);
console.log(`Escribe una frase, o ${c("bold", ":help")} para los comandos.`);
const rl = createInterface({ input: process.stdin, output: process.stdout });
rl.setPrompt(c("bold", "> "));
rl.prompt();

for await (const raw of rl) {
  const line = raw.trim();
  if (!line) {
    rl.prompt();
    continue;
  }
  if (line === ":q" || line === ":quit" || line === ":exit") {
    rl.close();
    break;
  } else if (line === ":help") help();
  else if (line === ":stats") session.summary();
  else if (line.startsWith(":info")) {
    const arg = line.slice(5).trim();
    for (const name of arg ? [arg] : session.active) session.info(name);
  } else if (line.startsWith(":use")) {
    const arg = line.slice(4).trim();
    if (arg === "all") session.active = [...session.baseModules];
    else {
      const names = arg.split(",").map((s) => s.trim()).filter(Boolean);
      const bad = names.filter((n) => !session.baseModules.includes(n));
      if (bad.length) console.log(`Desconocidos: ${bad.join(", ")}`);
      session.active = names.filter((n) => session.baseModules.includes(n));
    }
    if (!session.active.length) session.active = [...session.baseModules];
    console.log(session.status());
  } else if (line.startsWith(":thr")) {
    const arg = line.slice(4).trim();
    if (arg === "off" || arg === "") session.threshold = undefined;
    else {
      const t = Number(arg);
      if (Number.isNaN(t) || t < 0 || t > 1) console.log("Uso: :thr <0..1|off>");
      else session.threshold = t;
    }
    console.log(session.status());
  } else {
    if (line.length > 500) console.log("(frase demasiado larga, máx 500 caracteres)");
    else {
      console.log("");
      session.runPhrase(line);
    }
  }
  rl.prompt();
}

console.log("\nHasta luego.");
