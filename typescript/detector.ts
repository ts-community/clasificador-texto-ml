/**
 * Clasificador "¿busca compañeros de equipo?" — inferencia en TypeScript puro, sin dependencias.
 * Reproduce exactamente el TF-IDF + regresión logística de scikit-learn (ver train.py).
 *
 *   const detector = TeamSearchDetector.fromFile("model/model.json");
 *   const { score, decision } = detector.predict("busco gente para ranked");
 */
import { readFileSync } from "node:fs";

type Terms = Record<string, [idf: number, coef: number]>;

interface RawBlock {
  ngramRange: [number, number];
  terms: Terms;
}

export interface RawModel {
  version: number;
  labels: string[];
  threshold: number;
  intercept: number;
  words: RawBlock;
  chars: RawBlock;
}

export interface Prediction {
  score: number; // probabilidad de "looking" (no está calibrada: valídala con datos reales)
  decision: "looking" | "not_looking";
}

interface Block {
  range: [number, number];
  terms: Map<string, [number, number]>;
}

const COMBINING = /[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]/g;
const WORD = /[\p{L}\p{N}_]{2,}/gu;

// Igual que TfidfVectorizer(lowercase=True, strip_accents="unicode")
function preprocess(text: string): string {
  return text.toLowerCase().normalize("NFKD").replace(COMBINING, "");
}

// Igual que analyzer="word" (token_pattern \b\w\w+\b)
function wordNgrams(text: string, [minN, maxN]: [number, number]): string[] {
  const tokens = text.match(WORD) ?? [];
  const out: string[] = [];
  for (let n = minN; n <= Math.min(maxN, tokens.length); n++) {
    for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
  }
  return out;
}

// Igual que analyzer="char_wb": n-gramas de caracteres dentro de cada palabra rodeada de espacios
function charWbNgrams(text: string, [minN, maxN]: [number, number]): string[] {
  const out: string[] = [];
  for (const word of text.replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean)) {
    const w = Array.from(` ${word} `); // por puntos de código, como Python
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      out.push(w.slice(offset, offset + n).join(""));
      while (offset + n < w.length) {
        offset++;
        out.push(w.slice(offset, offset + n).join(""));
      }
      if (offset === 0) break; // palabra más corta que n: se cuenta una sola vez
    }
  }
  return out;
}

// TF sublineal (1 + ln tf) · idf, normalizado L2, producto con los coeficientes
function contribution(block: Block, grams: string[]): number {
  const tf = new Map<string, number>();
  for (const g of grams) if (block.terms.has(g)) tf.set(g, (tf.get(g) ?? 0) + 1);
  let dot = 0;
  let sq = 0;
  for (const [g, count] of tf) {
    const [idf, coef] = block.terms.get(g)!;
    const v = (1 + Math.log(count)) * idf;
    dot += v * coef;
    sq += v * v;
  }
  return sq > 0 ? dot / Math.sqrt(sq) : 0;
}

function toBlock(raw: RawBlock): Block {
  return { range: raw.ngramRange, terms: new Map(Object.entries(raw.terms)) };
}

export class TeamSearchDetector {
  readonly threshold: number;
  private readonly intercept: number;
  private readonly words: Block;
  private readonly chars: Block;

  constructor(raw: RawModel) {
    this.threshold = raw.threshold;
    this.intercept = raw.intercept;
    this.words = toBlock(raw.words);
    this.chars = toBlock(raw.chars);
  }

  static fromFile(path: string | URL): TeamSearchDetector {
    return new TeamSearchDetector(JSON.parse(readFileSync(path, "utf8")) as RawModel);
  }

  score(text: string): number {
    const t = preprocess(text);
    const z =
      this.intercept +
      contribution(this.words, wordNgrams(t, this.words.range)) +
      contribution(this.chars, charWbNgrams(t, this.chars.range));
    return 1 / (1 + Math.exp(-z));
  }

  /** `threshold` opcional: sobrescribe el del modelo (p. ej. desde una variable de entorno). */
  predict(text: string, threshold: number = this.threshold): Prediction {
    const score = this.score(text);
    return { score, decision: score >= threshold ? "looking" : "not_looking" };
  }
}
