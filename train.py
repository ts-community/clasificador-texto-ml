"""Entrena un clasificador TF-IDF + regresión logística por cada módulo de data/.

Un módulo = un fichero `data/<nombre>.csv` (columnas `text,label,group`). Al ejecutar
`python train.py` se entrena cada uno y se exporta `model/<nombre>.json`.

Las etiquetas de cada CSV son libres, pero deben ser exactamente dos y la negativa
debe empezar por `no_` o `not_` (p. ej. `looking` / `not_looking`, `joining` /
`not_joining`). Así basta con crear un dataset nuevo para tener un módulo nuevo.

Datos opcionales reales por módulo: `data/<nombre>.real.csv` (ignorado por git;
columnas `text,label`). `group` agrupa variantes de una misma frase: nunca se
reparten entre entrenamiento y prueba.

Uso:
    python train.py                 # entrena todos los módulos
    python train.py --module participation
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
NEG_PREFIXES = ("no_", "not_")
SEED = 42
# Suavizado L2 del score (ver detector.ts): evita que una sola palabra genérica
# ("quiero", "voy"...) dispare el score. Calibrado para no perder recall legítimo.
SMOOTH = 10.0
# Precisión objetivo de la clase positiva por módulo (más alto = más estricto: menos
# falsos positivos a cambio de algo de recall). Sobreescribible con --min-precision.
MODULE_MIN_PRECISION = {"participation": 0.96}
DEFAULT_MIN_PRECISION = 0.95

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
          "blue": "34", "magenta": "35", "cyan": "36"}
_color = False


def paint(text, *styles):
    """Colorea/bold con ANSI. Sin efecto si la salida no es una terminal o NO_COLOR está fijado."""
    if not _color or not styles:
        return str(text)
    prefix = "".join(f"\033[{_CODES[s]}m" for s in styles)
    return f"{prefix}{text}\033[0m"


def grade_color(value, good, ok):
    """Verde si value >= good, amarillo si >= ok, rojo si por debajo."""
    return "green" if value >= good else ("yellow" if value >= ok else "red")


def err_color(rel):
    return "green" if rel < 0.10 else ("yellow" if rel < 0.20 else "red")


def banner(title, subtitle=""):
    inner = len(title) + 2
    top = "╭" + "─" * inner + "╮"
    mid = f"│ {title} │"
    bot = "╰" + "─" * inner + "╯"
    print()
    print(paint(top, "cyan"))
    print(paint("│", "cyan") + paint(f" {title} ", "bold", "cyan") + paint("│", "cyan"))
    print(paint(bot, "cyan") + (f"  {paint(subtitle, 'dim')}" if subtitle else ""))


def enable_ansi():
    """Activa secuencias ANSI en la consola de Windows (sin dependencias). Devuelve si hay color."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
    except Exception:
        pass
    return False


def make_pipeline():
    words = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True)
    chars = TfidfVectorizer(strip_accents="unicode", lowercase=True, analyzer="char_wb", ngram_range=(2, 5),
                            min_df=2, sublinear_tf=True)
    return Pipeline([
        ("features", FeatureUnion([("words", words), ("chars", chars)])),
        ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=2000)),
    ])


def discover_modules():
    names = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name.endswith(".real.csv"):
            continue
        names.append(path.stem)
    return names


def resolve_labels(labels):
    """Etiquetas del CSV -> (negativa, positiva). La negativa empieza por no_/not_."""
    values = sorted(set(labels))
    if len(values) != 2:
        raise SystemExit(f"Se esperan exactamente 2 etiquetas, hay {values}")
    neg = [v for v in values if v.lower().startswith(NEG_PREFIXES)]
    if len(neg) != 1:
        raise SystemExit(
            f"No se distingue la etiqueta negativa en {values}. Debe empezar por no_ o not_ "
            f"(p. ej. joining / not_joining)")
    negative = neg[0]
    positive = next(v for v in values if v != negative)
    return negative, positive


def load_data(module):
    frames = [pd.read_csv(DATA_DIR / f"{module}.csv")]
    real = DATA_DIR / f"{module}.real.csv"
    if real.exists():
        r = pd.read_csv(real)[["text", "label"]]
        r["group"] = [f"real-{i}" for i in range(len(r))]
        frames.append(r)
        print(f"  data/{module}.real.csv: {len(r)} ejemplos reales")
    df = pd.concat(frames, ignore_index=True)
    if "group" not in df:
        df["group"] = np.nan
    df["group"] = df["group"].fillna(pd.Series([f"fila-{i}" for i in range(len(df))], index=df.index))

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates("text")
    key = df["text"].str.lower()
    conflicts = df[key.duplicated(keep=False)].groupby(key)["label"].nunique()
    if (conflicts > 1).any():
        raise SystemExit(f"Mismo texto con etiquetas distintas: {list(conflicts[conflicts > 1].index)[:5]}")
    df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    return df


def metrics(y_true, p, thr):
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = int(len(y_true))
    errors = int(fp + fn)
    return {"n": n, "precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4),
            "f1": round(float(f1[0]), 4), "accuracy": round((tp + tn) / n, 4) if n else 0.0,
            "error_abs": errors, "error_rel": round(errors / n, 4) if n else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def pct(x):
    return f"{100 * x:.1f}%"


def metric_row(name, values, colors=None, header=False):
    cells = []
    for i, val in enumerate(values):
        text = f"{val:>12}"
        color = colors[i] if colors else None
        cells.append(paint(text, color) if color else text)
    return f"  {name:<10} " + " ".join(cells)


def choose_threshold(y_val, p_val, min_precision, floor):
    """Umbral más bajo (>= floor) que cumple la precisión mínima de la clase positiva en validación."""
    for thr in np.arange(floor, 0.995, 0.005):
        pred = p_val >= thr
        if pred.sum() and y_val[pred].mean() >= min_precision:
            return round(float(thr), 3)
    return 0.99


def r6(x):
    return round(float(x), 6)


def export(pipe, threshold, module, negative, positive, path, metrics):
    feats = dict(pipe.named_steps["features"].transformer_list)
    coef = pipe.named_steps["lr"].coef_[0]
    out = {"version": 4, "module": module, "labels": [negative, positive],
           "negative": negative, "positive": positive, "threshold": threshold,
           "intercept": r6(pipe.named_steps["lr"].intercept_[0]), "smooth": SMOOTH,
           "metrics": metrics}
    offset = 0
    for name in ("words", "chars"):
        vec = feats[name]
        terms = {t: [r6(vec.idf_[i]), r6(coef[offset + i])] for t, i in vec.vocabulary_.items()}
        out[name] = {"ngramRange": list(vec.ngram_range), "terms": terms}
        offset += len(vec.vocabulary_)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def train_module(module, args):
    df = load_data(module)
    negative, positive = resolve_labels(df["label"])
    labels = {negative: 0, positive: 1}
    df["y"] = df["label"].map(labels)
    min_precision = args.min_precision if args.min_precision is not None \
        else MODULE_MIN_PRECISION.get(module, DEFAULT_MIN_PRECISION)

    n_groups = df["group"].nunique()
    n_splits = min(6, n_groups)
    if n_splits < 3:
        raise SystemExit(f"'{module}' necesita al menos 3 grupos distintos para evaluarse")

    folds = [idx for _, idx in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
             .split(df["text"], df["y"], df["group"])]
    test_idx = folds[0]
    dev_idx = np.concatenate(folds[1:])
    te, dev = df.iloc[test_idx], df.iloc[dev_idx]

    oof_idx, oof_p = [], []
    for k in range(1, n_splits):
        val_k = folds[k]
        train_k = np.setdiff1d(dev_idx, val_k)
        m = make_pipeline().fit(df["text"].iloc[train_k], df["y"].iloc[train_k])
        oof_idx.append(val_k)
        oof_p.append(m.predict_proba(df["text"].iloc[val_k])[:, 1])
    oof_idx, p_val = np.concatenate(oof_idx), np.concatenate(oof_p)
    va = df.iloc[oof_idx]

    thr = choose_threshold(va["y"].to_numpy(), p_val, min_precision, args.min_threshold)
    pipe = make_pipeline().fit(dev["text"], dev["y"])
    p_te = pipe.predict_proba(te["text"])[:, 1]

    val = metrics(va["y"], p_val, thr)
    test = metrics(te["y"], p_te, thr)

    report = {"date": str(date.today()), "n_examples": len(df), "threshold": thr,
              "validation": val, "test": test}
    real_mask = te["group"].str.startswith("real-").to_numpy()
    if real_mask.any():
        report["test_real_only"] = metrics(te["y"][real_mask], p_te[real_mask], thr)

    final = make_pipeline().fit(df["text"], df["y"])
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{module}.json"
    model_metrics = {k: report[k] for k in ("date", "n_examples", "validation", "test", "test_real_only")
                     if k in report}
    export(final, thr, module, negative, positive, path, model_metrics)
    size = path.stat().st_size / 1e6
    report["file"] = f"{module}.json"
    report["file_mb"] = round(size, 3)

    pos_n, neg_n = int(df["y"].sum()), int((1 - df["y"]).sum())
    print(f"\n{paint('▍', 'cyan')}{paint(module, 'bold', 'cyan')}  {paint(f'({positive} / {negative})', 'dim')}")
    print(f"  {paint('datos', 'bold')}      {len(df)} ejemplos · {n_groups} grupos · "
          f"{positive} {pos_n} · {negative} {neg_n} · modelo {size:.2f} MB · umbral {thr}")
    print(metric_row("métrica", ["validación", "prueba"], header=True))
    print(metric_row("precisión", [pct(val["precision"]), pct(test["precision"])],
                     [grade_color(val["precision"], 0.92, 0.85), grade_color(test["precision"], 0.92, 0.85)]))
    print(metric_row("recall", [pct(val["recall"]), pct(test["recall"])],
                     [grade_color(val["recall"], 0.85, 0.70), grade_color(test["recall"], 0.85, 0.70)]))
    print(metric_row("F1", [pct(val["f1"]), pct(test["f1"])],
                     [grade_color(val["f1"], 0.88, 0.75), grade_color(test["f1"], 0.88, 0.75)]))
    print(metric_row("aciertos", [pct(val["accuracy"]), pct(test["accuracy"])],
                     [grade_color(val["accuracy"], 0.90, 0.80), grade_color(test["accuracy"], 0.90, 0.80)]))
    print(metric_row("error abs", [val["error_abs"], test["error_abs"]],
                     [err_color(val["error_rel"]), err_color(test["error_rel"])]))
    print(metric_row("error rel", [pct(val["error_rel"]), pct(test["error_rel"])],
                     [err_color(val["error_rel"]), err_color(test["error_rel"])]))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", help="entrenar solo este módulo (por defecto, todos)")
    ap.add_argument("--min-precision", type=float, default=None,
                    help="precisión mínima de la clase positiva en validación para elegir umbral "
                         "(por defecto: 0.95, o el valor propio del módulo)")
    ap.add_argument("--min-threshold", type=float, default=0.5,
                    help="suelo del umbral: no baja de aquí aunque logre la precisión mínima")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    global _color
    _color = bool(sys.stdout.isatty()) and os.environ.get("NO_COLOR") is None and enable_ansi()

    modules = [args.module] if args.module else discover_modules()
    if not modules:
        raise SystemExit("No hay datasets en data/. Crea data/<modulo>.csv")

    banner("Entrenando módulos", f"{len(modules)} módulo(s): {', '.join(modules)}")

    reports = {}
    for module in modules:
        if not (DATA_DIR / f"{module}.csv").exists():
            raise SystemExit(f"No existe data/{module}.csv")
        reports[module] = train_module(module, args)

    width = max((len(m) for m in modules), default=8)
    header = f"{'módulo':<{width}}  {'ejemplos':>8}  {'umbral':>6}  {'precisión':>9}  {'recall':>6}  {'error':>6}  {'tamaño':>8}"
    print(f"\n{paint('Resumen', 'bold', 'cyan')}  {paint('(validación · verde = bien · amarillo = aceptable · rojo = a revisar)', 'dim')}")
    print("  " + paint(header, "bold"))
    for module in modules:
        r = reports[module]
        v = r["validation"]
        mid = f"  {r['n_examples']:>8}  {r['threshold']:>6.3f}  "
        prec = paint(pct(v["precision"]).rjust(9), grade_color(v["precision"], 0.92, 0.85))
        rec = paint(pct(v["recall"]).rjust(6), grade_color(v["recall"], 0.85, 0.70))
        err = paint(pct(v["error_rel"]).rjust(6), err_color(v["error_rel"]))
        print(f"  {paint(module.ljust(width), 'bold')}{mid}{prec}  {rec}  {err}  {r['file_mb']:>6.2f} MB")
    print(f"\n{paint('✔', 'green')} {len(modules)} módulo(s) exportado(s) -> {paint('model/', 'bold')}")


if __name__ == "__main__":
    main()
