"""Entrena TF-IDF + regresión logística, evalúa y exporta model/model.json.

Uso:  python train.py
Datos: data/synthetic.csv (incluido; columnas text,label y opcionalmente group)
       + data/real.csv (opcional, ignorado por git; columnas text,label).
       label = looking | not_looking. 'group' agrupa variantes de una misma frase: nunca se
       reparten entre entrenamiento y prueba.
"""
import argparse
import json
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
MODEL_DIR = ROOT / "model"
LABELS = {"not_looking": 0, "looking": 1}
SEED = 42

def make_pipeline():
    words = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True)
    chars = TfidfVectorizer(strip_accents="unicode", lowercase=True, analyzer="char_wb", ngram_range=(2, 5),
                            min_df=2, sublinear_tf=True)
    return Pipeline([
        ("features", FeatureUnion([("words", words), ("chars", chars)])),
        ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=2000)),
    ])


def load_data():
    frames = [pd.read_csv(ROOT / "data" / "synthetic.csv")]
    real = ROOT / "data" / "real.csv"
    if real.exists():
        r = pd.read_csv(real)[["text", "label"]]
        r["group"] = [f"real-{i}" for i in range(len(r))]
        frames.append(r)
        print(f"data/real.csv: {len(r)} ejemplos reales")
    df = pd.concat(frames, ignore_index=True)
    if "group" not in df:
        df["group"] = np.nan
    df["group"] = df["group"].fillna(pd.Series([f"fila-{i}" for i in range(len(df))], index=df.index))

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise SystemExit(f"Etiquetas no válidas: {bad}. Usa looking | not_looking")
    df = df.drop_duplicates("text")
    key = df["text"].str.lower()
    conflicts = df[key.duplicated(keep=False)].groupby(key)["label"].nunique()
    if (conflicts > 1).any():
        raise SystemExit(f"Mismo texto con etiquetas distintas: {list(conflicts[conflicts > 1].index)[:5]}")
    df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    df["y"] = df["label"].map(LABELS)
    return df


def metrics(y_true, p, thr):
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y_true)), "precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4),
            "f1": round(float(f1[0]), 4), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def choose_threshold(y_val, p_val, min_precision, floor):
    """Umbral más bajo (>= floor) que cumple la precisión mínima de 'looking' en validación."""
    for thr in np.arange(floor, 0.995, 0.005):
        pred = p_val >= thr
        if pred.sum() and y_val[pred].mean() >= min_precision:
            return round(float(thr), 3)
    return 0.99


def r6(x):
    return round(float(x), 6)


def export(pipe, threshold, path):
    feats = dict(pipe.named_steps["features"].transformer_list)
    coef = pipe.named_steps["lr"].coef_[0]
    out = {"version": 1, "labels": ["not_looking", "looking"], "threshold": threshold,
           "intercept": r6(pipe.named_steps["lr"].intercept_[0])}
    offset = 0
    for name in ("words", "chars"):
        vec = feats[name]
        terms = {t: [r6(vec.idf_[i]), r6(coef[offset + i])] for t, i in vec.vocabulary_.items()}
        out[name] = {"ngramRange": list(vec.ngram_range), "terms": terms}
        offset += len(vec.vocabulary_)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-precision", type=float, default=0.97,
                    help="precisión mínima de 'looking' en validación para elegir umbral")
    ap.add_argument("--min-threshold", type=float, default=0.85,
                    help="suelo del umbral (la validación sintética es optimista)")
    args = ap.parse_args()

    df = load_data()
    print(f"{len(df)} ejemplos | looking={int(df['y'].sum())} not_looking={int((1 - df['y']).sum())} "
          f"| grupos={df['group'].nunique()}")

    # Partición por grupos: variantes de la misma plantilla nunca cruzan entre entrenamiento y evaluación.
    # Fold 0 = prueba (no se toca hasta el final). Folds 1-5 = validación cruzada para elegir el umbral.
    folds = [idx for _, idx in StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
             .split(df["text"], df["y"], df["group"])]
    test_idx = folds[0]
    dev_idx = np.concatenate(folds[1:])
    te, dev = df.iloc[test_idx], df.iloc[dev_idx]
    print(f"desarrollo={len(dev)} prueba={len(te)}")

    oof_idx, oof_p = [], []
    for k in range(1, 6):
        val_k = folds[k]
        train_k = np.setdiff1d(dev_idx, val_k)
        m = make_pipeline().fit(df["text"].iloc[train_k], df["y"].iloc[train_k])
        oof_idx.append(val_k)
        oof_p.append(m.predict_proba(df["text"].iloc[val_k])[:, 1])
    oof_idx, p_val = np.concatenate(oof_idx), np.concatenate(oof_p)
    va = df.iloc[oof_idx]

    thr = choose_threshold(va["y"].to_numpy(), p_val, args.min_precision, args.min_threshold)
    pipe = make_pipeline().fit(dev["text"], dev["y"])
    p_te = pipe.predict_proba(te["text"])[:, 1]

    report = {"date": str(date.today()), "n_examples": len(df), "threshold": thr,
              "validation": metrics(va["y"], p_val, thr), "test": metrics(te["y"], p_te, thr)}
    real_mask = te["group"].str.startswith("real-").to_numpy()
    if real_mask.any():
        report["test_real_only"] = metrics(te["y"][real_mask], p_te[real_mask], thr)

    print(f"\nUmbral: {thr}")
    for k in ("validation", "test", "test_real_only"):  # validation = validación cruzada por grupos
        if k in report:
            print(f"{k:15s}", report[k])
    fps = te[(p_te >= thr) & (te["y"] == 0)]["text"].tolist()
    fns = te[(p_te < thr) & (te["y"] == 1)]["text"].tolist()
    fp_groups = te[(p_te >= thr) & (te["y"] == 0)]["group"].nunique()
    print(f"\nFalsos positivos (test): {len(fps)} en {fp_groups} plantillas distintas ->", fps[:15])
    print("Falsos negativos (test):", fns[:15])

    # Modelo final con todos los datos
    final = make_pipeline().fit(df["text"], df["y"])
    MODEL_DIR.mkdir(exist_ok=True)
    export(final, thr, MODEL_DIR / "model.json")

    size = (MODEL_DIR / "model.json").stat().st_size / 1e6
    print(f"\nExportado model/model.json ({size:.2f} MB)")

    names = list(dict(final.named_steps["features"].transformer_list)["words"].get_feature_names_out())
    w = final.named_steps["lr"].coef_[0][:len(names)]
    order = np.argsort(w)
    print("Palabras más 'looking':    ", [names[i] for i in order[::-1][:12]])
    print("Palabras más 'not_looking':", [names[i] for i in order[:12]])


if __name__ == "__main__":
    main()
