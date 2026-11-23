"""Paired, row-clustered comparisons between Myna-v6 and every reference arm.

Resamples bundles (rows) with replacement, recomputes 16-cell macro for both systems on the same
draw, and reports the paired difference with a percentile interval plus a McNemar test on the
question-level discordant pairs.

    python -m bench.paired_bootstrap

Every `runs/baseline_seed*_*ep.json` becomes its own arm, keyed
`deberta_percell_seed{seed}_{ep}ep`. Laya and VegaML are unchanged. No TF-IDF arm lives here.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
B = 20000
SEED = 20261104
CHUNK = 500
PER_CELL_ARMS = ("deberta_percell_seed21_6ep", "deberta_percell_seed7_6ep")
PER_CELLS = ("dbpedia14/category", "mnli/relation", "agnews/topic", "trec/answer_type")


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def ok_map(rows: list[dict]) -> dict[tuple[str, str], int]:
    out = {}
    for r in rows:
        rid = r["meta"]["id"]
        for q, v in r["questions"].items():
            out[(rid, f"{r['meta']['source']}/{q}")] = int(v.get("ok", 0))
    return out


def baseline_map(path: Path) -> dict[tuple[str, str], int]:
    d = json.loads(path.read_text())
    return {(r["row_id"], r["cell"]): int(r["ok"]) for r in d["records"]}


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    keys = set(a) & set(b)
    n01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    n10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, math.erfc(math.sqrt(chi / 2))


def build(m: dict, rid_ix: dict, cell_ix: dict, R: int, C: int):
    H = np.zeros((R, C), dtype=np.float32)
    N = np.zeros((R, C), dtype=np.float32)
    for (rid, cell), ok in m.items():
        i, j = rid_ix.get(rid), cell_ix.get(cell)
        if i is None or j is None:
            continue
        H[i, j] += ok
        N[i, j] += 1
    return H, N


def macro_full(H: np.ndarray, N: np.ndarray) -> float:
    n = N.sum(0)
    h = H.sum(0)
    acc = np.where(n > 0, h / np.maximum(n, 1), np.nan)
    return float(np.nanmean(acc))


def macro_draws(H: np.ndarray, N: np.ndarray, idx: np.ndarray) -> np.ndarray:
    h = H[idx].sum(1)
    n = N[idx].sum(1)
    acc = np.where(n > 0, h / np.maximum(n, 1), np.nan)
    return np.nanmean(acc, axis=1)


def overall_full(H: np.ndarray, N: np.ndarray) -> float:
    return float(H.sum() / max(1.0, N.sum()))


def main() -> None:
    myna = ok_map(load_rows(REPO / "runs/paper_myna_v6_test.rowoutcomes.json"))
    rows = sorted({k[0] for k in myna})
    cells = sorted({k[1] for k in myna})
    rid_ix = {r: i for i, r in enumerate(rows)}
    cell_ix = {c: j for j, c in enumerate(cells)}
    R, C = len(rows), len(cells)

    arms: dict[str, dict[tuple[str, str], int]] = {}
    baseline_files = sorted(REPO.glob("runs/baseline_seed*_*ep.json"))
    for path in baseline_files:
        m = re.match(r"baseline_seed(\d+)_(\d+)ep\.json$", path.name)
        if not m:
            continue
        arms[f"deberta_percell_seed{m.group(1)}_{m.group(2)}ep"] = baseline_map(path)
    for name, rel in (("laya_0419", "runs/laya_0419_test.rowoutcomes.json"),
                      ("vegaml_0150", "runs/vegaml_0150_test.rowoutcomes.json")):
        arms[name] = ok_map(load_rows(REPO / rel))
    # Exp 4: the TF-IDF *rebuild* arm only (0.663), never the published 0.7111 point estimate,
    # which has no per-question records. Its own label keeps the two apart.
    tfidf = REPO / "runs/tfidf_test.rowoutcomes.json"
    if tfidf.exists():
        arms["tfidf_rebuild"] = ok_map(load_rows(tfidf))

    H0, N0 = build(myna, rid_ix, cell_ix, R, C)
    mats = {name: build(m, rid_ix, cell_ix, R, C) for name, m in arms.items()}

    rng = np.random.default_rng(SEED)
    diffs = {name: [] for name in mats}
    cell_diffs = {a: {c: [] for c in PER_CELLS} for a in PER_CELL_ARMS}
    for start in range(0, B, CHUNK):
        k = min(CHUNK, B - start)
        idx = rng.integers(0, R, size=(k, R))
        m0 = macro_draws(H0, N0, idx)
        for name, (H, N) in mats.items():
            diffs[name].append(m0 - macro_draws(H, N, idx))
            if name in cell_diffs:
                for c in PER_CELLS:
                    j = cell_ix[c]
                    a0 = H0[:, j][idx].sum(1) / np.maximum(N0[:, j][idx].sum(1), 1)
                    a1 = H[:, j][idx].sum(1) / np.maximum(N[:, j][idx].sum(1), 1)
                    cell_diffs[name][c].append(a0 - a1)

    out: dict = {
        "cmd": "python -m bench.paired_bootstrap",
        "B": B,
        "seed": SEED,
        "baseline_files": [str(p.relative_to(REPO)) for p in baseline_files],
    }
    for name, (H, N) in mats.items():
        d = np.concatenate(diffs[name])
        lo, hi = np.percentile(d, [2.5, 97.5])
        n01, n10, p = mcnemar(myna, arms[name])
        out[name] = {
            "overall_myna": round(overall_full(H0, N0), 6),
            "overall_arm": round(overall_full(H, N), 6),
            "macro_myna": round(macro_full(H0, N0), 6),
            "macro_arm": round(macro_full(H, N), 6),
            "delta_macro": round(macro_full(H0, N0) - macro_full(H, N), 6),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "mcnemar": {"myna_only": n10, "arm_only": n01, "p": f"{p:.3g}"},
        }
    out["per_cell"] = {
        a: {c: {"delta": round(float(np.mean(np.concatenate(v))), 4),
                "ci95": [round(float(x), 4) for x in np.percentile(np.concatenate(v), [2.5, 97.5])]}
            for c, v in cells_d.items()}
        for a, cells_d in cell_diffs.items()
    }

    (REPO / "runs/paired_bootstrap.json").write_text(json.dumps(out, indent=1))
    for name, d in out.items():
        if name in ("cmd", "B", "seed", "baseline_files", "per_cell"):
            continue
        print(f"{name:30s} delta={d['delta_macro']:+.4f}  CI {d['ci95']}  "
              f"McNemar p={d['mcnemar']['p']}")
    print(f"arms: {len(mats)} | baseline files: {len(baseline_files)} | B={B} seed={SEED}")


if __name__ == "__main__":
    main()
