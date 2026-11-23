#!/usr/bin/env python
"""Dyad-style two-encoder baseline for the Myna resubmission (Exp 5 design / Exp 6 pilot).

What this is
------------
A *transplant* of Dyad's factorisation idea onto this suite: a typed decision is scored as
``f(e, h) . g(e, a)`` — a state encoder that never sees the options, and an option encoder
that never sees the state, sharing only the task prefix ``e`` (the "environment description"
on both sides). Training is supervised cross-entropy on the gold option, exactly Dyad's
alignment loss in the single-shot case; there is no RL, no GRPO, and no reward.

    f input:  [CLS] {prefix} | Context: {state} [SEP]        -> q, [D]
    g input:  [CLS] {prefix} | Option: {option_text} [SEP]   -> z_a, [D]
    logit_a = z_a . q ; softmax over the question's options.

The arm is distinguishable from v6's decoupled head in exactly one place: v6's decoupled
arm encodes each option *against the state* (``_packed_ids(prefix, state, [opt])``); here
the option encoder's input drops the state field. Everything else — backbone, data,
splits, seeds, epochs, metric, recipe — is held fixed.

Commands
--------
    python release/dyad_baseline.py --table                 # per-cell option-identity table
    python release/dyad_baseline.py --guard                 # circularity checks A + B (CPU)
    python release/dyad_baseline.py --epochs 3 --seed 42    # pilot (GPU; not run yet)

No checkpoint is written by --table or --guard.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from myna.real_data import _option_text, _state_text          # noqa: E402
from myna.v6_engine import MynaV6                             # noqa: E402

BACKBONE = str(REPO / "myna-v5-prototype 2" / "deberta-v5-finetuned")
CANARY = "ZZQXCANARY7F3"          # appears nowhere in the suite


# ---------------------------------------------------------------- renderers
def prefix_for(cell: str, spec: dict) -> str:
    """The engine's own prefix, unchanged (MynaV6._prefix touches only `spec`)."""
    return MynaV6._prefix(None, cell, spec)


def options_for(spec: dict) -> list[str]:
    """The engine's own option texts (handles the banking77 `null` description trap)."""
    return MynaV6._options(None, spec)


def labels_for(spec: dict) -> list[str]:
    return MynaV6._labels(spec)


def gold_for(spec: dict) -> int:
    """Same semantics as the corrected release/train_v6.py gold_index."""
    t = spec["type"]
    if t == "noul":
        return 1 if spec["label"] else 0
    if t == "score":
        return int(spec["label"])          # 0-based index, NOT label - 1
    return labels_for(spec).index(spec["label"])


def _fix_tokenizer_config(backbone) -> None:
    """This checkpoint ships extra_special_tokens as a list; transformers >=4.45 wants a dict
    and raises "'list' object has no attribute 'keys'". Drop the redundant key (the tokens are
    already declared individually and in added_tokens_decoder)."""
    cfg = Path(backbone) / "tokenizer_config.json"
    if not cfg.exists():
        return
    d = json.loads(cfg.read_text())
    if isinstance(d.get("extra_special_tokens"), list):
        d.pop("extra_special_tokens")
        try:
            cfg.write_text(json.dumps(d, indent=1))
            print(f"patched {cfg} (removed list-valued extra_special_tokens)", flush=True)
        except OSError as exc:
            print(f"could not patch {cfg} ({exc}); continuing", flush=True)


def f_text(prefix: str, state: str) -> str:
    return f"[CLS] {prefix} | Context: {state} [SEP]"


def g_text(prefix: str, option: str) -> str:
    return f"[CLS] {prefix} | Option: {option} [SEP]"


# ---------------------------------------------------------------- data
def load_records(split: str = "train", limit: int = 0) -> list[dict]:
    path = REPO / "data" / "decision-v2-pilot" / f"{split}.jsonl"
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if limit:
        lines = lines[:limit]
    out = []
    for ln in lines:
        row = json.loads(ln)
        source = row["_meta"]["source"]
        state = _state_text(row["state"])
        for qname, spec in row["questions"].items():
            cell = f"{source}/{qname}"
            opts = options_for(spec)
            gold = gold_for(spec)
            assert 0 <= gold < len(opts), f"{cell}: gold {gold} outside 0..{len(opts)-1}"
            out.append({"row_id": row["_meta"]["id"], "cell": cell, "prefix": prefix_for(cell, spec),
                        "state": state, "options": opts, "gold": gold, "primitive": spec["type"]})
    return out


def option_identity_table(records: list[dict]) -> dict:
    """Per cell: is (prefix, options) constant across rows? (drives the eval cache strategy)"""
    seen: dict[str, set] = defaultdict(set)
    rows: dict[str, set] = defaultdict(set)
    for r in records:
        seen[r["cell"]].add((r["prefix"], tuple(r["options"])))
        rows[r["cell"]].add(r["row_id"])
    table = {}
    for cell in sorted(seen):
        table[cell] = {"n_rows": len(rows[cell]), "distinct_prefix_option_sets": len(seen[cell]),
                       "option_identity_holds": len(seen[cell]) == 1}
    return table


# ---------------------------------------------------------------- model
class AttentionPool(nn.Module):
    """Attention pooling over last hidden states -> Linear(D, D). Same on both sides."""

    def __init__(self, hidden: int = 1024, out: int = 1024):
        super().__init__()
        self.score = nn.Linear(hidden, 1)
        self.proj = nn.Linear(hidden, out)

    def forward(self, hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = self.score(hs).squeeze(-1).masked_fill(mask == 0, -1e9)
        a = torch.softmax(w, dim=-1)
        return self.proj((a.unsqueeze(-1) * hs).sum(1))


class Tower(nn.Module):
    def __init__(self, backbone: str, freeze: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(backbone)
        self.pool = AttentionPool()
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.pool.parameters():
                p.requires_grad_(False)

    def forward(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
        return self.pool(hs, attn)


class Dyad(nn.Module):
    def __init__(self, backbone: str = BACKBONE):
        super().__init__()
        self.f = Tower(backbone, freeze=True)     # state encoder: frozen in the pilot
        self.g = Tower(backbone, freeze=False)    # option encoder: separate copy

    def logits(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return z @ q


# ---------------------------------------------------------------- batching
def enc_batch(tok, texts: list[str], device, max_len: int):
    e = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return e["input_ids"].to(device), e["attention_mask"].to(device)


@torch.no_grad()
def score_question(model, tok, prefix, state, options, device, max_len=512, cache=None):
    """Returns softmax scores aligned to `options` (one question)."""
    q_ids, q_att = enc_batch(tok, [f_text(prefix, state)], device, max_len)
    q = model.f(q_ids, q_att)[0]
    zs = []
    for opt in options:
        key = (prefix, opt)
        if cache is not None and key in cache:
            zs.append(cache[key])
            continue
        z_ids, z_att = enc_batch(tok, [g_text(prefix, opt)], device, max_len)
        z = model.g(z_ids, z_att)[0]
        if cache is not None:
            cache[key] = z
        zs.append(z)
    z = torch.stack(zs)
    return torch.softmax(z @ q, dim=-1)


# ---------------------------------------------------------------- training
def _plan(by_cell, seed, epochs, batch):
    """Deterministic micro-batch order: same seed -> same plan, so resume can skip by index."""
    plan = []
    for epoch in range(epochs):
        cells = sorted(by_cell)
        random.Random(seed * 1000 + epoch).shuffle(cells)
        for cell in cells:
            items = by_cell[cell][:]
            random.Random(seed * 1000 + epoch).shuffle(items)
            for start in range(0, len(items) - batch + 1, batch):
                plan.append((epoch, cell, items[start:start + batch]))
    return plan


def _save_ckpt(path, model, optim, sched, step, chunk, epoch, args, device):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"g": model.g.state_dict(),
                "f_pool": model.f.pool.state_dict(),
                "optim": optim.state_dict(), "sched": sched.state_dict(),
                "step": step, "chunk": chunk, "epoch": epoch,
                "config": vars(args),
                "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}},
               path)


def train(args) -> None:
    from transformers import AutoTokenizer
    _fix_tokenizer_config(BACKBONE)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    records = load_records("train", args.limit)
    if args.only_cells:
        keep = {c.strip() for c in args.only_cells.split(",") if c.strip()}
        records = [r for r in records if r["cell"] in keep]
    for i, r in enumerate(records):
        r["_idx"] = i
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = Dyad().to(device)
    f_device = torch.device(args.f_device)
    if not args.q_cache and f_device.type != device.type:
        model.f.to(f_device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if hasattr(model.g.encoder, "gradient_checkpointing_enable"):
        model.g.encoder.gradient_checkpointing_enable()
        model.g.encoder.enable_input_require_grads()
    model.f.eval()

    Q = None
    if args.q_cache and args.q_cache_file and Path(args.q_cache_file).exists():
        Q = torch.load(args.q_cache_file, map_location=device)
        model.f.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"q loaded from {args.q_cache_file}", flush=True)
    elif args.q_cache:
        print(f"precomputing q for {len(records)} questions ...", flush=True)
        tq = time.time()
        Q = torch.zeros(len(records), 1024, device=device)
        qbs = 32
        for i in range(0, len(records), qbs):
            part = records[i:i + qbs]
            ids, attn = enc_batch(tok, [f_text(c["prefix"], c["state"]) for c in part],
                                  device, args.max_len)
            with torch.no_grad():
                qv = model.f(ids, attn)
            Q[i:i + len(part)] = qv
            if (i // qbs) % 200 == 0:
                print(f"  q {i}/{len(records)} ({time.time()-tq:.0f}s)", flush=True)
        model.f.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.q_cache_file:
            Path(args.q_cache_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save(Q.cpu(), args.q_cache_file)
            print(f"q saved to {args.q_cache_file}", flush=True)
        print(f"q cached in {time.time()-tq:.0f}s; f moved off GPU", flush=True)

    params = [p for p in model.g.parameters() if p.requires_grad]
    optim = torch.optim.AdamW([
        {"params": list(model.g.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(model.g.pool.parameters()), "lr": args.lr_head},
    ], weight_decay=0.01)

    by_cell = defaultdict(list)
    for r in records:
        by_cell[r["cell"]].append(r)
    plan = _plan(by_cell, args.seed, args.epochs, args.batch)
    total = len(plan) // args.accum
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / max(1, total)))

    step, chunk_start, epoch = 0, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.g.load_state_dict(ck["g"]); model.f.pool.load_state_dict(ck["f_pool"])
        optim.load_state_dict(ck["optim"]); sched.load_state_dict(ck["sched"])
        for st in optim.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        step, chunk_start, epoch = ck["step"], ck["chunk"], ck["epoch"]
        random.setstate(ck["rng"]["python"]); torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda" and ck["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resumed from {args.resume}: step {step}, chunk {chunk_start}, epoch {epoch}", flush=True)

    print(f"dyad: {len(records)} questions | {len(by_cell)} cells | micro-batch {args.batch} "
          f"x accum {args.accum} | total {total} steps | warmup {warmup} | device {device}", flush=True)

    t0 = time.time()
    _steps_done = 0
    for ci in range(chunk_start, len(plan)):
        ep, cell, chunk = plan[ci]
        shared = len({(c["prefix"], tuple(c["options"])) for c in chunk}) == 1
        if shared:
            pfx = chunk[0]["prefix"]; opts = chunk[0]["options"]
            from torch.utils.checkpoint import checkpoint as _ckpt
            zs = []
            for oi in range(0, len(opts), args.opt_batch):
                part = opts[oi:oi + args.opt_batch]
                z_ids, z_att = enc_batch(tok, [g_text(pfx, o) for o in part], device, args.max_len)
                zs.append(_ckpt(lambda a, b: model.g(a, b), z_ids, z_att, use_reentrant=False))
            z = torch.cat(zs, 0)
        if Q is not None:
            q = Q[[c["_idx"] for c in chunk]]
        else:
            q_ids, q_att = enc_batch(tok, [f_text(c["prefix"], c["state"]) for c in chunk],
                                     f_device, args.max_len)
            with torch.no_grad():
                q = model.f(q_ids, q_att).to(device)
        if shared:
            logits = q @ z.t()
        else:
            rows = []
            max_k = max(len(c["options"]) for c in chunk)
            for i, c in enumerate(chunk):
                zs = []
                for o in c["options"]:
                    ids, attn = enc_batch(tok, [g_text(c["prefix"], o)], device, args.max_len)
                    zs.append(model.g(ids, attn)[0])
                row = torch.stack(zs) @ q[i]
                if row.numel() < max_k:
                    row = torch.cat([row, row.new_full((max_k - row.numel(),), -1e4)])
                rows.append(row)
            logits = torch.stack(rows)
        gold = torch.tensor([c["gold"] for c in chunk], device=device)
        loss = F.cross_entropy(logits, gold) / args.accum
        loss.backward()
        if (ci + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1
            _steps_done += 1
            if step % args.ckpt_every == 0:
                _save_ckpt(REPO / args.ckpt_dir / "latest.pt", model, optim, sched, step,
                           ci + 1, ep, args, device)
                print(f"  ckpt step {step}/{total} ({time.time()-t0:.0f}s)", flush=True)
            if step % 200 == 0:
                print(f"  step {step}/{total} loss {loss.item()*args.accum:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if args.max_steps and step >= args.max_steps:
                print("max-steps reached; stopping cleanly", flush=True)
                break
        epoch = ep
    _save_ckpt(REPO / args.ckpt_dir / "latest.pt", model, optim, sched, step,
               min(ci + 1, len(plan)), epoch, args, device)
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"g": model.g.state_dict(), "f_pool": model.f.pool.state_dict(),
                "config": vars(args), "step": step}, out / "dyad.pt")
    if _steps_done:
        print(f"train_wall_s: {time.time()-t0:.1f} steps: {_steps_done} "
              f"s_per_step: {(time.time()-t0)/_steps_done:.2f}", flush=True)
    print("wrote", out, "at step", step, flush=True)
    if device.type == "mps":
        try:
            print("peak_mps_mib:", torch.mps.driver_allocated_memory() // (1024*1024), flush=True)
        except Exception as e:
            print("peak_mps_mib: n/a", e, flush=True)


# ---------------------------------------------------------------- guards
def evaluate(args) -> None:
    """Score a trained checkpoint on the test split; print and optionally write JSON."""
    from transformers import AutoTokenizer
    _fix_tokenizer_config(BACKBONE)
    from collections import defaultdict
    device = torch.device("mps" if getattr(torch.backends, "mps", None) and
                          torch.backends.mps.is_available() and not torch.cuda.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = Dyad()
    ck = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)
    model.g.load_state_dict(ck["g"] if "g" in ck else ck)
    model.f.eval(); model.g.eval(); model.to(device)
    recs = load_records("test")
    if args.only_cells:
        keep = {c.strip() for c in args.only_cells.split(",") if c.strip()}
        recs = [r for r in recs if r["cell"] in keep]
    per = defaultdict(list); bundles = defaultdict(list)
    for r in recs:
        zs = []
        for o in r["options"]:
            ids, attn = enc_batch(tok, [g_text(r["prefix"], o)], device, args.max_len)
            with torch.no_grad():
                zs.append(model.g(ids, attn)[0])
        q = None
        with torch.no_grad():
            qids, qatt = enc_batch(tok, [f_text(r["prefix"], r["state"])], device, args.max_len)
            q = model.f(qids, qatt)[0]
        pred = int((torch.stack(zs) @ q).argmax())
        ok = int(pred == r["gold"])
        per[r["cell"]].append(ok); bundles[r["row_id"]].append(ok)
    cells = {c: sum(v)/len(v) for c, v in per.items()}
    macro = sum(cells.values())/len(cells)
    overall = sum(sum(v) for v in per.values())/sum(len(v) for v in per.values())
    bundle_exact = sum(1 for v in bundles.values() if all(v))/len(bundles)
    print(f"macro {macro:.6f} overall {overall:.6f} bundle_exact {bundle_exact:.6f} cells {len(cells)}", flush=True)
    for c, v in sorted(cells.items()):
        print(f"  {c} {v:.4f} n={len(per[c])}", flush=True)
    if args.eval_out:
        Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.eval_out).write_text(json.dumps(
            {"macro": macro, "overall": overall, "bundle_exact": bundle_exact,
             "cells": cells, "n": len(recs), "footing": "Dyad D-frozen, 15 cells, banking77 excluded (OOM)"}, indent=1))


def guard(args) -> None:
    """Checks A and B from the plan; CPU only; writes runs/dyad_guard.json."""
    from transformers import AutoTokenizer
    torch.manual_seed(0)          # the pools are randomly initialised; fix them for reproducibility
    device = torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = Dyad().to(device).eval()
    records = load_records("test")
    by_cell = defaultdict(list)
    for r in records:
        by_cell[r["cell"]].append(r)

    t0 = time.time()
    # ---- Check A: canary in f, absent from g, on one sample per cell
    a_fail = []
    for cell in sorted(by_cell):
        r = by_cell[cell][0]
        marked_state = f"{r['state']} {CANARY}"
        ft = f_text(r["prefix"], marked_state)
        gts = [g_text(r["prefix"], o) for o in r["options"]]
        ok = (CANARY in ft) and all(CANARY not in t for t in gts) and \
             all(t == g_text(r["prefix"], o) for t, o in zip(gts, r["options"]))
        if not ok:
            a_fail.append(cell)
    print(f"CHECK_A: {'PASS' if not a_fail else 'FAIL'} ({len(by_cell)} cells, "
          f"canary {CANARY})", flush=True)

    # ---- Check B: state swap changes scores; option permutation leaves pairs unchanged
    picks = [by_cell[c][0] for c in sorted(by_cell) if len(by_cell[c][0]["options"]) >= 3][:2]
    b_state_ok, b_perm_ok, deltas = True, True, []
    cache: dict = {}
    for r in picks:
        other = picks[1] if r is picks[0] else picks[0]
        s1 = score_question(model, tok, r["prefix"], r["state"], r["options"], device,
                            cache=None)
        s_swapped = score_question(model, tok, r["prefix"], other["state"], r["options"],
                                   device, cache=None)
        d = (s1 - s_swapped).abs().max().item()
        deltas.append(d)
        if d < 1e-6:
            b_state_ok = False
        order = list(range(len(r["options"])))[::-1]
        s_perm = score_question(model, tok, r["prefix"], r["state"],
                                [r["options"][i] for i in order], device, cache=None)
        # s_perm[j] corresponds to option order[j]; compare as a set of (option, score)
        pairs_a = {r["options"][i]: float(s1[i]) for i in range(len(r["options"]))}
        pairs_b = {r["options"][i]: float(s_perm[order.index(i)]) for i in range(len(r["options"]))}
        if any(abs(pairs_a[k] - pairs_b[k]) > 1e-6 for k in pairs_a):
            b_perm_ok = False
    print(f"CHECK_B: state-swap {'PASS' if b_state_ok else 'FAIL'} (max|d|={max(deltas):.3e}), "
          f"permutation {'PASS' if b_perm_ok else 'FAIL'}", flush=True)

    payload = {
        "check_a": {"pass": not a_fail, "cells": len(by_cell), "fail_cells": a_fail,
                    "canary": CANARY},
        "check_b": {"state_swap_pass": b_state_ok, "permutation_pass": b_perm_ok,
                    "max_abs_delta": max(deltas), "n_questions": len(picks)},
        "check_c": {
            "sentence": ("The option encoder input drops the state field; everything else "
                         "(backbone, data, splits, seeds, epochs, metric, recipe) is held fixed."),
            "status": "PENDING second reader (Ash) — the author does not certify his own code",
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    path = REPO / "runs/dyad_guard.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    print("wrote", path, flush=True)
    if a_fail or not b_state_ok or not b_perm_ok:
        raise SystemExit("GUARD FAILED")


def table(args) -> None:
    records = load_records("train")
    t = option_identity_table(records)
    path = REPO / "runs/dyad_identity_table.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(t, indent=1))
    print(json.dumps(t, indent=1))
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/dyad_pilot_seed42")
    ap.add_argument("--ckpt-dir", default="runs/dyad_ckpt")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--opt-batch", type=int, default=8)
    ap.add_argument("--f-device", default="cpu")
    ap.add_argument("--q-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--only-cells", default=None)
    ap.add_argument("--q-cache-file", default=None)
    ap.add_argument("--eval-checkpoint", default=None)
    ap.add_argument("--eval-out", default=None)
    args = ap.parse_args()
    if args.eval_checkpoint:
        evaluate(args); return
    if args.table:
        table(args)
    elif args.guard:
        guard(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
