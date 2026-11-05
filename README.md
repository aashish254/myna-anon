# Myna — measurement artifact

Code, frozen data and per-question records for a measurement study of a typed decision engine.
Every number in the paper is recomputed from the files in `runs/`; nothing is transcribed by hand.

## What the study measures

A typed decision contract (`choice`, `score`, boolean `noul`) answered by two engines: a
14.45M-parameter gated-linear-attention streaming model with a constant 576 KiB state, and a
446.63M-parameter fine-tuned DeBERTa-v3-large cross-encoder with a cardinality-routed readout.
The suite is 16 cells over 11 public sources (1,176 test bundles, 1,440 questions).

On that suite the cross-encoder reaches **0.8620** macro (0.8632 overall, 0.8393 joint bundle
exact). Its fair control — the same backbone fine-tuned on the same split with a conventional
per-cell classifier — reaches **0.7931** at six epochs, where the schedule plateaus, so the readout
accounts for **6.9 points**, concentrated on multi-class choice cells.

## Layout

| path | contents |
|---|---|
| `src/myna/` | the engine: contract, streaming trunk, v6 cross-encoder, calibration, serving |
| `bench/` | every harness that produced a table or figure, one per claim |
| `tests/` | the invariants those harnesses rely on (parity, masking, aggregation) |
| `data/decision-v2-pilot/` | the frozen suite: train / development / test JSONL plus manifests |
| `runs/` | committed per-question witnesses; every reported number is read from these |
| `release/` | training notebooks, the reference trainer, the calibration-split builder |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # numpy, torch, transformers
python -m bench.paper_numbers        # Table III, Table V, calibration and selective numbers
python -m bench.paired_bootstrap     # paired intervals and McNemar tests
python -m bench.mondrian_conformal   # released / refit / Mondrian coverage
python -m bench.baseline_suite       # the in-domain DeBERTa control, per seed
```

No GPU is needed for any of the aggregation steps; they read committed records.

## Reproducing each claim

| claim | command | witness |
|---|---|---|
| macro / overall / bundle accuracy | `bench.paper_numbers` | `runs/paper_numbers.json` |
| cardinality routing ablation | `bench.v6_audit --max-packed 1000` | `runs/v6_packed_forced.json` |
| bundle consistency | `bench.consistency_audit` | `runs/consistency_audit.json` |
| conformal coverage | `bench.mondrian_conformal` | `runs/mondrian_conformal.json` |
| calibration independence | `bench.cal_split_check` | `runs/cal_split_check.json` |
| paired differences | `bench.paired_bootstrap` | `runs/paired_bootstrap.json` |
| in-domain control | `bench.baseline_suite` | `runs/baseline_seed*_2ep.json` |
| row-clustered intervals | `bench.paper_bootstrap` | `runs/paper_bootstrap.json` |

The aggregation refuses a record whose correctness flag disagrees with its own prediction and
gold, so a table cannot drift from the records it cites. `bench/reproduce.py --check` re-derives
every quoted figure from its witness and is part of the test suite.

## Data

`decision-v2-pilot` is derived from eleven public datasets; `data/decision-v2-pilot/upstream_manifest.json`
pins each source, revision and split. The suite is English-only and inherits the biases of those
sources. No demographic labels are included, so no fairness audit is possible on it.

## Checkpoint

The trained checkpoint is 1.6 GB of fine-tuned backbone plus 48 MB of readout heads. It is too
large for a code artifact and is hosted separately; during blind review it is available from the
artifact on request, and it will be linked publicly in the camera-ready.
