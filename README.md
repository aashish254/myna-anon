# Myna — artifact

Code, data and per-question records for the paper *Myna: Option-Set Cardinality, Not Encoder
Capacity, Bounds Typed-Decision Accuracy*. The submission PDF is the authoritative description;
this directory is the evidence behind its numbers.

## Reviewer quick start (5 minutes, no GPU)

```bash
pip install -e .        # from this directory
make repro              # 49/49 rows: every published figure still matches its committed witness
make gates              # 7/7 verdicts: 5 met, 2 not met — printed with the artifact each rests on
```

Every number in the paper has a JSON witness under `runs/`, so the tables can be audited without
loading a model. `runs/paper_numbers.json` carries the headline macro (0.8620), the per-cell
accuracies and the calibration block; `runs/paired_bootstrap.json` carries every interval quoted in
the paper; `runs/setfit_16cell.json` carries the SetFit baseline (0.6455).

## What is here

| Path | Contents |
|---|---|
| `data/decision-v2-pilot/` | the frozen suite — test 1,176 bundles / 1,440 questions / 16 cells; train 57,904 rows; calibration 448 rows / 568 questions |
| `runs/` | per-question records and JSON witnesses for every arm in the paper |
| `src/myna/` | the two engines behind the typed contract (`choice` / `score` / `noul`) |
| `bench/` | scoring, bootstrap, calibration, conformal and needle harnesses |
| `tests/` | unit and contract tests |
| `release/` | the training and campaign scripts that produced the checkpoints |

## The numbers and where they live

| Paper claim | Witness |
|---|---|
| 0.8620 macro / 0.8632 overall / 0.8393 bundle exact | `runs/paper_numbers.json`, `runs/paper_myna_v6_test.rowoutcomes.json` |
| converged control 0.7950 (2 seeds, 6 epochs) | `runs/baseline_seed21_6ep.json`, `runs/baseline_seed7_6ep.json` |
| routing ablation 0.0345 → 0.6724 on `banking77/intent` | `runs/paper_numbers.json`, `runs/v6_packed_forced.rowoutcomes.json` |
| SetFit baseline 0.6455 | `runs/setfit_16cell.json` |
| paired intervals (B=20000, seed 20261104) | `runs/paired_bootstrap.json` |
| conformal coverage 83.68% at 90% | `runs/mondrian_conformal.json` |
| held-out cells 0.9203 | `runs/ood_c1_dbpedia14.rowoutcomes.json`, `runs/ood_c1_trec.rowoutcomes.json` |
| JEVal probe 0.4287 (out-of-gate) | `runs/jeval_subset_v6.rowoutcomes.json` |
| needle accuracy 0.1500 at 16,384 tokens | `runs/needle_lc1_stageC_final.json` |

## Checkpoints

The weights are not in this artifact: `deberta-v5-finetuned/model.safetensors` is 1.74 GB and the
readout heads add another 50 MB. They are available to reviewers through the review forum, with the
sha256 of each file so a re-run can be verified byte for byte. The reported figures do not require
them — `make repro` re-reads committed witnesses, not models.

## Requirements

Python 3.11 or newer. Scoring and witness checks run on CPU; a GPU is needed only to retrain.

## Licence

Apache-2.0. See `LICENSE`.

## Commands (preserved verbatim from the artifact's own documentation)

Each block below is the exact command that produced, or re-checks, a witness in `runs/`. They are
kept here so that `make repro` can verify every published figure against the command that made it.

```bash
uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 --out runs/v6_rescore_test.json
```

```bash
uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 \
  --out runs/v6_g1_test.json --md runs/v6_g1_test.md \
  --metrics runs/v6_test.metrics.json --rowout runs/myna_v6_test.rowoutcomes.json
uv run python -m myna.report --metrics runs/v6_test.metrics.json --split test \
  --out runs/v6_test.report.json
uv run python bench/significance.py --myna runs/myna_v6_test.rowoutcomes.json \
  --laya runs/laya_0419_test.rowoutcomes.json --report runs/v6_test.report.json \
  --B 20000 --seed 20261031 --out runs/significance_v6.json --md runs/significance_v6.md
```

```bash
uv sync --extra v6      # transformers + safetensors + huggingface_hub
```

```bash
uv run python scripts/upload_to_hf.py --ckpt-dir "myna-v5-prototype 2" --repo-id <anonymous-checkpoint> --dry-run
```

```bash
pip install -e .            # run from the root of this artefact
myna-weights          # → ~/.cache/myna/antiprior_off_s0-weights/{model.pt,tokenizer.json}
```

```bash
gh release download antiprior_off_s0-weights -D ~/.cache/myna/antiprior_off_s0-weights
```

```bash
myna-serve --ckpt ~/.cache/myna/antiprior_off_s0-weights --port 8080
curl -s localhost:8080/v1/predict -H 'content-type: application/json' \
     -d '{"state":"My order never arrived and I want a refund.","questions":{"department":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":["billing","technical","shipping","returns","other"]}}}'
```

```bash
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
```

```bash
uv run python bench/eval_needle.py --ckpt runs/myna-v0 --device cpu --n 16 --seed 0 --lengths 128 1024 4096 8192 16384
```

```bash
uv run python bench/eval_scratch.py --split test --seeds 0 1 --out runs/scratch_decision_v2_test.md --metrics-out runs/scratch_metrics_test.json
```

```bash
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/scratch_metrics_test.json --laya runs/laya_decision_v2_test.json --out runs/report_scratch_vs_laya.json
```

```bash
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx \
    --scan-chunk 16 --n-chunks 4 --suite data/decision-v2-pilot   # → runs/onnx_parity.json
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx-wide --questions 2 --q-len 1152 --scan-chunk 64 --report runs/onnx_parity_widest.json
uv run python bench/browser_expected.py                            # → browser/expected.json
npm ci && npm run selftest                                         # 7 checks, node
npm run g3                                                         # Chrome: bytes, cold, p50, shots
npm run g3 -- --artifact runs/onnx_int8 --out runs/browser_g3_int8.json   # the int8 column
uv run python bench/quantize_int8.py                               # the int8 row above
```

```bash
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 3 --lengths 128 --drift-rows 12 --keep trunk.tok.weight .gate.weight --fp32-out runs/myna-v0-mlx-fp32-keepgate --int8-out runs/myna-v0-mlx-int8-keepgate --out runs/bench_mlx_int8_keepgate.md
uv run python bench/diag_mlx_int8_gem.py --iters 200 --trials 7 --out runs/mlx_int8_gem_probe.md
```

```bash
uv run python -m myna.train --steps 9000 --batch 32 --device mps --out runs/myna-v0
uv run python -m myna.rlcd --ckpt runs/myna-v0 --steps 800 --out runs/myna-v0-rlcd
```

```bash
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split calibration \
  --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio.md
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split calibration \
  --out runs/risk_coverage_v6.md
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split test \
  --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio_test.md
```

```bash
uv run python -m bench.risk_coverage --ckpt runs/myna-v0 --out runs/risk_coverage.md
```

```bash
make repro                                               # or:
uv run python bench/reproduce.py --check     # every row, or it says which one drifted
```

```bash
make gates                                               # or:
uv run python bench/gates.py --check    # every verdict, or it says which one lost its witness
```

```bash
# README architecture table · SPEC §2.1 (params, weights on disk)  [here]
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
# README 'Measured on an Apple M5' · SPEC §2.1 G2  [external-laya]
PYTHONPATH="<laya checkout>" .venv/bin/python -m bench.bench_latency_matched
# README 'Trained v0' · SPEC §2.1 G6  [retrain]
uv run python -m myna.train --steps 9000 --batch 32 --device mps --out runs/myna-v0
# PLAN v1 · TODO 4 (RLCD scoring pass)  [retrain]
uv run python -m myna.rlcd --ckpt runs/myna-v0 --steps 800 --out runs/myna-v0-rlcd
# README abstention table · SPEC §2.2 G5  [here]
uv run python -m bench.risk_coverage --ckpt runs/myna-v0 --out runs/risk_coverage.md
# SPEC §2.2 G4 · README context column  [here]
uv run python bench/eval_needle.py --ckpt runs/myna-v0 --device cpu --n 16 --seed 0 --lengths 128 1024 4096 8192 16384
# SPEC §2.2 G4 · §9.64 (the 4k-trained ladder, replayed from its Release)  [here]
uv run python bench/eval_needle.py --ckpt "$(uv run python -c 'from myna.weights import fetch_weights; print(fetch_weights(tag="lc1-4k-s0-stageC-weights"))')" --device cpu --n 60 --seed 0 --lengths 128 1024 4096 8192 16384 --out runs/needle_lc1_stageC_replay.md
# SPEC §2.2 G6 · §9.65 (the v1b arm on the synthetic test split)  [here]
uv run python bench/eval_synthetic.py --ckpt runs/v1b-kaggle-3600b --device cpu --seed 0 --n-train 4000 --n-eval 1200 --out runs/g6_v1b_synthetic_s0.md
# SPEC §2.1 G1 competitor row · README real-corpus table  [external-laya]
uv run python bench/eval_laya_real.py --split test --n-per-source 40 --seed 0 --out runs/laya_decision_v2_test.json
# SPEC §2.2 G3 (torch↔onnxruntime) · README browser section  [here]
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx --scan-chunk 16 --n-chunks 4 --suite data/decision-v2-pilot
# SPEC §5 P6 6b (the 1,067-token request)  [here]
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx-wide --questions 2 --q-len 1152 --scan-chunk 64 --report runs/onnx_parity_widest.json
# SPEC §5 P6 (the node gate that backs every browser cell)  [external-chrome]
npm run selftest
# README in-tab table, left column · SPEC §2.2 G3  [external-chrome]
npm run g3
# README in-tab table, right column  [external-chrome]
npm run g3 -- --artifact runs/onnx_int8 --out runs/browser_g3_int8.json
# SPEC §5 P6 (the int8 attempt, and the node gate going red)  [here]
uv run python bench/quantize_int8.py
# README 'On an Apple silicon Mac' · SPEC §5 P6 6c  [here]
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
# SPEC §5 P6 6c (the one row that flips when the gates are kept fp32)  [here]
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 3 --lengths 128 --drift-rows 12 --keep trunk.tok.weight .gate.weight --fp32-out runs/myna-v0-mlx-fp32-keepgate --int8-out runs/myna-v0-mlx-int8-keepgate --out runs/bench_mlx_int8_keepgate.md
# README 'and it prices that at the kernel' · SPEC §5 P6 6c  [here]
uv run python bench/diag_mlx_int8_gem.py --iters 200 --trials 7 --out runs/mlx_int8_gem_probe.md
# SPEC §4.2 (the majority-label floors the accuracy gates are set against)  [here]
uv run pytest tests/test_report.py -q
# SPEC §5 P3 3c (the Kaggle image is python 3.12/3.11, this box is 3.13)  [here]
uv run python bench/check_python311.py
# README upstream table, control row · SPEC §8's disclosure  [here]
uv run python bench/eval_scratch.py --split test --seeds 0 1 --out runs/scratch_decision_v2_test.md --metrics-out runs/scratch_metrics_test.json
# README upstream table, the gap line · SPEC §8  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/scratch_metrics_test.json --laya runs/laya_decision_v2_test.json --out runs/report_scratch_vs_laya.json
# README upstream table's disclosure row · SPEC §5 P1, §9.23  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/myna-v1-rich/metrics.json --laya runs/laya_decision_v2_test.json --out runs/report_void_vs_laya.json
# SPEC §2.1 G1 myna row · TODO 3i  [gated-kaggle]
# no local command: the run this row named is now committed as `v1b-kaggle-macro`, so G1's verdict no longer rests here.
# SPEC §5 P9 9e (the `ce` vs `emd` verdict) · §9.35, §9.36  [retrain]
python bench/ordinal_ab.py --ab-dir /tmp/ab2
# SPEC §2.1 G1 myna row · §5 P8 · KAGGLE_LAUNCH_INSTRUCTIONS.md  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/v1b_kaggle_3600b.metrics.json --laya runs/laya_decision_v2_test.json --out runs/v1b_kaggle_3600b.report.json
# SPEC §5 P10 10c (the fresh `off` control) · §9.53(iii)  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/antiprior_off_s0.metrics.json --laya runs/laya_decision_v2_test.json --out runs/antiprior_off_s0.report.json
# SPEC §5 P10 10c (Tier 0 over the `off` control) · §9.55(iii)  [retrain]
uv run python bench/diag_question_ablation.py --run-dir runs/antiprior_off_s0 --metrics runs/antiprior_off_s0.metrics.json --out runs/tier0_antiprior_off_s0.json
# SPEC §5 P10 10c (the `on` arm, dose-matched) · §9.56  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/antiprior_on_s0c.metrics.json --laya runs/laya_decision_v2_test.json --out runs/antiprior_on_s0c.report.json
# SPEC §5 P10 10c (Tier 0 over the `on` arm) · §9.56  [retrain]
uv run python bench/diag_question_ablation.py --run-dir runs/antiprior_on_s0c --metrics runs/antiprior_on_s0c.metrics.json --out runs/tier0_antiprior_on_s0c.json
# SPEC §9.41 (the dbpedia14 dev→test delta) · TODO 9h  [here]
uv run python -m myna.report --split dev --metrics runs/v1b_kaggle_3600b.metrics.json --out runs/v1b_kaggle_3600b.dev.report.json > runs/v1b_kaggle_3600b.dev.report.log
# SPEC §5 P9 9d · TODO 9d  [here]
uv run python -m bench.scope_pricing
# KAGGLE_LAUNCH_INSTRUCTIONS.md cost table · kaggle/campaign.py  [here]
grep -E "^step" runs/v1b_kaggle_3600b.train.log | tail -1
# SPEC §2.1 G1 · §9.38 · KAGGLE_LAUNCH_INSTRUCTIONS.md dose lane  [here]
grep -E "dev-mid acc" runs/v1b_kaggle_3600b.train.log | uniq | tail -5
# SPEC §5 P10 · `campaign.py`'s P10 pair · TODO 10  [here]
uv run python bench/anti_prior_audit.py --compare --out runs/anti_prior_audit.json
# SPEC §5 P10 9l (§9.48) — Tier 0 input ablation  [retrain]
uv run python bench/diag_question_ablation.py --run-dir runs/v1b-kaggle-3600b --out runs/diag_question_ablation.json
# SPEC §9.66 — myna's per-question outcomes over the 1,176 test rows  [here]
uv run python bench/row_outcomes.py --out runs/myna_v1b_test.rowoutcomes.json
# SPEC §9.66 — laya 0.4.19's outcomes over the same rows  [external-laya]
PYTHONPATH="<laya 0.4.19 checkout>" "<laya venv>/bin/python" bench/eval_laya_row_outcomes.py --out runs/laya_0419_test.rowoutcomes.json
# SPEC §2.2 G1 · §9.66 — every headline accuracy with a 95% interval  [here]
uv run python bench/significance.py --myna runs/myna_v1b_test.rowoutcomes.json --laya runs/laya_0419_test.rowoutcomes.json --B 20000 --seed 20261031 --out runs/significance.json --md runs/significance.md
# SPEC §9.67 — vegaml 0.15.0's outcomes over the same rows  [external-vegaml]
"<vegaml 0.15.0 venv>/bin/python" bench/eval_vegaml_row_outcomes.py --every 100 --out runs/vegaml_0150_test.rowoutcomes.json
# SPEC §2.2 G1 · §9.67 — the control in the frozen instrument's competitor slot  [here]
uv run python bench/vegaml_control.py --control runs/vegaml_0150_test.rowoutcomes.json --B 20000 --seed 20261031 --out runs/vegaml_control.json --md runs/vegaml_control.md
# README v6 table · the local re-score of the shipped checkpoint  [here]
uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 --out runs/paper_v6_audit.json --md runs/paper_v6_audit.md --metrics runs/paper_v6_test.metrics.json --rowout runs/paper_myna_v6_test.rowoutcomes.json
# SPEC §2.2 G5 · README abstention table (v6, all pooled sources)  [here]
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split calibration --out runs/risk_coverage_v6.md
# SPEC §2.2 G5 · README abstention table (G5's three named sources)  [here]
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split calibration --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio.md
# SPEC §2.2 G5 · README v6 abstention table (the same trio curve out-of-sample, on `test.jsonl`)  [here]
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split test --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio_test.md
# SPEC §2.2 G5 · SPEC §9.69(iv) (the same trio curve on a third split, where it is harder and still passes)  [here]
uv run python -m bench.risk_coverage --ckpt "myna-v5-prototype 2" --split development --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio_dev.md
# SPEC §2.2 G1 · README v6 gate paragraph (the dumps the frozen instrument reads)  [here]
uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 --out runs/v6_g1_test.json --md runs/v6_g1_test.md --metrics runs/v6_test.metrics.json --rowout runs/myna_v6_test.rowoutcomes.json
# SPEC §2.2 G1 · README v6 gate paragraph (the frozen report's own verdict on v6's rows)  [here]
uv run python -m myna.report --metrics runs/v6_test.metrics.json --split test --out runs/v6_test.report.json
# SPEC §2.2 G1 · README v6 gate paragraph · benchmarks.html engines table (G1's number with its own sampling error)  [here]
uv run python bench/significance.py --myna runs/myna_v6_test.rowoutcomes.json --laya runs/laya_0419_test.rowoutcomes.json --report runs/v6_test.report.json --B 20000 --seed 20261031 --out runs/significance_v6.json --md runs/significance_v6.md
```

```bash
uv sync --extra browser
uv run python -m myna.train          # 4,000 steps by default; v0's 9,000 took 15,693 s on an M5
uv run pytest                        # equivalence + isolation proofs
```

```bash
uv run --extra serve python -m myna.serve --ckpt runs/myna-v0 --port 8080
# --abstain-below 0.693 runs the server with a floor under its answers; /v1/health
# reports the value in force, so a deployment cannot forget its own policy
```
