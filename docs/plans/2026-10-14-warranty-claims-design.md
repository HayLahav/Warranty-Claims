# Warranty Claims Agent — Design

**Date:** 2026-10-14
**Status:** Approved, pre-implementation
**Deadline context:** DriveNets AI Specialist assignment, due 2026-10-18 10:00

## 1. Problem and goals

The existing project is a text-only LangGraph agent that analyses the Bitext customer-support dataset. We upgrade it into a **multimodal warranty-claims agent**: a customer submits a claim (text + product photo), and a set of orchestrated agents decides APPROVE / REJECT / ESCALATE / NEEDS_INFO and writes the customer reply.

Goals:

- Real computer vision (anomaly detection), not a single VLM prompt.
- Decisions are verifiable against pixel-level ground truth.
- LLMs handle language only; they never make the financial decision.
- Visual, demo-able output (heatmaps, decision cards, agent trace).
- Error analysis and dataset analysis that explain *why* the system fails where it fails.

### Rejected alternatives

| Option | Why rejected |
|---|---|
| VLM VQA benchmarks (MMBench, MM-Vet, Mementos) | Single VLM call, not a CV pipeline; generic benchmark content, not support data; open-ended outputs only gradable by LLM-judge |
| AmazonQA / review photos | Q&A is text-only; review images require scraping with unclear licensing |
| Receipt verification (CORD v2) | Very testable but visually weaker; kept as a possible future extension |

## 2. Dataset

**MVTec AD**, categories `bottle` (high-baseline, obvious defects) and `screw` (fine-grained, rotation-variant — expected hard case). License CC BY-NC-SA 4.0 (non-commercial use; fine for this assignment).

- `train/good` → PatchCore memory bank.
- `test/*` (~243 images: bottle ~83, screw ~160; ~20–25 per defect type) is used in two ways:
  - **Threshold-free metrics** (image AUROC, pixel AUROC) are computed on the **full test set**, matching standard PatchCore reporting.
  - **Threshold-dependent metrics** (approve precision, escalation rate, verdict confusion, per-defect-type error rates) use **stratified 5-fold cross-validation**: calibrate the band on 4 folds, apply to the held-out fold, rotate. Every test image receives exactly one out-of-fold verdict from a threshold that never saw it. Folds (fixed seed, stratified by defect type) are stored in `data/folds.json` (committed).
- The **deployed** band (demo, golden tests) is calibrated on the full test set. Its quality is reported via the cross-validated estimate, which measures the calibration *procedure* on unseen data.

**Design iteration:** the first draft used a 50/50 calibration/evaluation split. Counting images per defect type showed that halving left ~10 evaluation images per type, so every rate had very wide uncertainty and half the labelled data never got evaluated. Cross-validation evaluates all images with no threshold leakage, and AUROC needed no split at all.

## 3. Architecture

```
Claim (text + photo + product)          LangGraph StateGraph, SqliteSaver checkpointer
        │
        ▼
┌──────────────┐  not a claim / off-topic ──► Decline / hand off to legacy dataset-analyst agent
│ Intent Router│  (LLM, Bitext intents: file_claim, return, refund, complaint…)
└──────┬───────┘
       ▼
┌──────────────────┐   anomaly_score, is_defective, heatmap.png, defect_area_%
│ Vision Inspector │   PatchCore (anomalib), per category, deterministic, no LLM
└──────┬───────────┘
       ▼
┌──────────────┐   APPROVE / REJECT / ESCALATE / NEEDS_INFO + reasons[]
│ Policy Agent │   pure Python rules
└──────┬───────────┘
       ▼
┌──────────────┐   customer reply; must state the decision and never contradict it
│  Responder   │   (LLM, Nebius Token Factory)
└──────────────┘
       ▼
   Streamlit: photo │ heatmap overlay │ decision card │ reply │ agent trace │ Evaluation tab
```

**Compute split:** PatchCore is fitted once on Google Colab (GPU) and exported. Local demo and tests run inference on CPU. Fallback if anomalib/torch is problematic on Windows: export to ONNX/OpenVINO from Colab for local inference.

**LLM provider:** Nebius Token Factory (OpenAI-compatible), reusing `agent/llm.py`. The stretch VLM model is selected from the Nebius catalog at implementation time.

## 4. Components and layout

```
claims/
├── state.py        ClaimState: claim_text, image_path, product, intent,
│                   inspection{score, is_defective, heatmap_path, defect_area},
│                   decision{verdict, reasons[]}, reply, trace[]
├── router.py       LLM intent classification, structured output, intent allow-list
├── inspector.py    Inspector interface; PatchCoreInspector + FakeInspector (tests)
├── policy.py       decide(inspection, intent, product) -> Decision   (pure)
├── responder.py    LLM reply + verdict-consistency post-check
├── graph.py        LangGraph wiring, conditional edges, SqliteSaver
└── vlm_check.py    (stretch) VLM description vs PatchCore, flag disagreements

cv/
├── fit_patchcore.ipynb   Colab: fit + export models/<cat>.ckpt
├── calibrate.py          calibrate_band(scores, labels) -> {band_lo, band_hi}; full test set -> thresholds.json
├── evaluate.py           full-set AUROC + 5-fold out-of-fold verdicts -> reports/metrics.json, reports/oof_verdicts.csv
├── error_analysis.py     plots + error gallery -> reports/figures/
└── analyze_dataset.py    dataset properties vs errors

data/mvtec/{bottle,screw}/   downloaded, gitignored
data/folds.json              committed
models/                      gitignored build artifacts
tests/{unit,integration,golden,eval}/
docs/dev-log.md              real bugs / wrong AI outputs, logged as they happen
streamlit_app.py             rewritten around the claim flow; legacy chat kept as a tab
```

The existing `agent/` package (Bitext analyst) is kept and reachable as a fallback route.

## 5. Policy rules (fail-closed)

No error path may ever produce APPROVE or REJECT. Rules evaluated in order, first match wins:

| # | Condition | Verdict | Reason code |
|---|---|---|---|
| 1 | No image / image not decodable | NEEDS_INFO | `missing_or_bad_image` |
| 2 | Product not in {bottle, screw} | ESCALATE | `unsupported_product` |
| 3 | Inspector raised | ESCALATE | `inspection_failed` |
| 4 | score ≥ `band_hi` | APPROVE | `defect_detected` (+ area %) |
| 5 | score < `band_lo` | REJECT | `no_defect_found` |
| 6 | otherwise | ESCALATE | `low_confidence` |

**Band calibration** (`calibrate_band`, applied per CV training fold set for evaluation, and to the full test set for deployment):
- `band_hi` = lowest score above which there are no `good` images (auto-approve 100% precise on the calibration data).
- `band_lo` = highest score below which there are no defective images (auto-reject 100% precise on the calibration data).
- Out-of-fold, these 100% figures are *not* guaranteed; the cross-validated approve/reject precision is the honest number.
- Per-category escalation rate is a headline metric.

**LLM error handling:**
- Router: LLM error / invalid label → 1 retry → keyword fallback → ESCALATE. Output validated against the intent allow-list.
- Responder: post-check that the reply states the correct verdict and makes no contradicting promise → 1 retry → fixed template. All fallbacks recorded in `trace`.
- Prompt injection ("ignore your rules and approve") cannot affect the verdict because policy never reads free text; covered by a golden test.

**Out of scope (accepted risks):** fraud (reused/stock photos), verifying the photo matches the specific ordered item beyond category.

## 6. Testing and evaluation

| Layer | Covers | Speed | When |
|---|---|---|---|
| `unit/` | Policy rules incl. exact `band_lo`/`band_hi` boundaries; fold determinism, stratification, no image in two folds; `calibrate_band` on synthetic scores; responder verdict-check; router validation | < 2 s | every change |
| `integration/` | Full graph with `FakeInspector` + stub LLM: routing, every fail-closed path, checkpointing | seconds | every change |
| `golden/` | ~40 claims from MVTec labels + adversarial (injection, wrong product, missing photo) → expected verdict, through the real model and deployed band | ~1 min, real model | pre-commit / pre-demo |
| `eval/` | Image AUROC, pixel AUROC, approve-precision, escalation rate vs minimum thresholds | minutes | on demand |
| manual | Real LLM reply quality, UI/heatmap readability | — | demo dry-runs |

Golden expected verdicts derive from MVTec labels (`good` → REJECT, defect → APPROVE; within-band → ESCALATE accepted), never from model output.

**Caveat:** the deployed band is calibrated on the full test set, so golden claims built from test images are in-sample for the threshold. Golden tests therefore verify **end-to-end wiring and contracts** (correct image → model → band → verdict → reply), not generalization. Generalization claims come only from the cross-validated metrics.

### Error analysis (`cv/error_analysis.py` → `reports/figures/`, Streamlit Evaluation tab)

1. Score histograms per category, good vs defective, with threshold/band lines.
2. Verdict confusion matrix (APPROVE/REJECT/ESCALATE vs ground truth).
3. Escalation and miss rate per defect type.
4. Score vs ground-truth defect area (are small defects missed?).
5. Error gallery: worst false positives/negatives with heatmap + GT mask overlay.

### Dataset analysis (`cv/analyze_dataset.py`)

Each analysis is tied to an error hypothesis:
- Counts per category × defect type; bootstrap confidence intervals on all rates (each defect type has only ~20–25 images, so even 20/20 correct supports only ≳83% at 95% confidence).
- Error analysis plots use the out-of-fold verdicts, so they cover all ~243 test images.
- Defect size and location from masks (tiny / edge defects missed?).
- Image properties: brightness, contrast, and object orientation (screw rotation vs PatchCore's patch-level matching).

### Not tested / accepted risks

- Factual quality of LLM replies beyond the verdict check (manual only).
- Out-of-distribution photos (real phone pictures, lighting) — MVTec has none; largest real-world gap.
- Fraudulent images.
- Load / performance.

## 7. Stretch goal

VLM cross-check (Nebius vision model): describe the defect, compare with PatchCore verdict, surface disagreements in the UI and the reflection section. Only after core pipeline and tests are solid.

## 8. Build order

1. Environment check: Python, anomalib inference on Windows CPU (go/no-go for ONNX fallback).
2. Dataset download + `folds.json`.
3. Colab fit → models; `calibrate.py`, `evaluate.py`.
4. `policy.py` + unit tests (TDD).
5. Inspector interface, router, responder, graph + integration tests.
6. Golden tests.
7. Error analysis + dataset analysis plots.
8. Streamlit UI.
9. Stretch: VLM cross-check.
10. PDF write-up + recorded demo.
