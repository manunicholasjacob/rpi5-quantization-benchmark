# IEEE ESL Submission Checklist — FINAL v3
## Paper: "INT8 Quantization on Raspberry Pi 5: A Reproducible Benchmark of Latency, Energy, and Tail-Latency Failure Modes"
## Author: Manu Nicholas Jacob | manunicholasjacob@gmail.com | ORCID: 0009-0007-6589-6572

---

## SUBMISSION PORTAL
**URL**: https://ieee.atyponrex.com/journal/les-ieee
**Journal**: IEEE Embedded Systems Letters (ESL)
**Manuscript type**: Research Letter

---

## FORMAT COMPLIANCE
- [x] ≤ 4 pages including all figures, tables, references — **CONFIRMED: 4 pages exactly**
- [x] IEEE transactions two-column format (IEEEtran.cls, `journal` mode)
- [x] 10-pt font (IEEEtran default)
- [x] 7.875 in × 10.75 in trim size (IEEEtran default)
- [x] Maximum 5 keywords — **CONFIRMED: 5 exactly**
- [x] Written in clear, concise English

---

## AUTHOR / AFFILIATION
- [x] Author: Manu Nicholas Jacob
- [x] Affiliation: University of Massachusetts Amherst, dual B.S. EE & CpE (2025)
- [x] Email: manunicholasjacob@gmail.com
- [x] ORCID: 0009-0007-6589-6572
- [x] Single-blind review: author identity DISCLOSED (IEEE ESL policy — authors are identified)
- [x] GitHub repo in footnote: https://github.com/manunicholasjacob/rpi5-quantization-benchmark

---

## CONTENT REQUIREMENTS
- [x] Novel contribution: first empirical characterization of dataset-conditioned INT8 tail-latency failure mode on ARMv8 + ONNX Runtime
- [x] Sharp research questions (Q1, Q2) answered with evidence
- [x] Statistical validation: 300-iteration trials, 95% CIs, Cohen's d > 100, p < 0.001
- [x] Real accuracy numbers in Table 1 (FP32 top-1 from test set)
- [x] Honest limitation: INT8 accuracy not separately measured — stated in table caption and text
- [x] Corrected Q/DQ node counts verified from actual ONNX graphs (MV3-S: 326, R18: 108)
- [x] Cache-pressure mechanism correctly described and appropriately qualified (hypothesis, not directly measured)
- [x] Tail-latency failure shown across ALL batch sizes (1–8) — not just batch=1
- [x] Calibration ablation table (128/512/1024 samples) with real Pi data
- [x] Energy surrogate uncertainty explicitly quantified (±20–60%)
- [x] Scope/limitations section honest and complete
- [x] Reproducibility section with exact artifact URL

---

## IEEE POLICY COMPLIANCE
- [x] No funding conflict (none received)
- [x] Original work, not under review elsewhere

---

## FILES TO UPLOAD AT SUBMISSION
| File | Description |
|------|-------------|
| `manuscript.pdf` | Final compiled PDF (4 pages, 361 KB) |
| `latex_source.zip` | LaTeX source + all 3 figures (main.tex + fig_*.png) |
| `cover_letter.txt` | Cover letter (optional but recommended) |

---

## FIGURES VERIFIED
- [x] `fig_latency_v2.png` — P50 latency comparison with 95% CIs and speedup annotations
- [x] `fig_scaling_tail_v2.png` — Throughput scaling + P90/P50 by batch size (failure persists)
- [x] `fig_edp_v2.png` — EDP reduction factors

---

## REVIEWER SCORECARD (v3)
| Criterion | Score | Max |
|-----------|-------|-----|
| Problem importance | 4 | 4 |
| Novelty | 3 | 5 |
| Literature awareness | 3 | 4 |
| Clear research question | 4 | 4 |
| Claim strength | 4 | 4 |
| Experimental rigor | 4 | 5 |
| Baselines | 3 | 4 |
| Result significance | 4 | 5 |
| Writing/story | 4 | 4 |
| Venue fit | 4 | 4 |
| Distinctness | 3 | 4 |
| Honesty/limitations | 4 | 4 |
| Memorable concept | 2 | 3 |
| Reusable artifact | 3 | 3 |
| Generalizability | 2 | 3 |
| Reviewer excitement | 2 | 3 |
| **TOTAL** | **53/63** | — |
| **Scaled to /100** | **~84** | — |

**Interpretation**: 80+ = strong tier-2 candidate. IEEE ESL is a tier-2 journal. **SUBMIT.**

---

## SUBMISSION STEPS
1. Go to: https://ieee.atyponrex.com/journal/les-ieee
2. Create account or log in
3. Click "Submit a Manuscript"
4. Select type: **Research**
5. Enter title: `INT8 Quantization on Raspberry Pi 5: A Reproducible Benchmark of Latency, Energy, and Tail-Latency Failure Modes`
6. Enter author: Manu Nicholas Jacob, University of Massachusetts Amherst
7. Enter ORCID: 0009-0007-6589-6572
8. Upload `manuscript.pdf` as main document
9. Upload `latex_source.zip` as source files
10. Upload `cover_letter.txt` as cover letter
11. Enter keywords: Edge inference, INT8 quantization, tail latency, Raspberry Pi, ONNX Runtime
12. Complete ethical/conflict declarations (no conflicts, no funding)
13. Submit
