# Correction Report: stapes manuscript

Generated 2026-04-14 after discovering post-submission computational bug.
Use this as the checklist for applying corrections to the manuscript.

## Figure regeneration flags

- **Figure 2** (figure2.png/pdf) — MUST regenerate or remove. Visualizes the refuted "ROVER degraded on PriMock57 + psychiatric" claim.
- **Figure 1** — likely needs regeneration. OSCE column of Table 1 changes substantially, so any visualization of Table 1 solo WERs is stale.

## File: `abstract.md` (and identical text in `full_manuscript.md` Abstract section)

### A1 — Results sentence citing affected numbers

**Current:**
> "ROVER fusion reduced the best on-device WER from 17.6% to 10.7% on OSCE data, surpassing all cloud APIs, but degraded performance on both other datasets with no per-file improvement across 128 files and 91 model pairs."

**Corrected:**
> "ROVER fusion reduced the best on-device WER from 11.23% to 11.01% on OSCE data, from 13.85% to 13.03% on primary care, and from 7.30% to 7.15% on psychiatry. On OSCE, the best fused pair did not surpass the best commercial cloud API (Azure, 7.70%)."

### A2 — Remove dataset-dependent novelty claim

**Current:**
> "This dataset-dependent fusion behavior has not been previously reported."

**Action:** DELETE. The finding is refuted; fusion now improves on all three datasets (though by small margins).

### A3 — Conclusions sentence

**Current:**
> "ROVER fusion benefits cannot be assumed across clinical recording conditions, with implications for any multi-model clinical AI system."

**Corrected (optional reframe):**
> "ROVER fusion produced only modest WER improvements (≤ 0.82 percentage points) over the best single on-device model across all three datasets, suggesting that the cost of running two models per encounter may not be justified for clinical deployment."

## File: `introduction.md`

### I1 — Contributions list

**Current:**
> "(3) an evaluation of ROVER hypothesis fusion across all model pairs, revealing dataset-dependent behavior not previously reported;"

**Corrected:**
> "(3) an evaluation of ROVER hypothesis fusion across all model pairs on each dataset;"

## File: `results.md`

### R1 — OSCE best on-device WER

**Current:**
> "On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 17.59%) trailed the best cloud API (14.56%) by 3.03 percentage points."

**Corrected:**
> "On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 11.23%) trailed the best cloud API (Azure, 7.70%) by 3.53 percentage points."

### R3 — Google WER gap

**Current:**
> "Among cloud APIs, one service (Google medical_conversation) exhibited WER 5 to 16 percentage points higher than other cloud APIs, attributable to verbatim transcription of disfluencies (see Limitations)."

**Corrected:** recompute the range per dataset with corrected numbers. Also use "faithful" instead of "verbatim" (Google's higher WER reflects faithful filler transcription, not lower accuracy).

### R4 — ROVER Fusion OSCE + PriMock57 + psychiatric paragraph (the big one)

**Current:**
> "On the OSCE dataset, ROVER substantially improved performance. The best pair (parakeet-tdt-0.6b-v2 + sensevoice, 10.7%) reduced WER by 6.9 percentage points from the best single on-device model (17.6%) and outperformed all five cloud APIs. Fusion improved WER on all 272 files for the top pairs.
>
> On PriMock57 and the psychiatric dataset, ROVER degraded performance for every model pair tested (Figure 2). The best fused WER on PriMock57 (20.77%) was 6.92 percentage points worse than the best single model. On the psychiatric dataset, the best fused WER (17.04%) was 9.74 percentage points worse. No pair achieved a per-file improvement on either dataset (0 of 57 and 0 of 71 files, respectively)."

**Corrected:**
> "Across all three datasets, ROVER fusion produced small improvements over the best single on-device model. On the OSCE dataset, the best pair (parakeet-tdt-0.6b-v2 + sensevoice, 11.01%) improved WER by 0.22 percentage points over the best single on-device model (whisper-distil-v3.5, 11.23%); the best fused result did not surpass the best cloud API (Azure, 7.70%). On PriMock57, the best pair (parakeet-tdt-0.6b-v2 + qwen3-asr, 13.03%) improved WER by 0.82 percentage points over the best single model (parakeet-tdt-0.6b-v2, 13.85%). On the psychiatric dataset, the best pair (parakeet-tdt-0.6b-v2 + sensevoice, 7.15%) improved WER by 0.15 percentage points over the best single model (parakeet-tdt-0.6b-v2, 7.30%)."

### R5 — Dataset-dependent interpretation paragraph

**Current (approximate):**
> "This dataset-dependent behavior has not been previously reported. The top five on-device models had mean WER of 19.2% on the OSCE dataset, 15.7% on PriMock57, and 7.6% on the psychiatric dataset. ROVER improved results only on the dataset where individual models had the highest error rates. Pairwise analysis showed similar structural divergence between model outputs on OSCE (9.0% insertion/deletion rate) and PriMock57 (8.7%), suggesting that the difference in fusion outcome is driven by individual model error rates rather than structural dissimilarity between hypotheses."

**Action:** DELETE the entire paragraph. The causal argument explains a phenomenon that no longer exists.

## File: `discussion.md`

### D1 — ROVER Fusion discussion paragraph

**Current:**
> "ROVER fusion exhibited strikingly dataset-dependent behavior: improving WER on every OSCE file (reducing best on-device WER by 7 percentage points, outperforming all cloud APIs) while degrading performance on every file in both other datasets (0 improvements across 128 files and 91 model pairs). Quantitative analysis revealed that this difference was driven by baseline model accuracy rather than structural divergence... [continues]"

**Corrected:**
> "ROVER fusion produced only small WER improvements over the best single on-device model across all three datasets (−0.22 pp on OSCE, −0.82 pp on PriMock57, −0.15 pp on psychiatry). On OSCE, the best fused on-device pair did not surpass the best cloud API (Azure). Given that fusion requires running two models per encounter, approximately doubling compute, memory footprint, and latency, the marginal WER improvements observed here may not justify the deployment cost for clinical on-device use."

## File: `conclusion.md`

### C1 — ROVER sentence

**Current:**
> "In our evaluation, ROVER fusion improved transcription only where models had the highest baseline error rates and degraded performance where models were already accurate, suggesting that the benefit of multi-model fusion depends on baseline model performance."

**Corrected:**
> "In our evaluation, ROVER fusion produced only small WER improvements (≤ 0.82 percentage points) over the best single on-device model across all three datasets, and on OSCE the best fused on-device pair did not surpass the best commercial cloud API."

## File: `cover_letter.md`

### CL1 — "Second" key finding

**Current:**
> "Second, we report a previously undocumented dataset-dependent failure of ROVER hypothesis fusion: fusion improved transcription on the dataset with the highest baseline error rates but degraded performance on every file across both other datasets, with implications for any clinical AI system relying on multi-model combination."

**Corrected (or delete):**
> "Second, we find that ROVER hypothesis fusion produced only small WER improvements (≤ 0.8 percentage points) over the best single on-device model across all three datasets, and did not surpass the best commercial cloud API on OSCE, suggesting that the added inference cost of multi-model fusion may not be justified for clinical on-device deployment."

## File: `table1_wer.csv` — OSCE column corrections

All solo WER on figshare-osce after reference repair. PriMock57 and Psychiatric columns are UNCHANGED.

| Model | Current OSCE | Corrected OSCE |
|---|---|---|
| Whisper distil-v3.5 | 17.59 | **11.23** |
| Whisper turbo | 17.93 | **12.01** |
| Whisper base.en | 20.18 | **14.27** |
| Qwen3-ASR | 18.74 | **12.37** |
| Parakeet-TDT-0.6b-v2 | 22.93 | **17.06** |
| SenseVoice (no ITN) | 18.99 | **13.06** |
| SenseVoice | 18.75 | **12.80** |
| Paraformer-en | 23.97 | **17.95** |
| NeMo FastConformer | 27.70 | **22.44** |
| NeMo FastConformer int8 | 28.03 | **22.73** |
| Nemotron | 38.82 | **35.59** |
| Zipformer (zh-en) | 39.82 | **33.90** |
| MedASR | 58.31 | **56.45** (270/272 files)* |
| MedASR int8 | 57.85 | **55.94** (270/272 files)* |
| Azure Speech | 14.56 | **7.70** |
| Deepgram Nova-2 Medical | 17.04 | **10.10** |
| AssemblyAI | 16.43 | **10.29** |
| AWS Transcribe Medical† | 14.46† | **8.03†** (50-file subset) |
| Google medical_conversation | 23.08 | **16.53** |

Add a footnote explaining the OSCE column was regenerated after repairing the figshare-osce reference (apostrophes injected via 44-token contraction dictionary).

*MedASR and Zipformer-libriheavy figshare runs had 2 files with empty hypotheses that were skipped; buggy baselines match the paper within 0.3pp.

Draft editor letter text is in `correction_letter_draft.md` in the same directory.

## Key decisions for the authors

1. **Novelty claim.** The paper's third contribution and one of the cover letter's three key findings was the "dataset-dependent ROVER failure." That finding no longer exists. Drop to two contributions or reframe around small-magnitude-of-gains observation.
2. **OSCE cloud comparison framing.** No on-device configuration (solo or fused) beats Azure, Deepgram, or AssemblyAI on OSCE. Materially less favorable framing.
3. **Abstract rewrite scope.** Every paragraph of the abstract touches an affected claim.

## Unchanged (do not touch)

- Solo WERs on PriMock57 and Nazmulkazi
- Table 2 (CTR), Table 3 (cost)
- Methods section
- Limitations, Privacy/Access/Regulation, Future Directions
- Dataset and model descriptions
- `methods.md`, `table2_ctr.csv`, `table3_cost.csv`
