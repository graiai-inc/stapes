# OpenEvidence Feedback Round 2 (2026-04-08)

## Must Fix Before Submission

### 1. Terminology - "accuracy" vs "recall"
What we compute is recall (proportion of reference medical terms correctly transcribed), not accuracy.
Options: "clinical entity recall", "medical entity recall", or keep "clinical term accuracy" with explicit parenthetical definition.

### 2. Quantitative ROVER structural divergence evidence
Need numbers, not just "examination of transcripts revealed."
Compute: average pairwise edit distance between model outputs per dataset.
Show it differs significantly between OSCE (where ROVER worked) and others.

### 3. Tables still missing
- Table 1: All 14 on-device + 5 cloud, WER + CTA per dataset
- Table 2: ROVER best pairs per dataset, showing dataset-dependence
- Table 3: Cost comparison with projected annual

### 4. ROVER technical description fix
"higher confidence" is wrong for equal-weight ROVER. Fix to describe majority voting / word frequency selection.

## Should Fix

### 5. Cost analysis - devices clinicians already own
NOT "upfront hardware investment." Clinicians already have phones. On-device = zero additional cost.
Add access/equity paragraph: democratizes documentation for resource-limited settings, data sovereignty compliance.

### 6. Drug error mitigation
Why these drugs are hard (phonetic similarity, rare in training corpora).
Actionable: medication reconciliation workflows, formulary cross-reference.

### 7. Abstract reorder - lead with ROVER (most novel)
ROVER finding first, then on-device vs cloud, then drug errors, then costs.

### 8. Conclusion strengthened
End with: "Until medication transcription accuracy improves, clinical deployment requires robust verification workflows."
Add equity angle: "on-device may democratize access for resource-limited settings."

## Consider

### 9. Reposition paper around ROVER finding?
Alternative framing: "Dataset-dependent failure of multi-model fusion in clinical ASR"
More novel, more generalizable, challenges assumptions.
User should decide.

### 10. Supplementary materials
- Example transcripts (ROVER success vs failure)
- Full model specs/checkpoints
- Per-file results
- Drug name error examples with context

### 11. Decision framework formatting
Current paragraph is too dense. Break into bullets or table.
