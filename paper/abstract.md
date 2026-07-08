<!-- NOT built into the submission. The authoritative abstract is the "# Abstract"
     block in paper/full_manuscript.md (that is what assemble_submission.py renders).
     This file is a convenience mirror; keep it in sync when the manuscript changes. -->

## Abstract

**Objective:** To determine whether privacy-preserving on-device speech recognition achieves clinical term accuracy comparable to commercial cloud services for ambient clinical documentation, to characterize medication-name recognition failures, and to test whether multi-model fusion narrows the residual accuracy gap.

**Materials and Methods:** We evaluated 12 on-device ASR models and 5 cloud APIs on 400 simulated conversations from three public datasets (OSCE respiratory, n=272; PriMock57 primary care, n=57; Kazi et al. psychiatric, n=71). WER was reported in standard (Whisper normalizer) and meaning-preserving regimes. Clinical term recall (CTR) used UMLS 2025AB spans via QuickUMLS with BCa bootstrap 95% CIs. ROVER and six advanced fusion algorithms were evaluated.

**Results:** On-device CTR was within 1–3 pp of the best cloud service per dataset. The standard-WER gap was 0.5–5.2 pp; the meaning-preserving-WER gap narrowed to 0.7–3.3 pp. Medication names were the highest-error category for most models; the highest-error drugs were misrecognized in 85% to 93% of on-device occurrences versus 3% to 30% in the cloud. A medical-dictation model cut wellbutrin errors to 5%, showing this deficit is addressable through domain adaptation. ROVER gained ≤ 0.82 pp; no advanced algorithm exceeded naive voting by more than 0.4 pp.

**Discussion:** Medication errors reflect look-alike/sound-alike confusability, and their persistence alongside near-parity recall shows that aggregate accuracy can mask clinically critical failure modes.

**Conclusion:** On-device ASR is a clinically viable, lower-cost alternative that keeps patient audio local, and a single model suffices; fusion did not close the small residual gap. Until medication accuracy improves, deployment in either mode requires explicit medication verification before note finalization.
