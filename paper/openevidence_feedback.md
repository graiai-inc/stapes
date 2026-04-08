# OpenEvidence Feedback on Paper Draft (2026-04-08)

## Summary Assessment
Strong, rigorous work. Main vulnerabilities: (1) lack of real clinical data, (2) insufficient clinical safety/harm discussion, (3) limited regulatory/implementation context. ROVER finding is genuinely novel, could be primary focus.

## Critical Issues

### 1. Clinical Validity - Simulated Data Only
- ASR degrades in real environments: overlapping speech, noise, acoustic variability [1,2]
- NEJM AI reviewers will question generalizability
- **Actions:** Strengthen limitation discussion, cite degradation literature, consider pilot on real audio
- Refs: Ng 2025 [1], Alboksmaty 2025 [2], Tran 2023 [3] on non-lexical sounds

### 2. Safety and Clinical Significance of Errors
- 5.7-8.9% of ASR errors are clinically significant (Zhou 2018 [4])
- Drug error rates (60.4% wellbutrin) alarming but lack clinical context
- **Actions:** Add clinical harm subsection, categorize errors by harm potential, cite malpractice lit
- Refs: Zhou 2018 [4], Ghaith 2022 [5]

### 3. Regulatory and Governance Context
- NEJM AI emphasizes responsible AI deployment
- Need FDA/regulatory discussion, monitoring, bias surveillance
- **Actions:** Add regulatory paragraph, reference FDA lifecycle approach
- Refs: Labkoff 2024 [6], Shanmugam 2026 [7], Palmieri 2026 [8], Warraich 2025 [9]

### 4. Privacy Claims Need Nuance
- On-device eliminates transmission, not all privacy concerns
- Consent, transparency, data governance, accountability matter too
- **Actions:** Revise privacy language, discuss consent requirements, BAA limitations
- Refs: Elsayed 2026 [10], Lo 2005 [11], Spector-Bagdady 2023 [12]

## Methodological Concerns

### 5. UMLS Analysis Limitations
- UMLS includes non-clinical terms
- Manual stopword filtering may miss collisions
- Multi-word span scoring may overestimate errors
- **Action:** Add sensitivity analysis with different thresholds/scoring

### 6. ROVER Underexplained
- Structural divergence hypothesis plausible but not empirically validated
- **Actions:** Quantify structural similarity (edit distance between outputs), example transcripts in supplementary, mention confidence-weighted fusion as future work

### 7. Cost Analysis Incomplete
- On-device not truly zero (hardware, maintenance, IT)
- No clinician error-correction time cost
- Cloud costs may decrease
- **Action:** Acknowledge these limitations or provide total cost of ownership

## Writing and Presentation

### 8. Abstract
- Lead with clinical problem (burnout), not technical gap
- Quantify 400 conversations more prominently
- Emphasize ROVER finding more (most novel contribution)

### 9. Introduction
- Add ambient AI burnout data [13,14,15]
- Cite WER range (0.087% to >50%) earlier [1]
- Mention socio-technical risks framework [10]

### 10. Discussion
- Move Google WER discussion to limitations
- Add clinical implementation subsection (workflow, training, error correction)
- Connect to ambient AI mixed results literature [13]

### 11. Tables and Figures (MUST CREATE)
- Comprehensive model comparison table (all 14 on-device + 5 cloud x 3 datasets)
- ROVER fusion visualization (per-file improvement distributions)
- Example medical term error cases

### 12. Terminology
- "Medical term recall" is non-standard -> use "clinical term accuracy"
- Explain ROVER voting mechanism for clinical readers

### 13. Reproducibility
- Specify exact model versions/checkpoints
- Computational requirements (CPU specs, inference time per file)
- Inter-rater reliability for any manual annotations

### 14. Limitations Section Structure
- Separate technical (simulated data, single engine) from methodological (WER limits, UMLS coverage)
- Acknowledge no real-time performance assessment

## Strategic for NEJM AI

### 15. Emphasize Clinical Impact
- Frame as enabling privacy-preserving ambient documentation
- Connect to burnout literature [13,14,15]
- Resource-limited settings, data sovereignty laws

### 16. Decision Framework ("So What?")
- When should clinicians choose on-device vs cloud?
- Privacy requirements (substance abuse, mental health)
- Cost constraints (high-volume practices)
- Connectivity limitations (rural clinics)
- Specialty-specific needs (performance varies by conversation type)

### 17. Future Directions (Expand)
- Real clinical audio evaluation (Bridge2AI-Voice)
- Hybrid architectures (on-device ASR + cloud NLP)
- Specialty-specific benchmarks
- Downstream task integration (note generation, coding)

## New References (15 total)

1. Ng et al. 2025 - ASR clinical documentation systematic review. BMC Med Inform Decis Mak 25(1):236. doi:10.1186/s12911-025-03061-0
2. Alboksmaty et al. 2025 - AI voice-to-text quality of care systematic review. EBioMedicine 118:105861. doi:10.1016/j.ebiom.2025.105861
3. Tran et al. 2023 - Non-lexical sounds ambient documentation. JAMIA 30(4):703-711. doi:10.1093/jamia/ocad001
4. Zhou et al. 2018 - ASR error clinical significance. JAMA Net Open 1(3):e180530. doi:10.1001/jamanetworkopen.2018.0530
5. Ghaith et al. 2022 - Charting malpractice. West J Emerg Med 23(3):412-417. doi:10.5811/westjem.2022.1.53894
6. Labkoff et al. 2024 - AI-enabled CDS recommendations. JAMIA 31(11):2730-2739. doi:10.1093/jamia/ocae209
7. Shanmugam et al. 2026 - AI clinical trial regulatory review. J Clin Med 15(5):1937. doi:10.3390/jcm15051937
8. Palmieri et al. 2026 - Responsible AI guidance. JAMA 335(3):207-208. doi:10.1001/jama.2025.23059
9. Warraich et al. 2025 - FDA AI regulation. JAMA 333(3):241-247. doi:10.1001/jama.2024.21451
10. Elsayed 2026 - Socio-technical risks clinical STT. Int J Med Inform 214:106419. doi:10.1016/j.ijmedinf.2026.106419
11. Lo et al. 2005 - HIPAA professional judgment. JAMA 293(14):1766-71. doi:10.1001/jama.293.14.1766
12. Spector-Bagdady et al. 2023 - Health info collection AHA policy. Circulation 148(13):1061-1069. doi:10.1161/CIR.0000000000001173
13. Stults et al. 2025 - Ambient AI documentation evaluation. JAMA Net Open 8(5):e258614. doi:10.1001/jamanetworkopen.2025.8614
14. Shah et al. 2025 - Ambient AI burnout. JAMIA 32(2):375-380. doi:10.1093/jamia/ocae295
15. Olson et al. 2025 - Ambient AI scribes burnout. JAMA Net Open 8(10):e2534976. doi:10.1001/jamanetworkopen.2025.34976
