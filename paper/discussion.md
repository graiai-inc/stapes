## Discussion

Our benchmark demonstrates that on-device ASR models achieve clinical term recall within 1 to 3 percentage points of commercial cloud APIs across three clinical conversation datasets, supporting the feasibility of privacy-preserving on-device ASR for clinical documentation.

### Model Selection

No single on-device model dominated across all datasets and metrics. Whisper-turbo achieved the highest clinical term recall on two of three datasets, while parakeet-tdt-0.6b-v2 achieved the lowest WER on two of three datasets. This divergence suggests that model selection for clinical applications should consider domain-specific metrics rather than WER alone, accounting for conversation type, specialty, privacy requirements, cost constraints, and connectivity limitations.

### Clinical Safety

Drug names exhibited the highest error rates across all models, with wellbutrin misrecognized in 60.4% of occurrences and similarly high rates for ramipril, rosuvastatin, amoxicillin, and lisinopril. Prior work found that 5.7 to 8.9 percent of ASR errors in clinical documents are clinically significant, with medication errors carrying the highest harm risk. Current ASR systems cannot be relied upon for accurate medication transcription without downstream verification such as formulary cross-referencing and clinician review of high-risk terms. These failures were consistent across deployment modes, indicating an intrinsic challenge of current ASR technology.

### ROVER Fusion

ROVER fusion exhibited strikingly dataset-dependent behavior: improving WER on every OSCE file (reducing best on-device WER by 7 percentage points, outperforming all cloud APIs) while degrading performance on every file in both other datasets (0 improvements across 128 files and 91 model pairs). The top five models had mean WER of 19.2% on OSCE, 15.7% on PriMock57, and 7.6% on the psychiatric dataset. Pairwise analysis showed similar structural divergence across OSCE (9.0% insertion/deletion rate) and PriMock57 (8.7%), indicating that the fusion outcome was driven by individual model error rates rather than structural dissimilarity. When models are already accurate, alignment overhead exceeds correction benefit. This has implications for any clinical AI system relying on multi-model fusion.

### Privacy, Access, and Regulation

On-device ASR eliminates patient audio transmission, addressing one component of clinical AI privacy concerns, though broader considerations including consent, transparency, data governance, and accountability remain relevant for both deployment modes. Clinical deployment may fall under FDA oversight depending on intended use, requiring ongoing performance monitoring and bias surveillance. The zero-marginal-cost nature of on-device ASR also has equity implications: cloud costs of $2,000 to $20,000 annually may be prohibitive for community health centers, solo practitioners, and clinics in low- and middle-income countries. On-device models run on devices clinicians already own, and may be the only legally compliant option in regions with strict data sovereignty laws.

### Limitations

All three datasets consist of simulated or enacted encounters rather than real patient-clinician conversations, which present additional challenges including overlapping speech, background noise, and non-lexical conversational sounds. No open-license datasets of real clinical conversations currently exist. On-device models were evaluated using a single inference engine (sherpa-onnx); performance may vary under alternative runtimes. The UMLS-based clinical term identification uses a manually curated stopword list that may miss collisions or exclude valid terms, and the span-level scoring may overestimate error rates for longer phrases. One cloud API (Google medical_conversation) exhibited substantially higher WER due to verbatim transcription of disfluencies, illustrating a limitation of WER as a standalone metric. Cloud results reflect early 2026 pricing.

### Future Directions

Future work should evaluate on real clinical recordings such as the Bridge2AI-Voice corpus. The dataset-dependent ROVER failure motivates investigation of confidence-weighted fusion using per-token model probabilities. Specialty-specific benchmarks and evaluation of complete documentation pipelines would provide more clinically meaningful assessments.

