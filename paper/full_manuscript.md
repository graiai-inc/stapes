
# Abstract

Background: Clinical documentation consumes over half of physicians' working hours and is a leading driver of burnout. Ambient AI scribes that transcribe clinical encounters depend on accurate speech recognition, yet the accuracy of privacy-preserving on-device models has not been benchmarked against commercial cloud services on medical conversation data.

Methods: We evaluated 14 general-purpose on-device ASR models and 5 commercial cloud APIs (3 of which offered medical-specific models) across 400 clinical conversations (~80 hours) from three publicly available datasets spanning OSCE examinations, primary care, and psychiatry. We measured word error rate (WER) using standardized text normalization, clinical term recall using UMLS concept matching (2025AB Metathesaurus), ROVER hypothesis fusion across 91 model pairs per dataset, and cloud API costs.

Results: On-device models achieved clinical term recall within 1 to 3 percentage points of cloud APIs (best on-device: 93.6% vs. best cloud: 95.5% on OSCE; 89.4% vs. 92.4% on primary care; 92.0% vs. 93.0% on psychiatry). ROVER fusion reduced the best on-device WER from 17.6% to 10.7% on OSCE data, surpassing all cloud APIs, but degraded performance on both other datasets with no per-file improvement across 128 files and 91 model pairs. This dataset-dependent fusion behavior has not been previously reported. Drug names exhibited the highest error rates across all models regardless of deployment mode. Cloud costs for the benchmark ranged from $19 to $383; on-device inference required no per-encounter expenditure.

Conclusions: On-device ASR approaches cloud-level clinical term recall while eliminating patient audio transmission and recurring costs. ROVER fusion benefits cannot be assumed across clinical recording conditions, with implications for any multi-model clinical AI system. All evaluation code, model results, and analysis scripts are publicly available.

# Introduction

Physicians spend more than half their working hours on documentation, and administrative burden is a leading contributor to burnout. Ambient AI scribes, systems that passively transcribe clinical encounters for automated note generation, have emerged as a promising intervention. Recent evaluations report reductions in after-hours documentation time and improvements in clinician satisfaction, though results are mixed: one large cohort study of a commercial ambient scribe found no significant time savings and worsened after-hours EHR use. Regardless of their downstream effectiveness, all ambient documentation systems depend on the foundational accuracy of automatic speech recognition (ASR) for clinical conversations, where reported word error rates range from under 1% for controlled dictation to over 50% for naturalistic clinical speech.

Commercial cloud-based ASR services now offer medical-specific models, but these require transmitting patient audio to remote servers. Some ambient scribe services retain full audio recordings on remote servers for weeks after the encounter. While Business Associate Agreements (BAAs) address HIPAA liability, they do not eliminate the privacy risks of transmitting and storing protected health information with third parties, and recent analyses have raised broader concerns about consent, transparency, and data governance that extend beyond transmission security alone. On-device ASR, which processes audio locally without network transmission or third-party storage, addresses these concerns. Open-source ASR models have improved rapidly, with several now available in formats suitable for mobile and edge deployment. However, no standardized benchmark currently compares these on-device models against commercial cloud APIs on clinical conversation data.

Existing evaluations of medical ASR have significant limitations. Industry benchmarks report WER on proprietary datasets that cannot be independently verified. Afonja et al. introduced a medical word error rate for accented clinical speech but relied on a private dataset and a commercial NER service for entity identification, limiting reproducibility. The systematic review by Ng et al. noted the absence of a standardized evaluation framework for clinical ASR. More broadly, the socio-technical risks of clinical speech-to-text systems, including reliability failures and the absence of transparency in error reporting, remain largely unaddressed in the evaluation literature.

We present the first open, reproducible benchmark for medical conversation ASR. Our contributions are: (1) a standardized evaluation of 14 on-device models and 5 cloud APIs across three publicly available clinical conversation datasets totaling 400 conversations; (2) a clinical term recall analysis using the Unified Medical Language System (UMLS) to assess clinically relevant transcription errors; (3) an evaluation of ROVER hypothesis fusion across all model pairs, revealing dataset-dependent behavior not previously reported; and (4) a cost analysis comparing cloud API expenditure against zero-marginal-cost on-device inference. All code, results, and evaluation scripts are publicly available.

# Methods


## Datasets

We evaluated three publicly available English-language clinical conversation datasets covering different medical specialties and recording conditions.

The OSCE respiratory interview dataset (Fareez et al., Scientific Data 2022) contains 272 simulated patient-physician interviews focused on respiratory cases, recorded in a controlled OSCE examination setting. Audio files and manually corrected transcripts are available under a CC-BY license on figshare.

PriMock57 (Korfiatis et al., ACL 2022) contains 57 mock primary care consultations recorded between medical professionals acting as doctor and patient. The dataset includes audio recordings, utterance-level transcripts, and consultation notes, released under a CC-BY license.

The Kazi et al. psychiatric dataset (Zenodo/GitHub) contains 71 enacted psychiatric consultations generated by pairs of students reading from clinical transcripts. Audio was recorded and released under a CC-BY 4.0 license.

All three datasets consist of simulated or enacted encounters rather than real clinical conversations. No open-license datasets of real patient-clinician conversations currently exist, a systemic gap in medical AI research that limits the ecological validity of all ASR benchmarking efforts in this domain.


## On-Device Models

We evaluated 14 open-source ASR models spanning four architectural families: encoder-decoder (Whisper distil-v3.5, Whisper turbo, Whisper base.en, Qwen3-ASR), transducer (parakeet-tdt-0.6b-v2, NeMo FastConformer, NeMo FastConformer int8, Nemotron, Zipformer), CTC (SenseVoice, SenseVoice without ITN, MedASR, MedASR int8), and non-autoregressive (Paraformer-en). Model parameter counts ranged from 70M (Zipformer) to 809M (Whisper turbo). SenseVoice was evaluated in two configurations: with and without inverse text normalization (ITN). MedASR was evaluated in both full-precision and int8-quantized forms, as was NeMo FastConformer.

All 14 on-device models are general-purpose and were not fine-tuned for medical or clinical speech. All were run through the sherpa-onnx inference engine (v1.12.34), an open-source runtime that executes ONNX-format models on CPU without GPU acceleration, designed for mobile and edge deployment. Benchmarks were run on a Lenovo ThinkPad P14s Gen 4 laptop (AMD Ryzen 7 PRO 7840U, 64 GB RAM, Ubuntu 24.04 LTS); inference was faster than real-time for all models. The same 14 models were evaluated on all three datasets to ensure consistent comparison.


## Cloud APIs

Five commercial cloud ASR services were evaluated: Azure Speech, Google Cloud Speech (medical_conversation model), Deepgram Nova-2 Medical, AssemblyAI (universal-3-pro), and AWS Transcribe Medical. Three of five services (Google, Deepgram, AWS) used medical-specific models; Azure and AssemblyAI used general-purpose models. Azure, Google, Deepgram, and AssemblyAI were run on all files across all three datasets. AWS Transcribe Medical was run on representative subsets (50, 17, and 21 files from each dataset) due to cost.


## Text Normalization and WER Computation

Word error rate was computed after applying the Whisper English text normalizer (whisper-normalizer v1.0), which performs case folding, punctuation removal, number-to-word conversion, and contraction expansion. This normalizer is the standard pipeline used by the HuggingFace Open ASR Leaderboard (Gandhi et al., 2025) and MLPerf Inference ASR benchmarks. Both raw and normalized WER were computed for all model-file pairs; normalized WER is reported throughout. Because we evaluated a fixed, exhaustive set of models and files rather than sampling from a population, we report descriptive performance metrics without inferential statistics.


## ROVER Fusion

Recognizer Output Voting Error Reduction (ROVER) (Fiscus, 1997) combines two transcription hypotheses by aligning them at the word level and selecting the most frequently occurring word at each position through majority voting. When two models agree on a word, it is retained; when they disagree, the system selects the word that produces better alignment with the surrounding context. We applied ROVER to all pairwise combinations of the 14 on-device models (91 pairs) on each dataset, using equal model weighting and word-level similarity scoring for alignment. Fused WER was computed using the same normalization pipeline. Per-file win counts (files where fused WER was lower than both individual models) were recorded.


## Clinical Term Recall

To assess clinical relevance beyond aggregate WER, we measured the accuracy of medical terminology transcription using the Unified Medical Language System (UMLS). For each reference transcript, we identified all medical concept spans using QuickUMLS (medspacy fork, threshold 0.8) against the UMLS 2025AB Metathesaurus, restricted to English and to clinically relevant semantic types (drugs, conditions, symptoms, findings, procedures, anatomy, and lab results). QuickUMLS performs approximate string matching against the full UMLS vocabulary and returns both single-word terms (e.g., "hypertension") and multi-word phrases (e.g., "chest pain", "past medical history") with their character-level positions. Common English words that coincidentally match UMLS abbreviations or eponyms were excluded via a curated stopword list (e.g., "said" matches Simian AIDS, "still" matches Still's Disease). We then aligned each model's hypothesis to the reference using jiwer (which uses the rapidfuzz library for efficient Levenshtein alignment) and determined which reference word positions contained errors. A medical concept span was scored as an error if any word within the span was incorrectly transcribed. Clinical term recall was computed per model as the proportion of medical concept spans correctly transcribed.


## Cost Tracking

Cloud API costs were recorded from each provider's billing dashboard after all benchmark runs were complete. On-device inference costs were considered zero per encounter, as models run on local hardware with no API fees. Annual cost projections were estimated based on published per-minute or per-request pricing for a hypothetical practice performing 30 encounters per day.

# Results


## On-Device vs. Cloud ASR Performance

On-device models approached or matched cloud API performance across all three datasets (Table 1, Figure 1). On the Kazi et al. psychiatric dataset (71 conversations), the best on-device model (parakeet-tdt-0.6b-v2, 7.30%) was within 0.47 percentage points of the best cloud API (6.83%) and outperformed two of five cloud services. On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 17.59%) trailed the best cloud API (14.56%) by 3.03 percentage points. On PriMock57 (57 mock primary care consultations), the gap was widest: the best on-device model (parakeet-tdt-0.6b-v2, 13.85%) trailed the best cloud API (9.88%) by 3.97 percentage points.

No single on-device model performed best across all datasets. Whisper-distil-v3.5 led on the OSCE dataset while parakeet-tdt-0.6b-v2 led on both PriMock57 and the psychiatric dataset. Among cloud APIs, one service (Google medical_conversation) exhibited WER 5 to 16 percentage points higher than other cloud APIs, attributable to verbatim transcription of disfluencies (see Limitations).


## ROVER Fusion

ROVER (Recognizer Output Voting Error Reduction) was evaluated across all pairwise combinations of the 14 on-device models on each dataset (91 pairs per dataset).

On the OSCE dataset, ROVER substantially improved performance. The best pair (parakeet-tdt-0.6b-v2 + sensevoice, 10.7%) reduced WER by 6.9 percentage points from the best single on-device model (17.6%) and outperformed all five cloud APIs. Fusion improved WER on all 272 files for the top pairs.

On PriMock57 and the psychiatric dataset, ROVER degraded performance for every model pair tested (Figure 2). The best fused WER on PriMock57 (20.77%) was 6.92 percentage points worse than the best single model. On the psychiatric dataset, the best fused WER (17.04%) was 9.74 percentage points worse. No pair achieved a per-file improvement on either dataset (0 of 57 and 0 of 71 files, respectively).

This dataset-dependent behavior has not been previously reported. The top five on-device models had mean WER of 19.2% on the OSCE dataset, 15.7% on PriMock57, and 7.6% on the psychiatric dataset. ROVER improved results only on the dataset where individual models had the highest error rates. Pairwise analysis showed similar structural divergence between model outputs on OSCE (9.0% insertion/deletion rate) and PriMock57 (8.7%), suggesting that the difference in fusion outcome is driven by individual model error rates rather than structural dissimilarity between hypotheses.


## Clinical Term Recall

On-device models achieved clinical term recall (CTR) within 1 to 3 percentage points of cloud APIs across all datasets (Table 2). On the psychiatric dataset, the best on-device models (whisper-turbo 92.0%, whisper-distil 91.9%, parakeet-tdt 91.8%) were within 1 percentage point of the best cloud API (93.0%). On the OSCE dataset, the best on-device model (whisper-turbo, 93.6%) trailed the best cloud APIs (95.5%) by 1.9 percentage points. On PriMock57, the gap was widest: parakeet-tdt (89.4%) trailed the best cloud API (92.4%) by 3.0 percentage points.

Model rankings differed between WER and CTR. Whisper-turbo, which ranked second on overall WER, achieved the highest on-device CTR on two of three datasets. Drug names exhibited the highest per-term error rates across both on-device and cloud models, including wellbutrin (60.4% error rate on the psychiatric dataset), and common medications and symptoms on the primary care and OSCE datasets. These failures were consistent across model types, suggesting that medical terminology remains an intrinsic challenge for current ASR systems regardless of deployment mode.


## Cost Analysis

Total cloud API costs for evaluating approximately 80 hours of audio across all three datasets ranged from $19.33 to $382.64 (Table 3). The five services ranked by total cost were: Azure Speech ($19.33), Deepgram Nova-2 Medical ($25.78), AssemblyAI ($28.10), AWS Transcribe Medical ($99.63, representative subsets only due to cost), and Google Cloud Speech ($382.64). On-device inference incurred no per-encounter cost after initial model download (model sizes range from 37 MB to 1.5 GB). On-device processing runs on consumer mobile devices that clinicians already own and regularly update, requiring no additional hardware investment. For a practice transcribing 30 encounters per day, estimated annual cloud costs at published rates would range from approximately $2,000 to over $20,000 depending on the service, whereas on-device processing has zero marginal cost per encounter.

# Discussion

Our benchmark demonstrates that on-device ASR models achieve clinical term recall within 1 to 3 percentage points of commercial cloud APIs across three clinical conversation datasets, supporting the feasibility of privacy-preserving on-device ASR for clinical documentation. This performance was achieved using general-purpose models with no medical fine-tuning, compared against cloud APIs where three of five services deployed medical-specific models, suggesting that domain-adapted on-device models could further narrow or eliminate the gap.


## Model Selection

No single on-device model dominated across all datasets and metrics. Whisper-turbo achieved the highest clinical term recall on two of three datasets, while parakeet-tdt-0.6b-v2 achieved the lowest WER on two of three datasets. This divergence suggests that model selection for clinical applications should consider domain-specific metrics rather than WER alone, accounting for conversation type, specialty, privacy requirements, cost constraints, and connectivity limitations.


## Clinical Safety

Drug names exhibited the highest error rates across all models, with wellbutrin misrecognized in 60.4% of occurrences and similarly high rates for ramipril, rosuvastatin, amoxicillin, and lisinopril. Prior work found that 5.7 to 8.9 percent of ASR errors in clinical documents are clinically significant, with medication errors carrying the highest harm risk. Current ASR systems cannot be relied upon for accurate medication transcription without downstream verification. Specific mitigations include automatic cross-referencing of ASR output against patient medication lists and institutional formularies, flagging phonetically similar drug names for mandatory clinician review, and requiring explicit confirmation before finalizing notes containing high-risk medications. These safeguards should be implemented regardless of whether on-device or cloud ASR is used, as the failures were consistent across deployment modes.


## ROVER Fusion

ROVER fusion exhibited strikingly dataset-dependent behavior: improving WER on every OSCE file (reducing best on-device WER by 7 percentage points, outperforming all cloud APIs) while degrading performance on every file in both other datasets (0 improvements across 128 files and 91 model pairs). Quantitative analysis revealed that this difference was driven by baseline model accuracy rather than structural divergence: pairwise insertion/deletion rates (the proportion of words added or removed in alignment between two model hypotheses) were similar across OSCE (9.0%) and PriMock57 (8.7%), but mean model WER differed substantially (19.2% vs. 15.7% vs. 7.6%). When models are already accurate, there are fewer errors to fix through voting, but alignment artifacts are still introduced, producing a net increase in errors. This has implications for any clinical AI system relying on multi-model fusion: the benefit of hypothesis combination depends on baseline model performance and cannot be assumed without validation on representative data.


## Privacy, Access, and Regulation

On-device ASR eliminates patient audio transmission, addressing one component of clinical AI privacy concerns, though broader considerations including consent, transparency, data governance, and accountability remain relevant for both deployment modes. For privacy-sensitive specialties such as psychiatry, substance abuse treatment, and reproductive health, eliminating audio transmission may be a prerequisite for patient acceptance and regulatory compliance regardless of BAA protections. Clinical deployment may fall under FDA oversight depending on intended use, requiring ongoing performance monitoring and bias surveillance. The zero-marginal-cost nature of on-device ASR also has equity implications. A 2026 national study found that ambient AI adoption in U.S. hospitals is concentrated among institutions with stronger operating margins, larger size, and metropolitan location, with lower adoption in under-resourced settings. Cloud costs of $2,000 to $20,000 annually may reinforce this disparity. On-device models run on devices clinicians already own, eliminating cost as a barrier to adoption and potentially serving as the only legally compliant option in regions with strict data sovereignty laws.


## Limitations

All three datasets consist of simulated or enacted encounters rather than real patient-clinician conversations, which present additional challenges including overlapping speech, background noise, and non-lexical conversational sounds. No open-license datasets of real clinical conversations currently exist. On-device models were evaluated using a single inference engine (sherpa-onnx); performance may vary under alternative runtimes. The UMLS-based clinical term identification uses a manually curated stopword list that may miss collisions or exclude valid terms, and the span-level scoring may overestimate error rates for longer phrases. One cloud API (Google medical_conversation) exhibited substantially higher WER due to verbatim transcription of disfluencies, illustrating a limitation of WER as a standalone metric. Cloud results reflect early 2026 pricing.


## Future Directions

Future work should evaluate on real clinical recordings such as the Bridge2AI-Voice corpus. The data gap itself is addressable without patient data: OSCE examinations are conducted routinely at medical schools worldwide, are clinically realistic, involve students and evaluators with diverse accents and languages, and contain no protected health information. If institutions shared even a fraction of these recordings under open licenses, the research community would have access to a large, diverse, and ecologically valid clinical ASR benchmark. The barrier is not technical or ethical but institutional willingness. The dataset-dependent ROVER failure motivates investigation of confidence-weighted fusion using per-token model probabilities. Specialty-specific benchmarks and evaluation of complete documentation pipelines would provide more clinically meaningful assessments.

# Conclusion

On-device ASR models approach the clinical term recall of commercial cloud APIs while eliminating patient audio transmission and recurring costs. In our evaluation, ROVER fusion improved transcription only where models had the highest baseline error rates and degraded performance where models were already accurate, suggesting that the benefit of multi-model fusion depends on baseline model performance. Until medication transcription accuracy improves, clinical deployment of any ASR system requires robust verification workflows. By running on devices clinicians already own, on-device ASR may democratize access to documentation technology for settings where cloud services are unaffordable or legally prohibited, while addressing privacy concerns paramount in sensitive clinical specialties.


# Data and Code Availability

All evaluation code, model inference scripts, ROVER fusion implementation, and per-file results are publicly available at github.com/graiai-inc/stapes. The three clinical conversation datasets are available under CC-BY licenses at their respective repositories. UMLS 2025AB requires a free license from the National Library of Medicine.
