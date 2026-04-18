
# Abstract

Background: Clinical documentation consumes over half of physicians' working hours and is a leading driver of burnout. Ambient AI scribes that transcribe clinical encounters depend on accurate speech recognition, yet the accuracy of privacy-preserving on-device models has not been benchmarked against commercial cloud services on medical conversation data.

Methods: We evaluated 14 general-purpose on-device ASR models and 5 commercial cloud APIs (3 of which offered medical-specific models) across 400 clinical conversations (~80 hours) from three publicly available datasets spanning OSCE examinations, primary care, and psychiatry. We measured word error rate (WER) using standardized text normalization, clinical term recall using UMLS concept matching (2025AB Metathesaurus), ROVER hypothesis fusion across 91 model pairs per dataset, and cloud API costs.

Results: On-device models achieved clinical term recall within 1 to 3 percentage points of cloud APIs (best on-device: 93.6% vs. best cloud: 95.5% on OSCE; 89.4% vs. 92.4% on primary care; 92.0% vs. 93.0% on psychiatry). ROVER fusion produced small WER improvements over the best single on-device model across all three datasets: from 11.23% to 11.01% on OSCE, from 13.85% to 13.03% on primary care, and from 7.30% to 7.15% on psychiatry. On OSCE, the best fused on-device pair did not surpass the best cloud API. An exhaustive search of all pairwise and three-model combinations confirmed that adding a third model consistently degraded performance relative to the best pair. Drug names exhibited the highest error rates across all models regardless of deployment mode. Cloud costs for the benchmark ranged from $19 to $383; on-device inference required no per-encounter expenditure.

Conclusions: On-device ASR approaches cloud-level clinical term recall while eliminating patient audio transmission and recurring costs. ROVER fusion produced only modest improvements (≤ 0.82 percentage points) over the best single model, and three-model combinations offered no additional benefit, suggesting that the added inference cost of multi-model fusion may not be justified for clinical on-device deployment. All evaluation code, model results, and analysis scripts are publicly available.

# Introduction

Physicians spend more than half their working hours on documentation, and administrative burden is a leading contributor to burnout. Ambient AI scribes, systems that passively transcribe clinical encounters for automated note generation, have emerged as a promising intervention. Recent evaluations report reductions in after-hours documentation time and improvements in clinician satisfaction, though results are mixed: one large cohort study of a commercial ambient scribe found no significant time savings and worsened after-hours EHR use. Regardless of their downstream effectiveness, all ambient documentation systems depend on the foundational accuracy of automatic speech recognition (ASR) for clinical conversations, where reported word error rates range from under 1% for controlled dictation to over 50% for naturalistic clinical speech.

Commercial cloud-based ASR services now offer medical-specific models, but these require transmitting patient audio to remote servers. Some ambient scribe services retain full audio recordings on remote servers for weeks after the encounter. While Business Associate Agreements (BAAs) address HIPAA liability, they do not eliminate the privacy risks of transmitting and storing protected health information with third parties, and recent analyses have raised broader concerns about consent, transparency, and data governance that extend beyond transmission security alone. On-device ASR, which processes audio locally without network transmission or third-party storage, addresses these concerns. Open-source ASR models have improved rapidly, with several now available in formats suitable for mobile and edge deployment. However, no standardized benchmark currently compares these on-device models against commercial cloud APIs on clinical conversation data.

Existing evaluations of medical ASR have significant limitations. Industry benchmarks report WER on proprietary datasets that cannot be independently verified. Afonja et al. introduced a medical word error rate for accented clinical speech but relied on a private dataset and a commercial NER service for entity identification, limiting reproducibility. The systematic review by Ng et al. noted the absence of a standardized evaluation framework for clinical ASR. More broadly, the socio-technical risks of clinical speech-to-text systems, including reliability failures and the absence of transparency in error reporting, remain largely unaddressed in the evaluation literature.

We present the first open, reproducible benchmark for medical conversation ASR. Our contributions are: (1) a standardized evaluation of 14 on-device models and 5 cloud APIs across three publicly available clinical conversation datasets totaling 400 conversations; (2) a clinical term recall analysis using the Unified Medical Language System (UMLS) to assess clinically relevant transcription errors; (3) an evaluation of ROVER hypothesis fusion across all pairwise and three-model combinations, quantifying the marginal benefit of multi-model inference for clinical on-device ASR; and (4) a cost analysis comparing cloud API expenditure against zero-marginal-cost on-device inference. All code, results, and evaluation scripts are publicly available.

# Results


## On-Device vs. Cloud ASR Performance

On-device models approached or matched cloud API performance across all three datasets (Table 1, Figure 1). On the Kazi et al. psychiatric dataset (71 conversations), the best on-device model (parakeet-tdt-0.6b-v2, 7.30%) was within 0.47 percentage points of the best cloud API (6.83%) and outperformed two of five cloud services. On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 11.23%) trailed the best cloud API (Azure, 7.70%) by 3.53 percentage points. On PriMock57 (57 mock primary care consultations), the gap was widest: the best on-device model (parakeet-tdt-0.6b-v2, 13.85%) trailed the best cloud API (9.88%) by 3.97 percentage points.

No single on-device model performed best across all datasets. Whisper-distil-v3.5 led on the OSCE dataset while parakeet-tdt-0.6b-v2 led on both PriMock57 and the psychiatric dataset. Among cloud APIs, one service (Google medical_conversation) exhibited WER 4 to 16 percentage points higher than other cloud APIs, attributable to faithful transcription of disfluencies (see Limitations).


## ROVER Fusion

ROVER (Recognizer Output Voting Error Reduction) was evaluated on a pool of 9 competitive on-device models; four models with solo WER exceeding 30% on two or more datasets (Nemotron, Zipformer, MedASR, MedASR int8) and one int8-quantized duplicate (NeMo FastConformer int8) were excluded because weak models consistently degraded fused output. We exhaustively evaluated all 36 pairwise combinations on PriMock57 and the Kazi et al. psychiatric dataset, and all 84 three-model combinations on PriMock57, using progressive pairwise fusion with equal model weighting. On the psychiatric dataset, 12 of the 84 three-model combinations completed before the exhaustive search was terminated due to runtime; all 12 included parakeet-tdt-0.6b-v2 as an artifact of parallel job ordering, and no deliberate selection was applied. On the OSCE respiratory interview dataset, the O(n²) alignment cost across 272 files precluded exhaustive search in our pure-Python implementation, so we report only the targeted fusion of the two best-performing solo on-device models.

Across all three datasets, ROVER fusion produced small improvements over the best single on-device model (Figure 2). On PriMock57, the best pair (parakeet-tdt-0.6b-v2 + qwen3-asr, 13.03%) improved WER by 0.82 percentage points over the best single model (parakeet-tdt-0.6b-v2, 13.85%). On the psychiatric dataset, the best pair (parakeet-tdt-0.6b-v2 + sensevoice, 7.15%) improved WER by 0.15 percentage points over the best single model (parakeet-tdt-0.6b-v2, 7.30%). On the OSCE dataset, the best pair (parakeet-tdt-0.6b-v2 + sensevoice, 11.01%) improved WER by 0.22 percentage points over the best single on-device model (whisper-distil-v3.5, 11.23%); the best fused result did not surpass the best cloud API (Azure, 7.70%).

Three-model combinations did not outperform the best pair on any dataset. On PriMock57, the best triple (parakeet-tdt-0.6b-v2 + whisper-turbo + qwen3-asr, 13.12%) was 0.09 percentage points worse than the best pair. On the psychiatric dataset, the best triple (parakeet-tdt-0.6b-v2 + whisper-turbo + sensevoice, 7.26%) was 0.11 percentage points worse. The third model introduced alignment noise that offset any additional voting benefit.


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

ROVER fusion produced only small WER improvements over the best single on-device model across all three datasets (−0.22 pp on OSCE, −0.82 pp on PriMock57, −0.15 pp on psychiatry). On OSCE, the best fused on-device pair did not surpass the best cloud API (Azure, 7.70%). An exhaustive search of all three-model combinations found that adding a third model consistently degraded performance relative to the best pair, with the third model introducing alignment noise that offset any additional voting benefit. A depth-study on the strongest pair tested six fusion algorithms beyond naive equal-weight voting — confusion network combination with epsilon tuning, three confidence-weighted variants using per-token log probabilities, margin-weighted voting, and a learned per-word keep classifier under 5-fold cross-validation — none exceeded naive ROVER by more than ~0.4 percentage points, and the best practical method (epsilon-tuned CNC at 13.05% WER) remains nearly 4 percentage points above the oracle upper bound of 9.07% (Supplementary Table S1). The residual gap reflects consensus errors across the available models rather than a limitation algorithmic fusion can recover. Given that fusion requires running two or more models per encounter, approximately doubling compute, memory footprint, and latency, the marginal WER improvements observed here may not justify the deployment cost for clinical on-device use. These results suggest that investment in improving a single model through domain adaptation or medical fine-tuning is likely to yield greater returns than multi-model fusion for clinical ASR.


## Privacy, Access, and Regulation

On-device ASR eliminates patient audio transmission, addressing one component of clinical AI privacy concerns, though broader considerations including consent, transparency, data governance, and accountability remain relevant for both deployment modes. For privacy-sensitive specialties such as psychiatry, substance abuse treatment, and reproductive health, eliminating audio transmission may be a prerequisite for patient acceptance and regulatory compliance regardless of BAA protections. Clinical deployment may fall under FDA oversight depending on intended use, requiring ongoing performance monitoring and bias surveillance. The zero-marginal-cost nature of on-device ASR also has equity implications. A 2026 national study found that ambient AI adoption in U.S. hospitals is concentrated among institutions with stronger operating margins, larger size, and metropolitan location, with lower adoption in under-resourced settings. Cloud costs of $2,000 to $20,000 annually may reinforce this disparity. On-device models run on devices clinicians already own, eliminating cost as a barrier to adoption and potentially serving as the only legally compliant option in regions with strict data sovereignty laws.


## Limitations

All three datasets consist of simulated or enacted encounters rather than real patient-clinician conversations, which present additional challenges including overlapping speech, background noise, and non-lexical conversational sounds. No open-license datasets of real clinical conversations currently exist. On-device models were evaluated using a single inference engine (sherpa-onnx); performance may vary under alternative runtimes. The UMLS-based clinical term identification uses a manually curated stopword list that may miss collisions or exclude valid terms, and the span-level scoring may overestimate error rates for longer phrases. One cloud API (Google medical_conversation) exhibited substantially higher WER due to faithful transcription of disfluencies, illustrating a limitation of WER as a standalone metric. Cloud results reflect early 2026 pricing.


## Future Directions

Future work should evaluate on real clinical recordings such as the Bridge2AI-Voice corpus. The data gap itself is addressable without patient data: OSCE examinations are conducted routinely at medical schools worldwide, are clinically realistic, involve students and evaluators with diverse accents and languages, and contain no protected health information. If institutions shared even a fraction of these recordings under open licenses, the research community would have access to a large, diverse, and ecologically valid clinical ASR benchmark. The barrier is not technical or ethical but institutional willingness. The modest gains from ROVER fusion suggest that medical fine-tuning of a single strong model may be more productive than multi-model combination. Specialty-specific benchmarks and evaluation of complete documentation pipelines would provide more clinically meaningful assessments.

# Conclusion

On-device ASR models approach the clinical term recall of commercial cloud APIs while eliminating patient audio transmission and recurring costs. In our evaluation, ROVER fusion produced only small WER improvements (≤ 0.82 percentage points) over the best single on-device model across all three datasets, and three-model combinations offered no additional benefit, suggesting that domain adaptation of a single model is a more promising path than multi-model fusion for clinical on-device ASR. Until medication transcription accuracy improves, clinical deployment of any ASR system requires robust verification workflows. By running on devices clinicians already own, on-device ASR may democratize access to documentation technology for settings where cloud services are unaffordable or legally prohibited, while addressing privacy concerns paramount in sensitive clinical specialties.

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

The OSCE dataset reference transcripts were distributed with apostrophes removed from contractions (e.g., "IM" for "I'm"), which interacted with the normalizer's contraction expansion to inflate WER for all systems evaluated on this dataset. We corrected this by injecting apostrophes into the reference using a 44-token dictionary of English contractions, of which 39 were unambiguous (e.g., "dont" → "don't") and 5 were validated against the hypothesis texts as predominantly contraction forms (its, ill, id, lets, wed). Two ambiguous tokens (well, were) were left uncorrected because the bare-word interpretation was dominant in the hypotheses. The OSCE WER values reported in Table 1 reflect this corrected reference. The PriMock57 and psychiatric dataset references were unaffected.


## ROVER Fusion

Recognizer Output Voting Error Reduction (ROVER) (Fiscus, 1997) combines transcription hypotheses by aligning them at the word level and selecting the most frequently occurring word at each position through majority voting. Models with solo WER exceeding 30% on two or more datasets and int8-quantized duplicates were excluded from the fusion pool because weak models consistently degraded fused output, yielding a pool of 9 competitive on-device models. We exhaustively evaluated all 36 pairwise combinations with equal weighting on the PriMock57 and Kazi et al. psychiatric datasets, and all 84 three-model combinations on PriMock57 using progressive pairwise fusion (models A and B fused first, intermediate result then fused with C, equal weighting at each step). The exhaustive three-model search on the psychiatric dataset was terminated due to runtime; 12 of the 84 combinations completed (all 12 included parakeet-tdt-0.6b-v2 as an artifact of parallel job ordering) and are reported as-is. On the OSCE respiratory interview dataset, the O(n²) per-file alignment cost precluded exhaustive search in our pure-Python implementation, so we report only the targeted fusion of the two best-performing solo on-device models (parakeet-tdt-0.6b-v2 + sensevoice). Complete pair and triple results for all datasets are provided in Supplementary Tables S2 and S3. To test whether sophisticated fusion algorithms would exceed naive equal-weight voting, we additionally evaluated six fusion variants on the strongest two-model pair (parakeet-tdt-0.6b-v2 + whisper-distil-v3.5) on PriMock57: confusion network combination with epsilon tuning (Mangu et al., 2000); three confidence-weighted ROVER variants using mean token log-probability, Shannon entropy, and Tsallis entropy; margin-weighted voting; and a learned per-word keep classifier trained via 5-fold group cross-validation. Results are reported in Supplementary Table S1. Fused WER was computed using the same normalization pipeline.


## Clinical Term Recall

To assess clinical relevance beyond aggregate WER, we measured the accuracy of medical terminology transcription using the Unified Medical Language System (UMLS). For each reference transcript, we identified all medical concept spans using QuickUMLS (medspacy fork, threshold 0.8) against the UMLS 2025AB Metathesaurus, restricted to English and to clinically relevant semantic types (drugs, conditions, symptoms, findings, procedures, anatomy, and lab results). QuickUMLS performs approximate string matching against the full UMLS vocabulary and returns both single-word terms (e.g., "hypertension") and multi-word phrases (e.g., "chest pain", "past medical history") with their character-level positions. Common English words that coincidentally match UMLS abbreviations or eponyms were excluded via a curated stopword list (e.g., "said" matches Simian AIDS, "still" matches Still's Disease). We then aligned each model's hypothesis to the reference using jiwer (which uses the rapidfuzz library for efficient Levenshtein alignment) and determined which reference word positions contained errors. A medical concept span was scored as an error if any word within the span was incorrectly transcribed. Clinical term recall was computed per model as the proportion of medical concept spans correctly transcribed.


## Cost Tracking

Cloud API costs were recorded from each provider's billing dashboard after all benchmark runs were complete. On-device inference costs were considered zero per encounter, as models run on local hardware with no API fees. Annual cost projections were estimated based on published per-minute or per-request pricing for a hypothetical practice performing 30 encounters per day.

# Data and Code Availability

All evaluation code, model inference scripts, ROVER fusion implementation, and per-file results are publicly available at github.com/graiai-inc/stapes. The three clinical conversation datasets are available under CC-BY licenses at their respective repositories. UMLS 2025AB requires a free license from the National Library of Medicine.
