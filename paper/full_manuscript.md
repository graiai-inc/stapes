
# Abstract

**Objective:** To determine whether privacy-preserving on-device speech recognition achieves clinical term accuracy comparable to commercial cloud services for ambient clinical documentation, to characterize medication-name recognition failures, and to test whether multi-model fusion narrows the residual accuracy gap.

**Materials and Methods:** We evaluated 12 on-device ASR models and 5 cloud APIs on 400 simulated conversations from three public datasets (OSCE respiratory, n=272; PriMock57 primary care, n=57; Kazi et al. psychiatric, n=71). WER was reported in standard (Whisper normalizer) and meaning-preserving (collapsing orthographic variants and backchannels) regimes. Clinical term recall (CTR) used UMLS 2025AB spans via QuickUMLS with BCa bootstrap 95% CIs. ROVER and six advanced fusion algorithms were evaluated.

**Results:** On-device CTR was within 1–3 pp of the best cloud service per dataset. The standard-WER gap was 0.5–5.2 pp; the meaning-preserving-WER gap narrowed to 0.7–3.3 pp, consistent with differential backchannel handling. Medication names were the highest-error category for most models; the highest-error drugs were misrecognized in 85% to 93% of on-device occurrences versus 3% to 30% in the cloud. ROVER gained ≤ 0.82 pp; no advanced algorithm exceeded naive voting by more than 0.4 pp.

**Discussion:** The dominance of medication errors across models reflects look-alike/sound-alike confusability; near-parity clinical term recall alongside these persistent errors shows that aggregate accuracy can mask clinically critical failure modes.

**Conclusion:** Privacy-preserving on-device ASR is a viable, lower-cost alternative to cloud services for clinical transcription, and a single model suffices; fusion did not close the small residual gap. Until medication accuracy improves, deployment in either mode requires explicit medication verification before note finalization.

# Background and Significance

Physicians spend more than half their working hours on documentation, and administrative burden is a leading contributor to burnout.[1, 2] Ambient AI scribes, which passively transcribe clinical encounters for automated note generation, have emerged as a promising intervention. Evaluations report reductions in after-hours documentation time and improved clinician satisfaction,[3, 4] though results are mixed: one large cohort study of a commercial ambient scribe found no significant time savings and worsened after-hours EHR use.[5] Regardless of downstream effectiveness, all such systems depend on the accuracy of automatic speech recognition (ASR), where reported word error rates range from under 1% for controlled dictation to over 50% for naturalistic clinical speech.[6]

Commercial cloud ASR services now offer medical-specific models, but these require transmitting patient audio to remote servers, and some ambient scribe services retain full audio recordings for weeks after the encounter.[7] Business Associate Agreements (BAAs) address HIPAA liability but do not eliminate the privacy risks of transmitting and storing protected health information with third parties, and recent analyses raise broader concerns about consent, transparency, and data governance beyond transmission security alone.[7, 8] On-device ASR, which processes audio locally without network transmission or third-party storage, addresses these concerns. Open-source ASR models have improved rapidly, with several now suitable for mobile and edge deployment, yet no standardized benchmark compares them against commercial cloud APIs on clinical conversation data.

Existing evaluations of medical ASR have significant limitations. Industry benchmarks report WER on proprietary datasets that cannot be independently verified. Afonja et al. introduced a medical WER for accented clinical speech but relied on a private dataset and a commercial NER service, limiting reproducibility.[9] Ng et al.'s systematic review noted the absence of a standardized evaluation framework for clinical ASR,[6] and the socio-technical risks of clinical speech-to-text systems remain largely unaddressed.[7]

We present the first open, reproducible benchmark comparing on-device and cloud ASR for medical conversations. Our contributions are: (1) a standardized evaluation of 12 on-device models and 5 cloud APIs across three public clinical conversation datasets totaling 400 conversations; (2) a clinical term recall analysis using the Unified Medical Language System (UMLS) to assess clinically relevant errors; (3) a per-term error analysis identifying medication names as the highest-error category for most models in both deployment modes, with patient-safety implications for ambient documentation; (4) an evaluation of ROVER hypothesis fusion across pairwise and three-model combinations, quantifying the marginal benefit of multi-model inference; and (5) a cost analysis of cloud API expenditure versus zero-marginal-cost on-device inference.

# Materials and Methods


## Datasets

We evaluated three public English-language clinical conversation datasets spanning different specialties and recording conditions.

The OSCE respiratory interview dataset[10] contains 272 simulated patient-physician interviews focused on respiratory cases, recorded in a controlled OSCE examination setting; audio and manually corrected transcripts are available under a CC-BY license on figshare.

PriMock57[11] contains 57 mock primary care consultations recorded between medical professionals acting as doctor and patient. The dataset includes audio recordings, utterance-level transcripts, and consultation notes, released under a CC-BY license.

The Kazi et al. psychiatric dataset[12] contains 71 enacted psychiatric consultations generated by pairs of students reading from clinical transcripts, released under a CC-BY 4.0 license. All three datasets consist of simulated or enacted encounters; no open-license dataset of real patient-clinician conversations currently exists (see Limitations).


## On-Device Models

We evaluated 12 publicly available ASR models spanning four architectural families: encoder-decoder (Whisper distil-v3.5, Whisper turbo, Whisper base.en, Qwen3-ASR), transducer (parakeet-tdt-0.6b-v2, NeMo FastConformer, Nemotron, Zipformer), CTC (SenseVoice, SenseVoice without ITN, MedASR), and non-autoregressive (Paraformer-en). Model parameter counts ranged from 67M (Zipformer) to 809M (Whisper turbo). SenseVoice was evaluated with and without inverse text normalization (ITN), the only model in the pool to expose ITN as a runtime option; both settings are used in deployment. All on-device models were evaluated using their published int8-quantized builds, the form distributed for edge deployment; in two of these builds the upstream distribution leaves one component unquantized (the Zipformer decoder and the Qwen3-ASR convolutional frontend).

Eleven of the 12 on-device models are general-purpose; the twelfth, MedASR, is a medical-dictation model. All were run through the sherpa-onnx inference engine (v1.12.34), an open-source runtime that executes ONNX models on CPU without GPU acceleration for mobile and edge deployment. Benchmarks ran on a Lenovo ThinkPad P14s Gen 4 (AMD Ryzen 7 PRO 7840U, 64 GB RAM, Ubuntu 24.04 LTS); inference was faster than real-time for all models. The same 12 models were evaluated on all three datasets for consistent comparison.


## Cloud APIs

Five commercial cloud ASR services were evaluated: Azure Speech (Batch Transcription v3.2), Google Cloud Speech (v1, `medical_conversation` model), Deepgram Nova-2 Medical, AssemblyAI (Universal-3 Pro, invoked with the system prompt "This is a doctor-patient clinical conversation."), and AWS Transcribe Medical (Specialty=PRIMARYCARE, Type=CONVERSATION). Three of the five services (Google, Deepgram, AWS) used medical-specific models, Azure and AssemblyAI general-purpose models. Azure, Deepgram, AssemblyAI, and AWS Transcribe Medical were run on all 400 files; Google Cloud Speech on 396: four of the six longest psychiatric recordings had not completed within our 10-minute client-side wait and were not retried. Because each service exposes different clinical-configuration controls (Google and Deepgram offer medical model selection, AWS specialty and type parameters, AssemblyAI a free-text system prompt, and Azure none), these settings are not strictly equivalent across providers; the AssemblyAI clinical system prompt in particular has no exact analog in the other services and may modestly advantage or disadvantage it.


## Text Normalization and WER Computation

Word error rate was computed under two normalization regimes. **Standard WER** applied the Whisper English text normalizer (whisper-normalizer v1.0), which performs case folding, punctuation removal, number-to-word conversion, and contraction expansion; this is the de facto pipeline used by the HuggingFace Open ASR Leaderboard[13] and MLPerf Inference ASR benchmarks, reported here for direct comparability. **Meaning-preserving WER** additionally applies a normalization layer (described below) that collapses orthographic variants and conversational backchannels; both regimes are reported for every model and dataset. Because we evaluated an exhaustive set of models and files rather than sampling from a population, we report descriptive performance metrics without hypothesis tests. To quantify file-level variability in clinical term recall we computed 95% bias-corrected and accelerated (BCa) bootstrap confidence intervals over files (10,000 resamples). Resampling used common random numbers within each dataset (a fixed per-dataset seed), so the intervals are reproducible and any future model can be added without recomputing the others, while model-versus-model comparisons draw matched resamples. These intervals are reported alongside each recall estimate.

The OSCE dataset reference transcripts were distributed with apostrophes removed from contractions (e.g., "IM" for "I'm"), which interacted with the normalizer's contraction expansion to inflate WER for all systems on this dataset. We corrected this by injecting apostrophes into the reference using a 44-token contraction dictionary, of which 39 were unambiguous (e.g., "dont" → "don't") and 5 were validated against the hypothesis texts as predominantly contraction forms (its, ill, id, lets, wed). Two ambiguous tokens (well, were) were left uncorrected because the bare-word interpretation dominated in the hypotheses. All reported OSCE WER values reflect this corrected reference; the PriMock57 and psychiatric references were unaffected.


## Clinical Text Normalization

The Whisper normalizer is domain-agnostic: it does not collapse orthographic variants common in clinical text that do not change meaning, nor remove conversational backchannels that some services transcribe verbatim and others suppress. We added a meaning-preserving layer on the Whisper baseline, with rules in eight categories. (1) **British/American spelling**: a 100-entry dictionary (e.g., diarrhoea/diarrhea, oedema/edema, colour/color) plus suffix rules (-our/-or, -ise/-ize, -re/-er, -ae-/-e-, -oe-/-e-). (2) **Hyphenation**: hyphens replaced with spaces so that "x-ray", "xray", and "x ray" map identically. (3) **Open versus closed compounds**: a 35-entry dictionary (e.g., health care/healthcare, post operative/postoperative). (4) **Possessive eponyms**: common medical eponyms (Crohn, Alzheimer, Parkinson, and others) canonicalized so that "Crohn's", "Crohns", and "Crohn disease" map identically; the possessive 's is stripped before the Whisper normalizer expands it into "is". (5) **Spaced acronyms**: about 60 medical acronyms (CT, MRI, ECG, COPD, BID) joined when transcribed as spaced single letters ("c t scan" → "ct scan"). (6) **Honorifics**: abbreviated titles (Dr., Mr., Mrs., Ms.) expanded to spelled-out forms. (7) **Dosing units**: spelled-out units (milligrams, micrograms, milliliters) collapsed to abbreviations (mg, mcg, ml). (8) **Conversational backchannels**: a closed list of non-content tokens (um, uh, er, ah, hmm, mm, mhm, uhhuh, huh, hum, yeah, yep, yup, okay, ok, oh) stripped symmetrically from reference and hypothesis so that systems transcribing them verbatim are not penalized against systems that suppress them. We deliberately retained ambiguous tokens that can carry clinical meaning (e.g., "right" can refer to a side, "well" to health status, "no" and "yes" to clinical decisions). The full rule set (~300 lines, with self-tests) is in `scripts/extended_normalizer.py`; it is applied symmetrically, is idempotent, and runs over stored hypotheses without re-running ASR inference.


## ROVER Fusion

Recognizer Output Voting Error Reduction (ROVER)[14] combines hypotheses by aligning them at the word level and selecting the most frequent word at each position. Models with solo WER exceeding 30% on two or more datasets were excluded because weak models degraded fused output, yielding a pool of 9 competitive on-device models (the fusion runs predate the int8 consolidation and used the full-precision NeMo FastConformer build, which appears in no best-performing combination). We evaluated all 36 pairwise combinations with equal weighting on PriMock57 and the psychiatric dataset, and all 84 three-model combinations on PriMock57 using progressive pairwise fusion (equal weighting at each step). Exhaustive triple search on the psychiatric and OSCE datasets was infeasible in our pure-Python implementation (O(n²) per-file alignment); for OSCE we report only the targeted fusion of the two best solo on-device models (parakeet-tdt-0.6b-v2 + sensevoice). Complete pair and triple results are in Supplementary Tables S2 and S3. To test whether sophisticated algorithms exceed naive equal-weight voting, we evaluated six fusion variants on the pairing of the two best solo models (parakeet-tdt-0.6b-v2 + whisper-distil-v3.5) on PriMock57, the dataset with the largest pairwise fusion gain: confusion network combination with epsilon tuning;[15] three confidence-weighted ROVER variants using mean token log-probability, Shannon entropy, and Tsallis entropy; margin-weighted voting; and a learned per-word keep classifier trained via 5-fold group cross-validation (Supplementary Table S1). Fused WER used the same normalization pipeline.


## Clinical Term Recall

To assess clinical relevance beyond aggregate WER, we measured the accuracy of medical terminology transcription using the Unified Medical Language System (UMLS). For each reference transcript, we identified all medical concept spans using QuickUMLS[16] (medspacy fork, threshold 0.8) against the UMLS 2025AB Metathesaurus, restricted to English and clinically relevant semantic types (drugs, conditions, symptoms, findings, procedures, anatomy, and lab results). QuickUMLS performs approximate string matching, returning single-word terms (e.g., "hypertension") and multi-word phrases (e.g., "chest pain") with their character positions. Common English words coincidentally matching UMLS abbreviations or eponyms were excluded via a curated stopword list (e.g., "said" matches Simian AIDS, "still" matches Still's Disease). We aligned each model's hypothesis to the reference using jiwer (rapidfuzz Levenshtein alignment) and identified which reference positions contained errors. A span was scored as an error if any of its words was incorrectly transcribed; clinical term recall is the proportion of spans correctly transcribed.


## Cost Tracking

Cloud API costs were recorded from each provider's billing dashboard at published rates; introductory credits defrayed out-of-pocket cost on all five services, and only Google Cloud Speech exceeded its $300 credit during the benchmark. On-device inference cost was considered zero per encounter. Per-encounter cost was total cost divided by encounter count; the annual illustration assumes one full-time clinician (~5,000 encounters per year).


## AI Tool Use

This work used AI coding assistants (Anthropic Claude Code and Google Gemini Code Assist) for programming support in developing evaluation scripts, data-verification tools, and fusion code. All scientific design, dataset selection, analysis decisions, and manuscript text were authored and verified by the author; AI-generated code was reviewed and tested before use.

# Results


## On-Device vs. Cloud ASR Performance

On-device models approached or matched cloud API performance across all three datasets (Table 1, Figure 1). We report standard and meaning-preserving WER side by side (see Methods). On the OSCE respiratory interview dataset, the best on-device model (whisper-distil-v3.5) trailed the best cloud API by 3.53 percentage points in standard WER (11.23% vs. 7.70%, Azure) but only 1.25 percentage points in meaning-preserving WER (8.51% vs. 7.26%, Deepgram). On PriMock57, the best on-device model (parakeet-tdt-0.6b-v2) trailed the best cloud API (AWS Transcribe Medical) by 5.17 percentage points in standard WER (13.85% vs. 8.68%) and 3.30 percentage points in meaning-preserving WER (11.39% vs. 8.09%). On the Kazi et al. psychiatric dataset, the gap was small in both regimes: parakeet-tdt-0.6b-v2 (7.30% Std / 7.28% MP) trailed Azure (6.83% Std / 6.58% MP) by under one percentage point. Across the three datasets, the meaning-preserving-WER gap between the best on-device model and the best cloud API was 0.7 to 3.3 percentage points (vs. 0.5 to 5.2 in standard WER); the larger gap on conversational datasets reflects backchannel suppression, common in OSCE and primary-care recordings but rare in psychiatric interviews.

[[TABLE1]]

No single on-device model performed best across all datasets: whisper-distil-v3.5 led on OSCE, parakeet-tdt-0.6b-v2 on both PriMock57 and the psychiatric dataset. Among cloud APIs, Google medical_conversation showed substantially higher WER than the other four across all three datasets in both regimes (16.53% / 13.38% on OSCE, 26.00% / 23.49% on PriMock57, 10.52% / 10.30% on psychiatric; standard / meaning-preserving). Verbatim transcription of backchannels explains part of this gap: on OSCE, Google's output averaged 2,007 words per file against 1,314 in the reference (53% more tokens), with roughly 162 disfluency tokens per file versus 17 for Azure and 30 for AssemblyAI, consistent with the `medical_conversation` model being tuned to transcribe fillers and false starts verbatim rather than misconfigured. Google nonetheless remained the highest-WER cloud service after symmetric backchannel removal, indicating a residual content-word gap, not solely a disfluency artifact (see Limitations).

OSCE WER values use an apostrophe-injected reference (see Methods); this correction reduced every system's OSCE WER, by approximately 6 percentage points for most systems (1.6 to 6.9 overall), relative to the distributed reference.


## ROVER Fusion

ROVER was evaluated on a pool of 9 competitive on-device models: all 36 pairs on PriMock57 and the psychiatric dataset, all 84 triples on PriMock57, and (see Methods) the targeted fusion of the two best solo models on OSCE. Fusion produced only small improvements over the best single on-device model (Figure 2): the best pair gained 0.82 pp on PriMock57 (parakeet-tdt-0.6b-v2 + qwen3-asr, 13.03% vs. 13.85%), 0.22 pp on OSCE (parakeet-tdt-0.6b-v2 + sensevoice, 11.01% vs. 11.23% solo), and 0.15 pp on the psychiatric dataset. No fused result surpassed the best cloud API on any dataset (e.g., Azure 7.70% on OSCE).

On PriMock57, the only dataset with an exhaustive triple search, no triple beat the best pair: the best (13.12%) was 0.09 pp worse, with the third model introducing alignment noise that offset any voting benefit.


## Clinical Term Recall

On-device models achieved clinical term recall (CTR) within 1 to 3 percentage points of cloud APIs across all datasets, with overlapping or near-adjacent 95% bootstrap CIs for the strongest comparisons (Table 2). On the psychiatric dataset, the best on-device models (whisper-turbo 92.0% [91.3, 92.6], whisper-distil 91.9% [91.2, 92.5], parakeet-tdt 91.8% [91.1, 92.4]) were within 1 percentage point of the best cloud API (Azure 93.0% [92.3, 93.5]). On the OSCE dataset, the best on-device model (whisper-turbo 93.6% [93.2, 93.9]) trailed the best cloud API (AWS Transcribe Medical 95.6% [95.4, 95.9]) by 2.0 percentage points, with non-overlapping intervals. On PriMock57, the gap was widest: parakeet-tdt (89.4% [88.3, 90.4]) trailed AWS Transcribe Medical (92.3% [91.5, 93.1]) by 2.9 percentage points. Across all cells, 95% CIs were 1 to 3 percentage points wide, meaning the best on-device-vs-cloud CTR differences are of similar magnitude to the measurement uncertainty.

[[TABLE2]]

Model rankings differed between WER and CTR: whisper-turbo, second on overall WER, achieved the highest on-device CTR on two of three datasets. Medication names were the highest-error category for most models on every dataset, and the per-drug failures split sharply by deployment mode: pooled across the 12 on-device configurations, wellbutrin was misrecognized in 86% of occurrences (psychiatric), lisinopril in 92% and amoxicillin in 93% (PriMock57), and ramipril in 86%, rosuvastatin in 92%, and lisinopril in 85% (OSCE), versus 3% to 30% for the same drugs pooled across the five cloud services (per-model rates in Supplementary Table S4). The best overall on-device models were no exception: whisper-turbo erred on 50% to 78% of these occurrences, parakeet-tdt-0.6b-v2 on 83% to 100%. Drug names nonetheless remained the weakest cloud category on two of three datasets.


## Cost Analysis

Total cloud API costs for the benchmark's approximately 90 hours of audio (400 encounters) ranged from $19.33 (Azure Speech) to $397.96 (AWS Transcribe Medical), with Deepgram Nova-2 Medical at $25.78, AssemblyAI at $28.10, and Google Cloud Speech at $382.64 (Table 3). This corresponds to roughly $0.05 to $0.07 per encounter for Azure, Deepgram, and AssemblyAI versus about $1.00 per encounter for Google and AWS, which are priced 14 to 20 times higher. For a single full-time clinician (about 5,000 encounters per year), annual costs at published rates would range from roughly $240 to $5,000 depending on the service. On-device inference incurred no per-encounter cost after initial model download (37 MB to 1.5 GB) and runs on consumer mobile devices clinicians already own.

[[TABLE3]]

# Discussion

On-device ASR models reached clinical term recall within 1 to 3 percentage points of commercial cloud APIs across three datasets, making privacy-preserving on-device ASR a viable option for clinical documentation. The best-performing on-device models were general-purpose, with no medical fine-tuning, while three of the five cloud services deployed medical-specific models; a medical model adapted to conversational rather than dictation speech would likely narrow the gap.

No single on-device model dominated across all datasets and metrics: whisper-turbo achieved the highest clinical term recall on two of three datasets, and parakeet-tdt-0.6b-v2 the lowest WER on two of three. Model choice should depend on which metric matters and on deployment constraints, not WER alone.

Medication names were the highest-error category for most models, with the most severe per-drug failures concentrated on-device. MedASR suggests this on-device deficit is targetable: despite uncompetitive overall WER, this medical-dictation model erred on wellbutrin in only 5% of occurrences, versus 86% pooled on-device and near the 3% cloud rate (Supplementary Table S4). Prior work found that 5.7 to 8.9% of ASR errors in clinical documents are clinically significant, with medication errors carrying the highest harm risk.[17] This pattern mirrors the look-alike/sound-alike (LASA) medication-error phenomenon, which accounts for 6.2 to 14.7% of medication errors and is driven by measurable phonetic confusability.[18, 19] Mitigations include cross-referencing ASR output against patient medication lists and institutional formularies,[20] flagging phonetically similar drug names for mandatory clinician review,[18, 19] and requiring explicit confirmation before finalizing notes with high-risk medications.[17] Because drug names remained the weakest category for most cloud services as well, these safeguards apply to both on-device and cloud ASR.

ROVER fusion produced only small WER improvements over the best single on-device model (≤ 0.82 pp across datasets), and the best fused pair did not surpass the best cloud API on any dataset. None of six advanced fusion algorithms in the depth-study (Methods) exceeded naive equal-weight voting by more than ~0.4 percentage points, and the best practical method (epsilon-tuned CNC at 13.05% WER) remained nearly 4 percentage points above the oracle upper bound of 9.07%, the WER of an omniscient per-word selector (Supplementary Table S1). The residual gap reflects consensus errors across the available models rather than a limitation algorithmic fusion can recover.

For deployment, multi-model fusion roughly doubles compute and memory and adds latency without justifying the cost on-device; single-model improvement through medical fine-tuning, hotword biasing, or domain-specific pretraining is the more promising path.

On-device ASR eliminates patient audio transmission but not other privacy concerns: consent, transparency, data governance, and accountability still apply to both modes.[7, 8] In a 2025 study, patient willingness to consent to ambient documentation fell from 82% to 55% once AI use, data storage, and corporate involvement were disclosed, underscoring the need for transparent consent regardless of where audio is processed.[21] For privacy-sensitive specialties such as psychiatry, substance-abuse treatment, and reproductive health, eliminating audio transmission may be a prerequisite for patient acceptance and regulatory compliance regardless of BAA protections;[7] clinical deployment may also fall under FDA oversight depending on intended use.[22] The zero-marginal-cost nature of on-device ASR has equity implications: a 2026 national study found ambient AI adoption in U.S. hospitals concentrated among institutions with stronger operating margins, larger size, and metropolitan location,[23] and annual cloud costs ranging from hundreds to several thousand dollars may reinforce this disparity. On-device models run on consumer hardware clinicians already own and may be the only legally compliant option where data-sovereignty laws are strict.

**Implications for practice.** On-device ASR is a viable option for ambient documentation, particularly in cost-, connectivity-, or sovereignty-constrained settings (community health centers, solo practices, jurisdictions with data localization requirements). Regardless of mode, off-the-shelf ASR is not yet trustworthy enough to elide a structured medication-reconciliation step before note finalization.

Future work should evaluate real clinical recordings such as the Bridge2AI-Voice corpus. OSCE examinations, held routinely worldwide and free of protected health information, are an under-used resource: institutional sharing under open licenses would substantially expand the available benchmark. ASR accuracy is only the first stage of the ambient-documentation pipeline: transcription errors propagate into the downstream LLM-generated note, where they may be amplified or masked. A recent Veterans Health Administration evaluation found AI-generated notes scored lower than human-produced notes across all quality domains,[24] and an interview study found clinicians edit AI drafts primarily for clinical accuracy, transcription errors, and missing context;[25] the clinical term recall gaps reported here therefore likely understate the downstream review burden. Specialty-specific benchmarks and pipeline-level evaluations of how transcription errors propagate through note generation would add more clinical signal than WER alone.

Several aspects merit caution. First and most important, all three datasets consist of simulated or enacted encounters rather than real patient-clinician conversations. Real encounters add overlapping speech, variable microphone placement, background noise, and non-lexical conversational sounds that degrade ASR accuracy, with reported word error rates reaching 50% in naturalistic multi-speaker settings;[6, 26, 27] the values reported here should therefore be read as a performance ceiling that would likely degrade in real-world deployment. No open-license datasets of real clinical conversations currently exist. The datasets also have narrow speaker diversity: the OSCE and PriMock57 recordings are UK-based and the psychiatric dataset uses student actors, so accents and dialects are limited, and on-device models may perform differently with more diverse or accented speakers.[9]

Other limitations are methodological. On-device models were evaluated using a single inference engine (sherpa-onnx); performance may vary under alternative runtimes. The UMLS-based term identification relies on a 36-token stopword list curated by the author from high-frequency QuickUMLS collisions with common English words (e.g., "said"/Simian AIDS, "still"/Still's Disease); as a single-reviewer artifact it may still miss collisions or exclude valid terms. Clinical term recall used span-level scoring, counting a span as an error if any of its words was misrecognized; because 78% of the 52,693 reference spans are single-word and the mean span length is 1.27 words (maximum five), this affects only a minority of multi-word phrases. WER is sensitive to backchannel handling and orthographic variants, which the meaning-preserving regime addresses (see Results). Finally, cloud results reflect early 2026 pricing and may change as providers update their rate schedules.

# Conclusion

Privacy-preserving on-device ASR is a viable alternative to commercial cloud services for clinical conversation transcription, with clinical term recall within 1 to 3 percentage points of the best cloud service per dataset, standard WER within 0.5 to 5.2 percentage points, and meaning-preserving WER within 0.7 to 3.3 percentage points. The widest gaps occur on a single dataset (PriMock57); on the other two, meaning-preserving WER is within 1.3 percentage points of the best cloud service. Neither deployment mode reliably transcribes medication names, and on-device failures are far more frequent; until that improves, any clinical ASR deployment requires explicit medication verification before notes are finalized. On-device deployment lowers the cost barrier in under-resourced settings and keeps audio off the network in sensitive specialties; single-model medical fine-tuning is more promising than multi-model fusion as a path to closing the remaining gap with cloud services.

# Data Availability

The three clinical conversation datasets evaluated in this work are publicly available under CC-BY licenses at their respective repositories: the OSCE respiratory interview dataset on figshare (Fareez et al., Scientific Data 2022), PriMock57 on the Babylon Health GitHub (Korfiatis et al., ACL 2022), and the Kazi et al. psychiatric dataset on Zenodo/GitHub. The Unified Medical Language System (UMLS) 2025AB Metathesaurus requires a free license from the U.S. National Library of Medicine. Per-model inference outputs, aggregated WER and clinical term recall tables, and per-file fusion results for the exhaustive round-robin search and the fusion depth-study are deposited with the code repository (see Code Availability).

# Code Availability

All evaluation code, model inference scripts, ROVER fusion implementation, the exhaustive pair and triple round-robin search, and the fusion depth-study (naive ROVER and six advanced fusion algorithms on PriMock57) are publicly available under the MIT license at https://github.com/graiai-inc/stapes. The depth-study scripts and per-file results are in the `fusion_depth/` subdirectory. Cloud API evaluation scripts for each of the five services are included, along with the apostrophe-injection correction applied to the OSCE reference transcripts.

# Ethics Approval

This study did not constitute human subjects research and did not require institutional review board approval: it used only publicly available datasets of simulated or enacted clinical encounters, released by their original creators under open licenses, and did not involve human participants, identifiable private information, or protected health information.

# Funding

This study received no funding.

# Acknowledgments

The author thanks the creators of the OSCE respiratory interview corpus, PriMock57, and the Kazi et al. psychiatric dataset for releasing these recordings under open licenses, which made this benchmark possible.

# Competing Interests

The author declares no competing interests.

# Author Contributions

Following the CRediT (Contributor Roles Taxonomy): J.G.F.: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing (original draft), Writing (review and editing), Visualization, Project administration.
