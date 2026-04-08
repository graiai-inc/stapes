## Results

### On-Device vs. Cloud ASR Performance

We evaluated 14 on-device models and 5 cloud APIs across three clinical conversation datasets totaling 400 conversations and approximately 80 hours of audio. Word error rates (WER) were computed using the Whisper English text normalizer, adopted as the standard normalization pipeline by the HuggingFace Open ASR Leaderboard and MLPerf Inference benchmarks. All on-device models were evaluated using the sherpa-onnx inference engine (v1.12.34), an open-source, cross-platform runtime for ONNX-format speech models designed for edge deployment. Model performance may differ under alternative inference engines.

On-device models approached or matched cloud API performance across all three datasets (Table 1). On the Kazi et al. psychiatric dataset (71 conversations), the best on-device model (parakeet-tdt-0.6b-v2, 7.30%) was within 0.47 percentage points of the best cloud API (6.83%) and outperformed two of five cloud services. On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 17.59%) trailed the best cloud API (14.56%) by 3.03 percentage points. On PriMock57 (57 mock primary care consultations), the gap was widest: the best on-device model (parakeet-tdt-0.6b-v2, 13.85%) trailed the best cloud API (9.88%) by 3.97 percentage points.

No single on-device model performed best across all datasets. Whisper-distil-v3.5 led on the OSCE dataset while parakeet-tdt-0.6b-v2 led on both PriMock57 and the psychiatric dataset. Among cloud APIs, Google's medical_conversation model consistently exhibited WER 5 to 16 percentage points higher than other cloud APIs across all three datasets (23.08%, 26.00%, 10.52%). Examination of its transcripts revealed verbatim transcription of filled pauses, disfluencies, and hesitation repetitions that other services and the reference transcripts omit. This inflates WER despite potentially more faithful transcription, and illustrates a limitation of WER as a standalone metric.

### ROVER Fusion

ROVER (Recognizer Output Voting Error Reduction) was evaluated across all pairwise combinations of the 14 on-device models on each dataset (91 pairs per dataset).

On the OSCE dataset, ROVER substantially improved performance. The best pair (parakeet-tdt-0.6b-v2 + sensevoice, 10.71%) reduced WER by 6.88 percentage points relative to the best single on-device model and outperformed all five cloud APIs. Fusion improved WER on all 272 files for the top pairs.

On PriMock57 and the psychiatric dataset, ROVER degraded performance for every model pair tested. The best fused WER on PriMock57 (20.77%) was 6.92 percentage points worse than the best single model. On the psychiatric dataset, the best fused WER (17.04%) was 9.74 percentage points worse. No pair achieved a per-file improvement on either dataset (0 of 57 and 0 of 71 files, respectively).

This dataset-dependent behavior has not been previously reported. We examined the transcripts and found that on the OSCE dataset, models produced structurally similar outputs with independent per-word errors that voting could correct. On PriMock57 and the psychiatric dataset, models diverged structurally, omitting or rearranging different conversational segments. The word-level alignment then introduced errors rather than correcting them.

### Medical Terminology Accuracy

Medical term recall (MTR), the proportion of identified medical terms in the reference that were correctly transcribed, was computed for all models using pattern-based classification of error words into clinical categories (drugs, conditions, procedures, anatomy, abbreviations).

On-device models achieved MTR within 1 to 2 percentage points of cloud APIs across all datasets (Table 2). On the psychiatric dataset, the best on-device model (98.49%) exceeded three of five cloud APIs. On PriMock57, the best on-device model (98.90%) matched the fourth-ranked cloud API. On the OSCE dataset, the top on-device model (99.20%) trailed the best cloud API by 0.56 percentage points.

Drug names exhibited the highest per-word error rates across both on-device and cloud models: ramipril (67.5% error rate on the OSCE dataset), rosuvastatin (75.2%), amoxicillin (79.8% on PriMock57), and lisinopril (77.3% on PriMock57). These failures were consistent across model types, suggesting that medical terminology remains an intrinsic challenge for current ASR systems regardless of deployment mode.

### Cost Analysis

Total cloud API costs for evaluating approximately 80 hours of audio across all three datasets ranged from $19.33 to $382.64 (Table 3). The five services ranked by total cost were: Azure Speech ($19.33), Deepgram Nova-2 Medical ($25.78), AssemblyAI ($28.10), AWS Transcribe Medical ($99.63, representative subsets only due to cost), and Google Cloud Speech ($382.64, which may include preliminary test runs). On-device inference incurred no per-encounter cost after initial model download (model sizes range from 37 MB to 1.5 GB). For a practice transcribing 30 encounters per day, estimated annual cloud costs at published rates would range from approximately $2,000 to over $20,000, depending on the service, whereas on-device processing requires only the computational cost of a consumer mobile device.
