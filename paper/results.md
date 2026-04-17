## Results

### On-Device vs. Cloud ASR Performance

On-device models approached or matched cloud API performance across all three datasets (Table 1, Figure 1). On the Kazi et al. psychiatric dataset (71 conversations), the best on-device model (parakeet-tdt-0.6b-v2, 7.30%) was within 0.47 percentage points of the best cloud API (6.83%) and outperformed two of five cloud services. On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 11.23%) trailed the best cloud API (Azure, 7.70%) by 3.53 percentage points. On PriMock57 (57 mock primary care consultations), the gap was widest: the best on-device model (parakeet-tdt-0.6b-v2, 13.85%) trailed the best cloud API (9.88%) by 3.97 percentage points.

No single on-device model performed best across all datasets. Whisper-distil-v3.5 led on the OSCE dataset while parakeet-tdt-0.6b-v2 led on both PriMock57 and the psychiatric dataset. Among cloud APIs, one service (Google medical_conversation) exhibited WER 4 to 16 percentage points higher than other cloud APIs, attributable to faithful transcription of disfluencies (see Limitations).

### ROVER Fusion

ROVER (Recognizer Output Voting Error Reduction) was evaluated across all pairwise combinations of 9 competitive on-device models on each dataset (36 pairs per dataset). Three-model combinations (84 triples per dataset) were also evaluated using progressive pairwise fusion with equal model weighting. Models with solo WER exceeding 30% on two or more datasets (Nemotron, Zipformer, MedASR, MedASR int8) and int8-quantized duplicates (NeMo FastConformer int8) were excluded from the fusion pool, as weak models consistently degraded fused output.

Across all three datasets, ROVER fusion produced small improvements over the best single on-device model (Figure 2). On PriMock57, the best pair (parakeet-tdt-0.6b-v2 + qwen3-asr, 13.03%) improved WER by 0.82 percentage points over the best single model (parakeet-tdt-0.6b-v2, 13.85%). On the psychiatric dataset, the best pair (parakeet-tdt-0.6b-v2 + sensevoice, 7.15%) improved WER by 0.15 percentage points over the best single model (parakeet-tdt-0.6b-v2, 7.30%). On the OSCE dataset, the best pair (parakeet-tdt-0.6b-v2 + sensevoice, 11.01%) improved WER by 0.22 percentage points over the best single on-device model (whisper-distil-v3.5, 11.23%); the best fused result did not surpass the best cloud API (Azure, 7.70%).

Three-model combinations did not outperform the best pair on any dataset. On PriMock57, the best triple (parakeet-tdt-0.6b-v2 + whisper-turbo + qwen3-asr, 13.12%) was 0.09 percentage points worse than the best pair. On the psychiatric dataset, the best triple (parakeet-tdt-0.6b-v2 + whisper-turbo + sensevoice, 7.26%) was 0.11 percentage points worse. The third model introduced alignment noise that offset any additional voting benefit.

### Clinical Term Recall

On-device models achieved clinical term recall (CTR) within 1 to 3 percentage points of cloud APIs across all datasets (Table 2). On the psychiatric dataset, the best on-device models (whisper-turbo 92.0%, whisper-distil 91.9%, parakeet-tdt 91.8%) were within 1 percentage point of the best cloud API (93.0%). On the OSCE dataset, the best on-device model (whisper-turbo, 93.6%) trailed the best cloud APIs (95.5%) by 1.9 percentage points. On PriMock57, the gap was widest: parakeet-tdt (89.4%) trailed the best cloud API (92.4%) by 3.0 percentage points.

Model rankings differed between WER and CTR. Whisper-turbo, which ranked second on overall WER, achieved the highest on-device CTR on two of three datasets. Drug names exhibited the highest per-term error rates across both on-device and cloud models, including wellbutrin (60.4% error rate on the psychiatric dataset), and common medications and symptoms on the primary care and OSCE datasets. These failures were consistent across model types, suggesting that medical terminology remains an intrinsic challenge for current ASR systems regardless of deployment mode.

### Cost Analysis

Total cloud API costs for evaluating approximately 80 hours of audio across all three datasets ranged from $19.33 to $382.64 (Table 3). The five services ranked by total cost were: Azure Speech ($19.33), Deepgram Nova-2 Medical ($25.78), AssemblyAI ($28.10), AWS Transcribe Medical ($99.63, representative subsets only due to cost), and Google Cloud Speech ($382.64). On-device inference incurred no per-encounter cost after initial model download (model sizes range from 37 MB to 1.5 GB). On-device processing runs on consumer mobile devices that clinicians already own and regularly update, requiring no additional hardware investment. For a practice transcribing 30 encounters per day, estimated annual cloud costs at published rates would range from approximately $2,000 to over $20,000 depending on the service, whereas on-device processing has zero marginal cost per encounter.
