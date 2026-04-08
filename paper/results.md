## Results

### On-Device vs. Cloud ASR Performance

On-device models approached or matched cloud API performance across all three datasets (Table 1). On the Kazi et al. psychiatric dataset (71 conversations), the best on-device model (parakeet-tdt-0.6b-v2, 7.30%) was within 0.47 percentage points of the best cloud API (6.83%) and outperformed two of five cloud services. On the OSCE respiratory interview dataset (272 conversations), the best on-device model (whisper-distil-v3.5, 17.59%) trailed the best cloud API (14.56%) by 3.03 percentage points. On PriMock57 (57 mock primary care consultations), the gap was widest: the best on-device model (parakeet-tdt-0.6b-v2, 13.85%) trailed the best cloud API (9.88%) by 3.97 percentage points.

No single on-device model performed best across all datasets. Whisper-distil-v3.5 led on the OSCE dataset while parakeet-tdt-0.6b-v2 led on both PriMock57 and the psychiatric dataset.

### ROVER Fusion

ROVER (Recognizer Output Voting Error Reduction) was evaluated across all pairwise combinations of the 14 on-device models on each dataset (91 pairs per dataset).

On the OSCE dataset, ROVER substantially improved performance. The best pair (parakeet-tdt-0.6b-v2 + sensevoice, 10.7%) reduced WER by 6.9 percentage points from the best single on-device model (17.6%) and outperformed all five cloud APIs. Fusion improved WER on all 272 files for the top pairs.

On PriMock57 and the psychiatric dataset, ROVER degraded performance for every model pair tested. The best fused WER on PriMock57 (20.77%) was 6.92 percentage points worse than the best single model. On the psychiatric dataset, the best fused WER (17.04%) was 9.74 percentage points worse. No pair achieved a per-file improvement on either dataset (0 of 57 and 0 of 71 files, respectively).

This dataset-dependent behavior has not been previously reported. The top five on-device models had mean WER of 19.2% on the OSCE dataset, 15.7% on PriMock57, and 7.6% on the psychiatric dataset. ROVER improved results only on the dataset where individual models had the highest error rates. When models are already accurate, the alignment overhead of ROVER exceeds its correction benefit: there are fewer errors to fix by voting, but the same number of alignment artifacts are introduced. Pairwise analysis of model hypotheses showed similar structural divergence between model outputs on the OSCE dataset (9.0% insertion/deletion rate) and PriMock57 (8.7%), suggesting that the difference in fusion outcome is driven by individual model error rates rather than structural dissimilarity between hypotheses.

### Clinical Term Recall

On-device models achieved clinical term recall (CTR) within 1 to 3 percentage points of cloud APIs across all datasets (Table 2). On the psychiatric dataset, the best on-device models (whisper-turbo 92.0%, whisper-distil 91.9%, parakeet-tdt 91.8%) were within 1 percentage point of the best cloud API (93.0%). On the OSCE dataset, the best on-device model (whisper-turbo, 93.6%) trailed the best cloud APIs (95.5%) by 1.9 percentage points. On PriMock57, the gap was widest: parakeet-tdt (89.4%) trailed the best cloud API (92.4%) by 3.0 percentage points.

Model rankings differed between WER and CTR. Whisper-turbo, which ranked second on overall WER, achieved the highest on-device CTR on two of three datasets. Drug names exhibited the highest per-term error rates across both on-device and cloud models, including wellbutrin (60.4% error rate on the psychiatric dataset), and common medications and symptoms on the primary care and OSCE datasets. These failures were consistent across model types, suggesting that medical terminology remains an intrinsic challenge for current ASR systems regardless of deployment mode.

### Cost Analysis

Total cloud API costs for evaluating approximately 80 hours of audio across all three datasets ranged from $19.33 to $382.64 (Table 3). The five services ranked by total cost were: Azure Speech ($19.33), Deepgram Nova-2 Medical ($25.78), AssemblyAI ($28.10), AWS Transcribe Medical ($99.63, representative subsets only due to cost), and Google Cloud Speech ($382.64, which may include preliminary test runs). On-device inference incurred no per-encounter cost after initial model download (model sizes range from 37 MB to 1.5 GB). On-device processing runs on consumer mobile devices that clinicians already own and regularly update, requiring no additional hardware investment. For a practice transcribing 30 encounters per day, estimated annual cloud costs at published rates would range from approximately $2,000 to over $20,000 depending on the service, whereas on-device processing has zero marginal cost per encounter.
