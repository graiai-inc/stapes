## Conclusion

On-device ASR models approach the clinical term recall of commercial cloud APIs while eliminating patient audio transmission and recurring costs. In our evaluation, ROVER fusion produced only small WER improvements (≤ 0.82 percentage points) over the best single on-device model across all three datasets, and three-model combinations offered no additional benefit, suggesting that domain adaptation of a single model is a more promising path than multi-model fusion for clinical on-device ASR. Until medication transcription accuracy improves, clinical deployment of any ASR system requires robust verification workflows. By running on devices clinicians already own, on-device ASR may democratize access to documentation technology for settings where cloud services are unaffordable or legally prohibited, while addressing privacy concerns paramount in sensitive clinical specialties.

## Data and Code Availability

All evaluation code, model inference scripts, ROVER fusion implementation, and per-file results are publicly available at github.com/graiai-inc/stapes. The three clinical conversation datasets are available under CC-BY licenses at their respective repositories. UMLS 2025AB requires a free license from the National Library of Medicine.
