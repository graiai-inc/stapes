## Conclusion

On-device ASR models approach the clinical term recall of commercial cloud APIs while eliminating patient audio transmission and recurring costs. In our evaluation, ROVER fusion improved transcription only where models had the highest baseline error rates and degraded performance where models were already accurate, suggesting that the benefit of multi-model fusion depends on baseline model performance. Until medication transcription accuracy improves, clinical deployment of any ASR system requires robust verification workflows. By running on devices clinicians already own, on-device ASR may democratize access to documentation technology for settings where cloud services are unaffordable or legally prohibited, while addressing privacy concerns paramount in sensitive clinical specialties.

## Data and Code Availability

All evaluation code, model inference scripts, ROVER fusion implementation, and per-file results are publicly available at github.com/graiai-inc/stapes. The three clinical conversation datasets are available under CC-BY licenses at their respective repositories. UMLS 2025AB requires a free license from the National Library of Medicine.
