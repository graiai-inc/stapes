## Abstract

**Background:** Clinical speech recognition increasingly underpins ambient AI documentation systems, yet no open benchmark compares on-device models against commercial cloud APIs on medical conversation data. On-device inference eliminates patient data transmission but its accuracy relative to cloud services is unknown.

**Methods:** We evaluated 14 on-device ASR models (via the sherpa-onnx inference engine) and 5 cloud APIs (Azure, Google, Deepgram, AssemblyAI, AWS) across three publicly available clinical conversation datasets (400 conversations, ~80 hours): simulated OSCE examinations, mock primary care consultations, and enacted psychiatric interviews. We measured word error rate (WER), medical term recall using UMLS concept matching, ROVER hypothesis fusion across all model pairs, and cloud API costs.

**Results:** The best on-device models achieved WER within 0.5 to 4.0 percentage points of the best cloud APIs and medical term recall within 1 to 3 percentage points. ROVER fusion improved performance on the OSCE dataset (reducing the best on-device WER from 17.6% to 10.7%, outperforming all cloud APIs) but degraded performance on both other datasets (0 of 128 files improved). Drug names exhibited the highest error rates across both on-device and cloud models. Cloud API costs for the benchmark ranged from $19 to $383; on-device inference incurred no per-encounter cost.

**Conclusions:** On-device ASR approaches cloud-level medical term accuracy while eliminating data transmission and recurring costs. ROVER fusion benefits are dataset-dependent, a finding not previously reported. All code and results are publicly available at github.com/graiai-inc/stapes.
