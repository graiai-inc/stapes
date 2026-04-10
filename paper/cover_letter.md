Dear Editors,

No open, reproducible benchmark exists for evaluating clinical speech recognition. We submit the first such benchmark, "Open Benchmark of On-Device and Cloud ASR for Clinical Conversations," for consideration as a Datasets, Benchmarks, and Protocols article in NEJM AI.

Clinical documentation is a leading driver of physician burnout, and ambient AI scribes that depend on accurate speech recognition are being adopted rapidly by well-resourced health systems. Yet no standardized evaluation has compared privacy-preserving on-device ASR models against commercial cloud APIs on clinical conversation data. We address this gap with the first such evaluation: 14 general-purpose on-device models and 5 commercial cloud APIs across 400 publicly available clinical conversations.

Our key findings are:

First, on-device models achieve clinical term recall within 1 to 3 percentage points of cloud APIs across all three datasets, despite using general-purpose architectures while three of five cloud services deployed medical-specific models.
            
Second, we report a previously undocumented dataset-dependent failure of ROVER hypothesis fusion: fusion improved transcription on the dataset with the highest baseline error rates but degraded performance on every file across both other datasets, with implications for any clinical AI system relying on multi-model combination.

Third, drug names exhibited error rates exceeding 60% across both on-device and cloud models, indicating that medication transcription remains an unresolved patient safety concern for current ASR technology regardless of deployment mode.

By eliminating patient audio transmission and recurring API costs, on-device ASR addresses both privacy concerns and the documented disparities in ambient AI adoption between well-resourced and under-resourced institutions. We believe this benchmark addresses an important gap in the clinical AI evaluation literature and is well suited to NEJM AI's readership.

I am board-certified in Clinical Informatics (American Board of Preventive Medicine, 2022) and completed clinical informatics fellowship training at the Children's Hospital of Philadelphia. I currently serve as Assistant Professor of Hematology and Medical Oncology at Emory University School of Medicine and as Director of FHIR-FLI, an open-source international health information technology collaborative.

The graiai-inc GitHub organization hosts code from independent research projects under a name reserved for a potential future entity. It is not currently an incorporated company, has no employees, customers, or revenue, and no commercial product is associated with this work. All code and results in this submission are released under the MIT license with no commercial restrictions. I have no conflicts of interest to declare.

This manuscript has not been submitted elsewhere.

Sincerely, 
Jason Grey Faulkenberry, MD, MPH
Assistant Professor, Emory University School of Medicine  