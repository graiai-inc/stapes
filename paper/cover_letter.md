Dear Editors of npj Digital Medicine,

We submit "Open Benchmark of On-Device and Cloud ASR for Clinical Conversations" for your consideration as an Article.

Ambient AI clinical scribes are being adopted rapidly by well-resourced health systems while under-resourced settings are left behind, and privacy, cost, and connectivity are cited as principal barriers. The evidence base that clinicians, informaticists, and health-system leaders need to reason about these tradeoffs does not yet exist: to our knowledge, no prior study has benchmarked modern privacy-preserving on-device ASR against commercial cloud APIs on clinical conversation data. We built that benchmark using only publicly redistributable audio and code, so that others can reproduce, extend, and critique it. We believe this directly supports npj Digital Medicine's interest in rigorous, open evaluation of digital health technologies and in work that speaks to equitable deployment.

Beyond filling the evaluation gap, we report three findings that we think will be useful to the field regardless of which deployment mode a reader favors:

- General-purpose on-device ASR matches cloud clinical term recall within 1 to 3 percentage points across three datasets, despite three of the five cloud services being medical-specific. This is a genuinely surprising result given how the market has been priced and positioned.
- ROVER hypothesis fusion yields only modest WER improvements over the best single on-device model (≤ 0.82 percentage points across the three datasets), and adding a third model did not help on either dataset where an exhaustive pair-versus-triple search was feasible. Because reviewers naturally ask whether a more sophisticated fusion algorithm would change this conclusion, we added a depth-study (Supplementary Table S1) evaluating six advanced fusion algorithms on the strongest on-device pair on PriMock57 — confusion network combination with epsilon tuning, three confidence-weighted variants using per-token log probabilities (mean log-probability, Shannon entropy, Tsallis entropy), margin-weighted voting, and a learned per-word keep classifier trained via 5-fold cross-validation — and none exceeded naive equal-weight ROVER. This reframes fusion in the clinical on-device setting as a narrow efficiency/accuracy tradeoff rather than a frontier of easy gains.
- Drug names exhibit error rates above 60% across both on-device and cloud systems, including on cloud services marketed as medical-specific. This is a patient-safety signal that deserves attention independent of where transcription runs.

**Related work.** The manuscript is not under consideration at any other journal, and no related manuscript from our group is in press or under review elsewhere.

**Competing interests.** I have no financial conflicts of interest to declare. The graiai-inc GitHub organization hosts code from independent research projects under a name reserved for a potential future entity; it is not an incorporated company, has no employees, customers, or revenue, and no commercial product is associated with this work. All code, audio references, and results are released under the MIT license with no commercial restrictions.

**Data and code availability.** All datasets used are publicly available (OSCE respiratory interview corpus, PriMock57, and the Kazi et al. psychiatric dataset). All benchmarking code, evaluation scripts, and supplementary tables are released at the repository cited in the manuscript.

**Author background.** I am board-certified in Clinical Informatics (American Board of Preventive Medicine, 2022), completed clinical informatics fellowship training at the Children's Hospital of Philadelphia, and currently serve as Assistant Professor of Hematology and Medical Oncology at Emory University School of Medicine and as Director of FHIR-FLI, an open-source international health information technology collaborative.

Thank you for considering this submission.

Sincerely,

Jason Grey Faulkenberry, MD, MPH
Assistant Professor, Department of Hematology and Medical Oncology
Emory University School of Medicine
Email: grey@fhirfli.dev
[Institutional email, phone, and mailing address to be completed]
