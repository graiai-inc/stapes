Dear Editors,

I submit "Medication Names Are a Safety-Critical Failure Mode in Clinical Speech Recognition: An Open Benchmark of On-Device and Cloud Systems" for consideration in BMC Medical Informatics and Decision Making as a Research article.

The central finding is a patient-safety signal that holds regardless of where transcription runs. Across 12 on-device models and 5 cloud services evaluated on 400 clinical conversations, medication names were the highest-error clinical term category for most systems in both deployment modes: the highest-error drugs were misrecognized in 85% to 93% of on-device occurrences and 3% to 30% of cloud occurrences, and drug names remained the weakest cloud category on two of three datasets. Aggregate word error rate does not reveal this, because on-device and cloud accuracy are otherwise close. The practical implication is that any clinical speech recognition deployment, cloud or local, requires explicit medication verification before a note is finalized.

This journal published Ng et al. (2025), cited as reference 6 in this manuscript, which concluded that no standardized evaluation framework exists for clinical speech recognition. This work supplies one: an open, reproducible benchmark built entirely from publicly redistributable audio, with all code, per-model outputs, and normalization rules released under the MIT license so that other groups can extend, re-run, and critique it. The journal has also recently published work on transcription accuracy in simulated physician-patient encounters, which is the same evidentiary setting this benchmark uses.

Four findings are directly relevant to the medical informatics readership:

- Medication names were the highest-error clinical term category for most models in both deployment modes, with the per-drug failures splitting sharply by mode as described above. This is consistent with the look-alike/sound-alike (LASA) medication-error literature. A medical-dictation model reduced wellbutrin errors from 86% pooled across the on-device configurations to 5%, which shows the deficit is addressable through domain adaptation rather than intrinsic to speech recognition.

- General-purpose on-device speech recognition, running as a single model, matched cloud clinical term recall within 1 to 3 percentage points across three independent datasets, despite three of the five cloud services being medical-specific. The privacy, cost, and connectivity advantages of on-device deployment therefore do not require accepting a meaningful clinical accuracy penalty for conversation transcription. This matters because ambient AI scribes now deploying in health systems are built on cloud speech recognition that transmits, and often retains, protected health information; that dependence is a design choice, not a technical necessity.

- Because on-device accuracy already approached cloud, we tested whether fusing multiple on-device models could close the residual gap. ROVER hypothesis fusion yielded only modest improvements (at most 0.82 percentage points across the three datasets), and a depth-study of six advanced fusion algorithms (confusion-network combination with epsilon tuning, three confidence-weighted variants using per-token log probabilities, margin-weighted voting, and a learned per-word keep classifier under 5-fold cross-validation) found that none exceeded naive equal-weight ROVER by more than 0.4 percentage points. A single well-chosen on-device model is sufficient, which is a practical result given that fusion roughly doubles model-loading cost on resource-constrained devices.

- We introduce a meaning-preserving word error rate regime, reported alongside standard Whisper-normalized word error rate, that symmetrically removes orthographic variants and conversational backchannels which do not change clinical meaning. The full normalizer (eight categories, approximately 530 lines, with self-tests) is published with the code. Reporting both regimes lets reviewers and implementers separate genuine content-word errors from formatting and transcription-style differences across services, and is offered as a methodological contribution to the clinical speech recognition evaluation literature.

**Concurrent submissions.** This manuscript is not under consideration at any other journal, and no related manuscript from the author is in press or under review elsewhere.

**Use of artificial intelligence.** AI coding assistants (Anthropic Claude Code and Google Gemini Code Assist) were used for programming support during development of the evaluation scripts, data-verification tools, and fusion implementation, as disclosed in the Methods section and in the Declarations. No AI tool was used as an author or to generate the scientific content, analysis, or text of this manuscript; all scientific design, analysis decisions, interpretation, and manuscript text were authored and verified by the human author.

**Competing interests.** I have no financial conflicts of interest. The graiai-lab GitHub organization, which hosts the code accompanying this manuscript, is a personal organization used to publish my open-source research code; it is not an incorporated entity and has no employees, customers, revenue, or associated commercial product. All code, audio references, and results are released under the MIT license with no commercial restrictions.

**Data and code availability.** All datasets used (the OSCE respiratory interview corpus, PriMock57, and the Kazi et al. psychiatric dataset) are publicly redistributable under their respective licenses. All benchmarking code, evaluation scripts, and supplementary tables are released at the repository cited in the manuscript.

**Author background.** I am board-certified in Clinical Informatics (American Board of Preventive Medicine, 2022), completed clinical informatics fellowship training at the Children's Hospital of Philadelphia, and currently serve as Assistant Professor of Hematology and Medical Oncology at Emory University School of Medicine and as Director of FHIR-FLI, an open-source international health information technology collaborative.

Thank you for considering this submission.

Sincerely,

J. Grey Faulkenberry, MD, MPH
Assistant Professor, Department of Hematology and Medical Oncology
Emory University School of Medicine
36 Linden Ave NE, Atlanta, GA 30308, USA
Email: grey.faulkenberry@emory.edu
Phone: +1 404 778 1900
