Dear Editor,

I submit "Open Benchmark of On-Device and Cloud Speech Recognition for Clinical Conversations: Parity in Accuracy, Persistent Errors in Medication Names" for consideration in the Journal of the American Medical Informatics Association as a Research and Applications article.

Speech recognition is the foundation of every ambient AI clinical scribe currently being deployed in U.S. health systems, yet no open, reproducible benchmark has compared the privacy-preserving on-device models that have emerged in the past two years against the established cloud services dominating commercial deployment. JAMIA has published the two most-cited systematic reviews of clinical speech recognition (Blackley et al. 2019; Hodgson and Coiera 2016), which together documented weak and inconsistent prior evidence and significant gaps in standardized evaluation. This benchmark contributes open, reproducible empirical evidence in that gap, built using only publicly redistributable audio and code so that other groups can extend and critique it.

Four findings are directly relevant to the clinical informatics readership:

- General-purpose on-device ASR matched cloud clinical term recall within 1 to 3 percentage points across three independent datasets, despite three of the five cloud services being medical-specific. This indicates that the privacy, cost, and connectivity advantages of on-device deployment do not require accepting a meaningful clinical accuracy penalty for conversation transcription.

- Medication names exhibited the highest error rates of any clinical term category across every model and every deployment mode, with individual drugs misrecognized in more than 60% of occurrences, including by cloud services marketed as medical-specific. This is a patient-safety signal consistent with the look-alike/sound-alike (LASA) medication-error literature and is invariant to deployment choice. We argue that any clinical ASR deployment, regardless of mode, requires explicit medication verification before notes are finalized.

- ROVER hypothesis fusion yielded only modest WER improvements (≤ 0.82 percentage points across the three datasets), and a depth-study evaluating six advanced fusion algorithms (confusion-network combination with epsilon tuning, three confidence-weighted variants using per-token log probabilities, margin-weighted voting, and a learned per-word keep classifier under 5-fold cross-validation) found that none exceeded naive equal-weight ROVER. This reframes multi-model fusion in the clinical on-device setting as a narrow efficiency tradeoff rather than a frontier of easy accuracy gains.

- We introduce a meaning-preserving WER regime, reported alongside standard Whisper-normalized WER, that symmetrically removes orthographic variants and conversational backchannels which do not change clinical meaning. The full normalizer (eight categories, ~300 lines, with self-tests) is published with the code. Reporting both regimes allows reviewers and implementers to separate genuine content-word errors from formatting and transcription-style differences across services, and is offered as a methodological contribution to clinical ASR evaluation literature.

**Concurrent submissions.** This manuscript is not under consideration at any other journal, and no related manuscript from the author is in press or under review elsewhere.

**Use of AI tools.** AI coding assistants (Anthropic Claude Code and Google Gemini Code Assist) were used for programming support during the development of evaluation scripts, data verification tools, and fusion implementation code, as disclosed in the Materials and Methods section. No AI tool was used as an author or to generate the scientific content, analysis, or text of this manuscript; all scientific design, analysis decisions, interpretation, and manuscript text were authored and verified by the human author.

**Competing interests.** I have no financial conflicts of interest. The graiai-inc GitHub organization, which hosts the code accompanying this manuscript, is a name reservation for a potential future entity; it is not incorporated, has no employees, customers, or revenue, and no commercial product is associated with this work. All code, audio references, and results are released under the MIT license with no commercial restrictions.

**Data and code availability.** All datasets used (the OSCE respiratory interview corpus, PriMock57, and the Kazi et al. psychiatric dataset) are publicly redistributable under their respective licenses. All benchmarking code, evaluation scripts, and supplementary tables are released at the repository cited in the manuscript.

**Author background.** I am board-certified in Clinical Informatics (American Board of Preventive Medicine, 2022), completed clinical informatics fellowship training at the Children's Hospital of Philadelphia, and currently serve as Assistant Professor of Hematology and Medical Oncology at Emory University School of Medicine and as Director of FHIR-FLI, an open-source international health information technology collaborative.

Thank you for considering this submission.

Sincerely,

J. Grey Faulkenberry, MD, MPH
Assistant Professor, Department of Hematology and Medical Oncology
Emory University School of Medicine
Email: grey.faulkenberry@emory.edu
Phone: +1 404 778 1900
36 Linden Ave NE, GA 30308
