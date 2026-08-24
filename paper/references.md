## References

### Introduction claims and citations:

"Physicians spend more than half their working hours on documentation, and administrative burden is a leading contributor to burnout."
→ [1] Sahni, [2] Shah

"Recent evaluations report reductions in after-hours documentation time and improvements in clinician satisfaction"
→ [3] Stults, [4] Olson

"one large cohort study of a commercial ambient scribe found no significant time savings and worsened after-hours EHR use"
→ [5] Haberle

"word error rates range from under 1% for controlled dictation to over 50% for naturalistic clinical speech"
→ [6] Ng

"Some ambient scribe services retain full audio recordings on remote servers for weeks after the encounter"
→ [7] Elsayed (socio-technical risks of clinical STT)

"BAAs...do not eliminate the privacy risks...recent analyses have raised broader concerns about consent, transparency, and data governance"
→ [7] Elsayed, [8] Anderson

"Afonja et al. introduced a medical word error rate"
→ [9] Afonja

"The systematic review by Ng et al. noted the absence of a standardized evaluation framework"
→ [6] Ng

"the socio-technical risks of clinical speech-to-text systems...remain largely unaddressed"
→ [7] Elsayed

### Methods citations:

"OSCE respiratory interview dataset"
→ [10] Fareez

"PriMock57"
→ [11] Korfiatis

"Kazi et al. psychiatric dataset"
→ [12] Kazi

"HuggingFace Open ASR Leaderboard...and MLPerf Inference ASR benchmarks"
→ [13] Srivastav

"ROVER (Fiscus, 1997)"
→ [14] Fiscus

"confusion network combination (Mangu et al., 2000)"
→ [15] Mangu

"QuickUMLS"
→ [16] Soldaini

### Discussion citations:

"5.7 percent of errors in speech-recognition-generated clinical documents are clinically significant, with medication the most common clinical error category"
→ [17] Zhou

"A 2026 national study found that ambient AI adoption in U.S. hospitals is concentrated"
→ [23] Yang & Graetz

"Clinical deployment may fall under FDA oversight"
→ [22] Warraich

"For privacy-sensitive specialties such as psychiatry, substance abuse treatment"
→ [7] Elsayed (already cited)

"overlapping speech, background noise, and non-lexical conversational sounds"
→ [26] Tran, [27] Alboksmaty

"automatic cross-referencing of ASR output against patient medication lists and institutional formularies"
→ [20] Rash-Foanio

"flagging phonetically similar drug names for mandatory clinician review"
→ [18] Lambert, [19] Bryan

"requiring explicit confirmation before finalizing notes containing high-risk medications"
→ [17] Zhou (already cited)

"transcription errors propagate into the downstream LLM-generated note... AI-generated notes scored lower... clinicians edit AI drafts"
→ [24] Reddy, [25] Guo

"a 2025 study of consent practices for ambient documentation found wide variation in how consent is obtained"
→ [21] Lawrence

---

## Formatted Reference List

1. Sahni NR, Carrus B. Artificial intelligence in U.S. health care delivery. N Engl J Med. 2023;389(4):348-58. doi:10.1056/NEJMra2204673.

2. Shah SJ, Devon-Sand A, Ma SP, Jeong Y, Crowell T, Smith M, et al. Ambient artificial intelligence scribes: physician burnout and perspectives on usability and documentation burden. J Am Med Inform Assoc. 2025;32(2):375-80. doi:10.1093/jamia/ocae295.

3. Stults CD, Deng S, Martinez MC, Wilcox J, Szwerinski N, Chen KH, et al. Evaluation of an ambient artificial intelligence documentation platform for clinicians. JAMA Netw Open. 2025;8(5):e258614. doi:10.1001/jamanetworkopen.2025.8614.

4. Olson KD, Meeker D, Troup M, Barker TD, Nguyen VH, Manders JB, et al. Use of ambient AI scribes to reduce administrative burden and professional burnout. JAMA Netw Open. 2025;8(10):e2534976. doi:10.1001/jamanetworkopen.2025.34976.

5. Haberle T, Cleveland C, Snow GL, Barber C, Stookey N, Thornock C, et al. The impact of Nuance DAX ambient listening AI documentation: a cohort study. J Am Med Inform Assoc. 2024;31(4):975-9. doi:10.1093/jamia/ocae022.

6. Ng JJW, Wang E, Zhou X, Zhou KX, Goh CXL, Sim GZN, et al. Evaluating the performance of artificial intelligence-based speech recognition for clinical documentation: a systematic review. BMC Med Inform Decis Mak. 2025;25(1):236. doi:10.1186/s12911-025-03061-0.

7. Elsayed N. Socio-technical risks of clinical speech-to-text systems: transparency, privacy, and reliability challenges in AI-driven documentation. Int J Med Inform. 2026;214:106419. doi:10.1016/j.ijmedinf.2026.106419.

8. Anderson TN, Mohan V, Gold JA. Ethical considerations for clinical adoption of ambient digital scribe technology. J Am Med Inform Assoc. 2026;33(3):770-5. doi:10.1093/jamia/ocaf227.

9. Afonja T, Olatunji T, Ogun S, Etori NA, Owodunni A, Yekini M. Performant ASR models for medical entities in accented speech. In: Proceedings of Interspeech 2024. Kos: ISCA; 2024. p. 2315-9. doi:10.21437/Interspeech.2024-2261.

10. Fareez F, Parikh T, Wavell C, Shahab S, Chevalier M, Good S, et al. A dataset of simulated patient-physician medical interviews with a focus on respiratory cases. Sci Data. 2022;9(1):313. doi:10.1038/s41597-022-01423-1.

11. Papadopoulos Korfiatis A, Moramarco F, Sarac R, Savkov A. PriMock57: a dataset of primary care mock consultations. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (volume 2: short papers). Dublin: Association for Computational Linguistics; 2022. p. 588-98. doi:10.18653/v1/2022.acl-short.65.

12. Kazi N, Kuntz M, Kanewala U, Kahanda I. Dataset for automated medical transcription. Zenodo. 2020. https://doi.org/10.5281/zenodo.4279041.

13. Srivastav V, Zheng S, Bezzam E, Le Bihan E, Koluguri NR, Żelasko P, et al. Open ASR leaderboard: towards reproducible and transparent multilingual and long-form speech recognition evaluation. arXiv [Preprint]. 2025. doi:10.48550/arXiv.2510.06961.

14. Fiscus JG. A post-processing system to yield reduced word error rates: recognizer output voting error reduction (ROVER). In: Proceedings of the 1997 IEEE Workshop on Automatic Speech Recognition and Understanding. Santa Barbara: IEEE; 1997. p. 347-54. doi:10.1109/ASRU.1997.659110.

15. Mangu L, Brill E, Stolcke A. Finding consensus in speech recognition: word error minimization and other applications of confusion networks. Comput Speech Lang. 2000;14(4):373-400. doi:10.1006/csla.2000.0152.

16. Soldaini L, Goharian N. QuickUMLS: a fast, unsupervised approach for medical concept extraction. In: Proceedings of the Medical Information Retrieval Workshop (MedIR) at SIGIR 2016. Pisa; 2016.

17. Zhou L, Blackley SV, Kowalski L, Doan R, Acker WW, Landman AB, et al. Analysis of errors in dictated clinical documents assisted by speech recognition software and professional transcriptionists. JAMA Netw Open. 2018;1(3):e180530. doi:10.1001/jamanetworkopen.2018.0530.

18. Lambert BL, Dickey LW, Fisher WM, Gibbons RD, Lin SJ, Luce PA, et al. Listen carefully: the risk of error in spoken medication orders. Soc Sci Med. 2010;70(10):1599-608. doi:10.1016/j.socscimed.2010.01.042.

19. Bryan R, Aronson JK, Williams A, Jordan S. The problem of look-alike, sound-alike name errors: drivers and solutions. Br J Clin Pharmacol. 2021;87(2):386-94. doi:10.1111/bcp.14285.

20. Rash-Foanio C, Galanter W, Bryson M, Falck S, Liu KL, Schiff GD, et al. Automated detection of look-alike/sound-alike medication errors. Am J Health Syst Pharm. 2017;74(7):521-7. doi:10.2146/ajhp150690.

21. Lawrence K, Kuram VS, Levine DL, Sharif S, Polet C, Malhotra K, et al. Informed consent for ambient documentation using generative AI in ambulatory care. JAMA Netw Open. 2025;8(7):e2522400. doi:10.1001/jamanetworkopen.2025.22400.

22. Warraich HJ, Tazbaz T, Califf RM. FDA perspective on the regulation of artificial intelligence in health care and biomedicine. JAMA. 2025;333(3):241-7. doi:10.1001/jama.2024.21451.

23. Yang F, Graetz I. Ambient AI tool adoption in US hospitals and associated factors. Am J Manag Care. 2026;32(1):e25-30. doi:10.37765/ajmc.2026.89876.

24. Reddy A, Gunnink E, Wheat CL, Pawlikowski S, Payne CM, Wiltz S, et al. Rapid evaluation of artificial intelligence technology used for ambient dictation in primary care: comparing the quality of documentation of artificial intelligence-generated and human-produced clinical notes. Ann Intern Med. 2026;179(6):765-72. doi:10.7326/ANNALS-25-02772.

25. Guo Y, Hu D, Yang Z, Chow E, Tam S, Perret D, et al. Clinicians' rationale for editing ambient AI-drafted clinical notes: persistent challenges and implications for improvement. J Am Med Inform Assoc. 2026;33(7):1345-53. doi:10.1093/jamia/ocag059.

26. Tran BD, Latif K, Reynolds TL, Park J, Elston Lafata J, Tai-Seale M, et al. "Mm-hm," "Uh-uh": are non-lexical conversational sounds deal breakers for the ambient clinical documentation technology? J Am Med Inform Assoc. 2023;30(4):703-11. doi:10.1093/jamia/ocad001.

27. Alboksmaty A, Aldakhil R, Hayhoe BW, Ashrafian H, Darzi A, Neves AL. The impact of using AI-powered voice-to-text technology for clinical documentation on quality of care in primary care and outpatient settings: a systematic review. EBioMedicine. 2025;118:105861. doi:10.1016/j.ebiom.2025.105861.
