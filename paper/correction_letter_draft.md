Subject: Correction request before peer review — AI-26-00468

Dear Editor,

During routine post-submission self-review of our code for "Open Benchmark of On-Device and Cloud ASR for Clinical Conversations" (AI-26-00468, received 10-Apr-2026), we identified a computational error affecting the WER analysis. As the manuscript is still under initial editorial review, we are writing to request that we be permitted to submit a corrected version before it goes out to peer reviewers.

The error has two linked components. First, our ROVER fusion script stripped apostrophes from hypothesis text prior to word-level alignment; because the Whisper text normalizer used for WER expands contractions ("I'm" to "i am"), every contracted word in the reference produced two spurious errors in fused WER on any dataset containing apostrophes. Second, the figshare-osce reference transcripts had been distributed with apostrophes fully removed ("IM IM JUST"), which interacted with the tokenization bug to make fusion appear anomalously strong on that dataset. We corrected the reference using a 44-token contraction-injection dictionary validated against the hypotheses, and fixed the fusion tokenization.

Three conclusions change under the corrected analysis, and we believe reviewers should see the corrected version:

- PriMock57: ROVER fusion modestly improves over best solo (13.03% vs 13.85%, −0.82pp). Original paper reported degradation; direction inverts.
- Nazmulkazi (psychiatric): fusion modestly improves over best solo (7.15% vs 7.30%, −0.15pp). Direction inverts.
- figshare-osce: Azure (7.70%) and Deepgram (10.10%) both outperform our best fused result (11.01%). The headline claim that our fusion beat all cloud APIs on this dataset no longer holds.

Solo WERs on PriMock57 and Nazmulkazi, all methodology, dataset and model descriptions, and the inference speed, CTR, and cost analyses are unaffected.

We can provide the corrected manuscript either as a file attachment to this submission or as a resubmission under the same ID, whichever your editorial office prefers. Please let us know how to proceed.

Thank you for your consideration.

Sincerely,
[author name]
