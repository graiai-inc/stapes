Subject: Re: Technical check — MS fd95f4f4-d0e1-4858-a1b1-20c667d2605f v1.0 [62251619:1196208] — both DOI links verified working

Dear Shobanalakshmi,

Thank you for the technical check. I have checked both flagged links and they both seem to be working for me. I'm going to resubmit the manuscript without changes to the links, but I have included verification below.

1) https://doi.org/10.5281/zenodo.4279041: Kazi N, et al. Dataset for automated medical transcription. Zenodo, 2020. DataCite record state: findable (registered 18 Nov 2020). doi.org returns HTTP 302 to https://zenodo.org/records/4279041, which returns HTTP 200 and the record page titled "Dataset for Automated Medical Transcription".

2) https://doi.org/10.5281/zenodo.22081372: Faulkenberry JG. stapes (v1.0.0). Zenodo, 2026. DataCite record state: findable (registered 24 Aug 2026, 12:48 UTC). doi.org returns HTTP 302 to https://zenodo.org/records/22081372, which returns HTTP 200 and the record page titled "stapes: an open benchmark of on-device and cloud speech recognition for clinical conversations".

I checked both on 25 Aug 2026 at 11:51 UTC by GET and by HEAD request and both returned HTTP 200. The DOI registrations can be confirmed through DataCite, which should be available as well:
   https://api.datacite.org/dois/10.5281/zenodo.4279041
   https://api.datacite.org/dois/10.5281/zenodo.22081372

If this still does not work, would you be able to send me the exact failure code? Zenodo applies bot protection to its landing pages and intermittently returns HTTP 403 to automated link checkers while serving the same pages normally to browsers. This affects automated validation, but the link is still available. This is consistent with the fact that the third Zenodo link in the same statement (https://doi.org/10.5281/zenodo.22081371, the concept DOI) was not flagged, demonstrating the intermittent rate limiting rather than with the records being unavailable.

These two DOIs are the persistent identifiers required for the cited dataset and for the archived analysis code, so I would prefer to retain them. If your system continues to flag them, I am happy
to add the resolved record URLs (https://zenodo.org/records/4279041 and https://zenodo.org/records/22081372) alongside the DOIs — please let me know and I will make that change immediately.

Regards,

Grey Faulkenberry
Emory University
grey@fhirfli.dev
