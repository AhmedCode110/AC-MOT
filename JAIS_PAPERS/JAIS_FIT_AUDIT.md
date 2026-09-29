# Journal of Aerospace Information Systems — fit audit

Source for rules: AIAA "Journal Page Limits and Word Count Guidelines", Rev. August 2024
(`refs/aiaa_rules/limits_2024.txt`, fetched from aiaa.org); AIAA open-access page (`refs/aiaa_rules/open_access.txt`).

| Requirement | Paper 1 | Paper 2 |
|---|---|---|
| Regular article length: 7–10 published pages; 20–26 double-spaced 10-pt manuscript pages (all pages counted) | 24 pages | 20 pages (AIAA version; submitted version is the Springer one) |
| Publication fees | none unless Open Access (voluntary $2,700) | same |
| AIAA template (`new-aiaa.cls`, journal option) | yes | yes |
| Title ≤ 12 words, no abbreviations | 10 words | 9 words |
| Abstract 100–200 words | 179 | 193 |
| Nomenclature | yes | yes |
| Aerospace relevance | direct: unmanned aerial vehicle video, onboard-style processing throughput, UAVDT/VisDrone | indirect: detector/tracker exchange in airborne perception pipelines; evidence is ground-level video (MOT17, KITTI, DanceTrack) |
| Journal literature cited | 5 JAIS papers (UAV tracking, onboard detection, docking, refueling imagery) | same 5 |

## Assessment
- Paper 1 fits the journal's scope well (airborne sensing, real-time information processing on unmanned aircraft
  video). Its main risk is the attribution result (switching adds nothing over a matched static point), which a
  reviewer may read as a negative result; the paper frames it as the finding and gives the regime in which switching helps.
- Paper 2 fits less directly: no aerial result exists for the frozen layer. The motivation (component exchange in
  aerospace pipelines) is sound, but a reviewer is likely to ask for aerial evidence. The aerial run (VisDrone
  val-7/confirmation-16/test-dev) is fully scripted and only waits for the dataset download; adding it before
  submission would materially strengthen the fit. If it cannot be added, a computer-vision venue may be the better target.
