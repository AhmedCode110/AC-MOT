# KNOWLEDGE GRAPH SOURCE — typed entities and relations for Graphify

Parsed by `tools/build_context_graph.py` (rows of the two tables below only).
Entity names are `Type: Name` and must be reused verbatim. Relations are
UPPER_SNAKE_CASE. Every row must be backed by the file in "Defined in".
Edit this file when the project state changes, then rebuild the graph.

## Entities
| Entity | Status | Summary | Defined in |
|---|---|---|---|
| Project: Universal AC-MOT | CURRENT | adaptive control layer around frozen detector and frozen tracker for drone multi-object tracking | PROJECT_STATE.md |
| Version: Legacy AC-MOT | FROZEN SUPERSEDED | original handcrafted SCI controller YOLOv8n ByteTrack 2026-09-11 | ARCHITECTURE.md |
| Version: V1 | FROZEN SUPERSEDED | histogram normaliser ratio gate legacy SCI not temperature invariant | FROZEN_VERSIONS.md |
| Version: V2 variants | SUPERSEDED | V2b V2c V2d V2e V2f density budgets reliability feedback all failed | FAILED_EXPERIMENTS.md |
| Version: V3 | FROZEN SUPERSEDED | calibration-invariant ECDF z-logit gate with legacy SCI | FROZEN_VERSIONS.md |
| Version: V4 | FROZEN BASELINE ONLY | latest frozen version; baseline and ablation only, never the final system or a fallback; compute-budget-only, VisDrone-tuned constants | FROZEN_VERSIONS.md |
| Version: V5 learned controller | SUPERSEDED RESEARCH-ONLY | scene-state learned stump tree Optuna controller research upper bound not deployable | DECISIONS.md |
| Version: V5-TF | FINAL TARGET EXPERIMENTAL | final target and contribution: training-free online self-calibrating scene-state adaptive detector-agnostic tracker-agnostic real-time causal; under development, not frozen | ARCHITECTURE.md |
| Architecture: Original SCI architecture | SUPERSEDED | scene analysis SCI weighted sum resolution and sensitivity controller | ARCHITECTURE.md |
| Architecture: V4 compute-budget architecture | ABLATION | V4 is not the final architecture: compute-budget-only ablation, scene adaptation deletion rejected (D-009), VisDrone-tuned category E constants; resolution budget ECDF z-gate | ARCHITECTURE.md |
| Architecture: V5-TF target architecture | CURRENT | frame scene state analyzer online self-calibration AC controller before detector candidate handling tracker feedback | ARCHITECTURE.md |
| Component: Scene State Analyzer | EXPERIMENTAL | causal image motion detector tracker cues frames before t | ARCHITECTURE.md |
| Component: Online Self-Calibration | EXPERIMENTAL | rolling median MAD robust z ECDF Otsu over causal windows no training | ARCHITECTURE.md |
| Component: Universal AC Controller | EXPERIMENTAL | F3 selected by E36 but fixed Otsu bins and memory block freeze under C5 | ARCHITECTURE.md |
| Component: Compute Latency Constraint | CURRENT | resolution budget 640 736 832 largest level meeting target FPS | ARCHITECTURE.md |
| Component: ECDF Normaliser | CURRENT | order-only causal score normalisation exact Platt temperature invariance | ARCHITECTURE.md |
| Component: Otsu-3 Candidate Bands | EXPERIMENTAL | three-class Otsu on candidate logits primary extend-only discard | ARCHITECTURE.md |
| Component: Z-logit Leader Gate | FROZEN | V3 V4 gate demote candidates below EMA leader tau 0.75 | ARCHITECTURE.md |
| Component: Legacy SCI | REJECTED | old SCI handcrafted scene complexity index removed; old SCI cues rejected crowd tiny edges brightness darkness blur weights | DECISIONS.md |
| Component: Detector Adapter | CURRENT | detector-agnostic interface score floor 0.01 native suppression | ARCHITECTURE.md |
| Component: Tracker Adapter | CURRENT | tracker-agnostic interface native defaults set_retention association tolerance | ARCHITECTURE.md |
| Rule: F1 Otsu window bands | CANDIDATE | three-class Otsu on logits of causal window frames before t | ARCHITECTURE.md |
| Rule: F2 Otsu frame bands | CANDIDATE | three-class Otsu within frame t candidates | ARCHITECTURE.md |
| Rule: F3 motion-aware association | SELECTED EXPERIMENTAL | E36-selected current candidate F1 plus association tolerance scaled by motion ratio; category-E Otsu constants block freeze | DECISIONS.md |
| Rule: F5 scene-adaptive resolution R-res | FAILED DEVELOPMENT | F3 plus object-size ECDF resolution; E36 worse than F3 and random F5R at matched compute | FAILED_EXPERIMENTS.md |
| Rule: F5R random resolution control | CONTROL | F3 plus random resolution levels same budget guard not selectable | ARCHITECTURE.md |
| Rule: F4 online z-gate | DROPPED | identical to F1 by construction | DECISIONS.md |
| Cue: det_gap | CANDIDATE | detector leader gap signal beyond null both detectors | DECISIONS.md |
| Cue: det_count | CANDIDATE | detector candidate count density signal | DECISIONS.md |
| Cue: trk_survival | CANDIDATE | tracker survival association state | DECISIONS.md |
| Cue: trk_match | CANDIDATE | tracker match behaviour | DECISIONS.md |
| Cue: img_motion | CANDIDATE | global image motion used by F3 as ratio to rolling median | DECISIONS.md |
| Cue: img_motion_resp | CANDIDATE | motion response phase correlation peak | DECISIONS.md |
| Cue: img_edges | REJECTED | edge density rejected negative everywhere legacy SCI cue | DECISIONS.md |
| Cue: img_brightness | REJECTED | brightness darkness rejected legacy SCI cue | DECISIONS.md |
| Cue: img_blur | REJECTED | blur Laplacian rejected legacy SCI cue | DECISIONS.md |
| Cue: crowd count n/30 | REJECTED | legacy SCI crowd cue sign flips below random | DECISIONS.md |
| Cue: tiny object area | REJECTED | legacy SCI tiny cue no consistent benefit | DECISIONS.md |
| Cue: temporal persistence | REJECTED | high for clutter scene property not failure signal | FAILED_EXPERIMENTS.md |
| Parameter: RobustHistory window 100 | CATEGORY A | E39 insensitive at 50 and 200 both detectors no added catastrophic cell | PARAMETER_STATUS.md |
| Parameter: RobustHistory warm-up 5 | CATEGORY A | E39 insensitive at 3 and 10 | PARAMETER_STATUS.md |
| Parameter: Otsu window 10 frames | CATEGORY E FREEZE BLOCKER | E39 sensitive default retained only for recorded F3 not deployable under C5 | PARAMETER_STATUS.md |
| Parameter: OTSU_BINS 64 | CATEGORY E FREEZE BLOCKER | E39 sensitive default retained only for recorded F3 not deployable under C5 | PARAMETER_STATUS.md |
| Parameter: Z_REF 0.75 | CATEGORY E | equals V4 tau logging only must not enter V5-TF rule | PARAMETER_STATUS.md |
| Parameter: V4 tau 0.75 | CATEGORY E FOR V5-TF | VisDrone-selected gate threshold V4 only | PARAMETER_STATUS.md |
| Parameter: V4 sensitivity 0.4 | CATEGORY E FOR V5-TF | VisDrone-selected V4 only | PARAMETER_STATUS.md |
| Parameter: V4 association offset 0.10 | CATEGORY E FOR V5-TF | VisDrone-selected V4 only | PARAMETER_STATUS.md |
| Parameter: V4 NMS 0.45 | CATEGORY E FOR V5-TF | YOLO-derived V4 only V5-TF uses detector-native suppression | PARAMETER_STATUS.md |
| Parameter: V4 tracker 45/0.86 | CATEGORY E FOR V5-TF | legacy retention match V4 only | PARAMETER_STATUS.md |
| Parameter: F3 cap 0.95 | SAFETY BOUND | generic engineering bound category C | PARAMETER_STATUS.md |
| Parameter: native tracker defaults 30/0.8 | API | category D tracker defaults used by V5-TF | PARAMETER_STATUS.md |
| Parameter: fidelity gate thresholds | PROTOCOL CONSTANT | never relaxed after seeing gate | PARAMETER_STATUS.md |
| Detector: YOLOv8n | DEVELOPMENT | COCO pretrained frozen development detector | ENVIRONMENT_AND_PATHS.md |
| Detector: RT-DETR-L | DEVELOPMENT | COCO pretrained frozen NMS-free development detector | ENVIRONMENT_AND_PATHS.md |
| Detector: Faster R-CNN ResNet50-FPN v2 | PROTECTED | unseen detector transfer test no quality metrics before freeze | PROTECTED_EVALUATIONS.md |
| Tracker: ByteTrack | DEVELOPMENT | development tracker frozen | ENVIRONMENT_AND_PATHS.md |
| Tracker: BoT-SORT | PROTECTED | tracker transfer test for V5-TF | PROTECTED_EVALUATIONS.md |
| Dataset: VisDrone2019-MOT | USED | drone MOT benchmark train val test-dev | DATASETS_AND_SPLITS.md |
| Dataset: UAVDT | PROTECTED | unseen dataset transfer test | DATASETS_AND_SPLITS.md |
| Split: VisDrone2019-MOT-val | DEVELOPMENT NOT CLEAN | V1-V5 development secondary check only for V5-TF | DATASETS_AND_SPLITS.md |
| Split: VisDrone2019-MOT-train development-40 | DEVELOPMENT | V5-TF rule validation validate never fit | DATASETS_AND_SPLITS.md |
| Split: VisDrone2019-MOT-train confirmation-16 | PROTECTED | untouched evaluate once after V5-TF freeze | DATASETS_AND_SPLITS.md |
| Split: VisDrone2019-MOT-test-dev | HELD-OUT USED ONCE | V4 evaluated once post-hoc only for V5-TF | DATASETS_AND_SPLITS.md |
| Split: UAVDT test | PROTECTED | 20 sequences unseen dataset transfer | DATASETS_AND_SPLITS.md |
| Metric: HOTA | EVALUATION | TrackEval 12c8791 | VALIDATION_PROTOCOL.md |
| Metric: IDF1 | EVALUATION | motmetrics 1.4.0 | VALIDATION_PROTOCOL.md |
| Metric: MOTA | EVALUATION | motmetrics catastrophic cell MOTA below 0 | VALIDATION_PROTOCOL.md |
| Metric: IDS | EVALUATION | identity switches motmetrics | VALIDATION_PROTOCOL.md |
| Experiment: E17 signal audit | DONE | leader-relative score separates TP from clutter temporal persistence rejected | EXPERIMENT_REGISTRY.md |
| Experiment: E24-E26 SCI audit | DONE | legacy SCI no better than random allocation at matched compute | EXPERIMENT_REGISTRY.md |
| Experiment: E31 V4 test-dev held-out | DONE HELD-OUT | V4 evaluated once YOLO better RT-DETR below shared static | RESULTS_CANONICAL.md |
| Experiment: E32 S1 headroom | DONE DEVELOPMENT | headroom for tau association resolution sensitivity retention none | EXPERIMENT_REGISTRY.md |
| Experiment: E33 S2 cue utility | DONE DEVELOPMENT | candidate cues det_gap det_count trk_survival trk_match motion | EXPERIMENT_REGISTRY.md |
| Experiment: E34-E35 S3 learned controller | FAILED | learned controllers lose to V4 on val overfitting by scarcity | FAILED_EXPERIMENTS.md |
| Experiment: E36 V5-TF dev validation | DONE DEVELOPMENT | F3 selected; F5 worse than F3 and random F5R; F3 does not improve worst-detector criterion vs V4 | EXPERIMENT_REGISTRY.md |
| Experiment: E37 discovery S1-S3 on train | PLANNED OPTIONAL | research upper bound only never final fit | EXPERIMENT_REGISTRY.md |
| Experiment: E38 T4 fidelity gate | PENDING | Mac MPS vs Colab T4 cache fidelity before freeze | EXPERIMENT_REGISTRY.md |
| Experiment: E39 constant audit | DONE DEVELOPMENT | history window and warm-up insensitive A; Otsu bins and window sensitive E freeze blockers | EXPERIMENT_REGISTRY.md |
| Experiment: E40 F3 live replay parity | PASS DEVELOPMENT | exact tracks and controls on 80 of 80 frames two sequences two detectors no quality metric | EXPERIMENT_REGISTRY.md |
| Experiment: E41 constant-free family | PLANNED NEXT | new training-free candidate handling removing fixed bins and fixed memory must be declared before run | NEXT_STEPS.md |
| Result: V4 test-dev pooled metrics | HELD-OUT | outputs/heldout_v4/pooled_metrics.json | RESULTS_CANONICAL.md |
| Result: V5-TF E36 family choice | DEVELOPMENT | F3 selected 17 catastrophic cells worst-detector relative gain minus 0.339 percent; F5 18 and minus 2.646 percent | RESULTS_CANONICAL.md |
| Result: V5-TF E39 constant audit | DEVELOPMENT | Otsu bins and window sensitive category E; history and warm-up insensitive A | RESULTS_CANONICAL.md |
| Failure: V1 candidate explosion | FAILED | equal percentage not equal count | FAILED_EXPERIMENTS.md |
| Failure: V2 density budgets | FAILED | double budget additive offsets over-suppress | FAILED_EXPERIMENTS.md |
| Failure: V2f closed-loop trust windup | FAILED | reliability conflates hard scene with false candidates | FAILED_EXPERIMENTS.md |
| Failure: J2 max-min objective degenerate | FAILED | unnormalised max-min tracks weakest detector | FAILED_EXPERIMENTS.md |
| Failure: V1 temperature non-invariance | FAILED | histogram normaliser breaks under score recalibration | FAILED_EXPERIMENTS.md |
| Failure: legacy SCI below random | FAILED | handcrafted SCI cues do not generalise | FAILED_EXPERIMENTS.md |
| Failure: MOTA-aligned learning cost | FAILED | objective mismatch with HOTA IDF1 adoption criterion | FAILED_EXPERIMENTS.md |
| Failure: learned controller overfitting by scarcity | FAILED | stumps beat global value in training folds lose held-out | FAILED_EXPERIMENTS.md |
| Failure: unstable gate tau cue | FAILED | four different cues across folds tau not adapted | FAILED_EXPERIMENTS.md |
| Failure: V4 RT-DETR below shared static | FAILED HYPOTHESIS | H2 rejected recall-limited category E constants | FAILED_EXPERIMENTS.md |
| Failure: F5 size-state resolution below random | FAILED HYPOTHESIS | E36 F5 worse than fixed F3 and random F5R despite eligible compute | FAILED_EXPERIMENTS.md |
| Decision: D-005 remove legacy SCI | ACTIVE | legacy SCI removed from V4 decision path | DECISIONS.md |
| Decision: D-009 keep scene adaptation V4 is ablation | ACTIVE | why V4 is not the final architecture: V4 compute-only is an ablation, deleting scene adaptation rejected as final direction, V4 constants are VisDrone-tuned category E | DECISIONS.md |
| Decision: D-010 cue status | ACTIVE | candidate and rejected cues from S2 | DECISIONS.md |
| Decision: D-011 learned V5 not adopted | ACTIVE | move development to train | DECISIONS.md |
| Decision: D-012 40/16 train split | PARTLY SUPERSEDED | final fit clause superseded by D-015 | DECISIONS.md |
| Decision: D-014 fidelity gate constants | ACTIVE | T4 gate thresholds never relaxed | DECISIONS.md |
| Decision: D-015 training-free final AC | ACTIVE | final AC layer must not be trained Optuna discovery-only | DECISIONS.md |
| Decision: D-016 F4 dropped | ACTIVE | F4 identical to F1 | DECISIONS.md |
| Decision: D-018 final target V5-TF | ACTIVE | final target is V5-TF; V4 baseline ablation only never fallback; confirmation reported not a gate | DECISIONS.md |
| Decision: D-019 scene-state control R-res | ACTIVE | frozen V5-TF must contain scene-state control; selectable F3 F5 | DECISIONS.md |
| Decision: D-020 constant audit rule | ACTIVE | sensitivity audit insensitive to structural else reported category E never tuned | DECISIONS.md |
| Decision: D-022 E36 selects F3 | ACTIVE | F3 selected honestly; F5 scene-adaptive resolution did not beat random control | DECISIONS.md |
| Decision: D-023 E39 C5 blockers | ACTIVE | Otsu bins and window sensitive E block policy lock and freeze | DECISIONS.md |
| Decision: D-024 live replay parity | ACTIVE | F3 live motion path exactly equals cached replay on declared development subset | DECISIONS.md |
| Decision: D-017 repository is project memory | ACTIVE | context files over chat history | DECISIONS.md |
| Constraint: C0 final target V5-TF | HARD | FINAL TARGET = V5-TF; V4 historical baseline ablation only; never revert to V4 | HARD_CONSTRAINTS.md |
| Constraint: C1 training-free final AC layer | HARD | AC training NOT allowed no fitted controller no labels no GT at deployment | HARD_CONSTRAINTS.md |
| Constraint: C3 causality | HARD | frame t uses only frames up to t no future no GT | HARD_CONSTRAINTS.md |
| Constraint: C5 no category E constants | HARD | E39 resolved history and warm-up as A; sensitive Otsu window and bins remain freeze blockers | HARD_CONSTRAINTS.md |
| Constraint: C6 AC before detector | HARD | not merely a post-detector score filter V4 is ablation | HARD_CONSTRAINTS.md |
| Constraint: C7 Optuna research-only | HARD | S1 S2 S3 C1 C2 C3 Optuna discovery tools only | HARD_CONSTRAINTS.md |
| Constraint: C8 protected evaluations | HARD | protected datasets and evaluations currently: confirmation-16, UAVDT test, Faster R-CNN, BoT-SORT; test-dev used once; no quality metrics before freeze | HARD_CONSTRAINTS.md |
| Constraint: C9 T4 official timing | HARD | Mac MPS development only fidelity gate constants | HARD_CONSTRAINTS.md |
| Constraint: C11 honest reporting | HARD | report if training-free does not beat V4 never switch to trained controller | HARD_CONSTRAINTS.md |
| Protocol: Amendment 4 | HISTORICAL | V4 whole-controller generalization audit | VALIDATION_PROTOCOL.md |
| Protocol: Amendment 5d | PARTLY SUPERSEDED | 40/16 train split protocol | VALIDATION_PROTOCOL.md |
| Protocol: Amendment 5f | ACTIVE | cross-hardware fidelity gate | VALIDATION_PROTOCOL.md |
| Protocol: Amendment 7 | CURRENT | V5-TF final target, scene-adaptive resolution R-res, families F3 F5 F5R, constant audit | VALIDATION_PROTOCOL.md |
| Protocol: Amendment 6 | CURRENT | V5-TF training-free protocol families comparison freeze | VALIDATION_PROTOCOL.md |
| Tag: v1.0.0-acmot-frozen | FROZEN | legacy AC-MOT tag | FROZEN_VERSIONS.md |
| Tag: universal-acmot-v1-freeze | FROZEN | V1 tag | FROZEN_VERSIONS.md |
| Tag: universal-acmot-v3-freeze | FROZEN | V3 tag | FROZEN_VERSIONS.md |
| Tag: universal-acmot-v4-freeze | FROZEN | V4 tag latest frozen | FROZEN_VERSIONS.md |
| Tag: universal-acmot-v5tf-freeze | PLANNED | created only after rule design fixed and T4 gate PASS | FROZEN_VERSIONS.md |
| Commit: a6c1fa4 | HISTORY | legacy AC-MOT freeze commit | FROZEN_VERSIONS.md |
| Commit: e56c2f3 | HISTORY | V1 freeze commit | FROZEN_VERSIONS.md |
| Commit: c1e799d | HISTORY | V3 freeze commit | FROZEN_VERSIONS.md |
| Commit: fc003bf | HISTORY | V4 freeze commit | FROZEN_VERSIONS.md |
| Commit: 3684684 | HISTORY | Amendment 6 training-free requirement | DECISIONS.md |
| Commit: 71faf44 | HISTORY | V5-TF implementation F1 F2 F3 | DECISIONS.md |
| Lock: TESTDEV_LOCK_V4 | USED ONCE | V4 test-dev lock evaluated once E31 | ../TESTDEV_LOCK_V4.json |
| Lock: TRANSFER_LOCK_FASTERRCNN | LOCKED NOT EVALUATED | Faster R-CNN transfer lock for V4 | ../TRANSFER_LOCK_FASTERRCNN.json |
| Lock: TRANSFER_LOCK_UAVDT | LOCKED NOT EVALUATED | UAVDT transfer lock for V4 | ../TRANSFER_LOCK_UAVDT.json |
| Lock: TRAIN_SPLIT_V5 | ACTIVE | fixed 40/16 split seed 20260927 | ../TRAIN_SPLIT_V5.json |
| Families: detector and tracker families tested | EVIDENCE | detector families tested YOLOv8n RT-DETR-L; Faster R-CNN cached not yet evaluated; tracker families tested ByteTrack BoT-SORT | RESULTS_CANONICAL.md |
| Failure: Failures not to repeat | REGISTRY | failures that should not be repeated: legacy SCI, learned controllers, density budgets, temporal persistence, MOTA-aligned cost, closed-loop trust, max-min objective | FAILED_EXPERIMENTS.md |

## Relations
| Subject | Relation | Object |
|---|---|---|
| Project: Universal AC-MOT | HAS_VERSION | Version: Legacy AC-MOT |
| Project: Universal AC-MOT | HAS_VERSION | Version: V1 |
| Project: Universal AC-MOT | HAS_VERSION | Version: V2 variants |
| Project: Universal AC-MOT | HAS_VERSION | Version: V3 |
| Project: Universal AC-MOT | HAS_VERSION | Version: V4 |
| Project: Universal AC-MOT | HAS_VERSION | Version: V5 learned controller |
| Project: Universal AC-MOT | HAS_VERSION | Version: V5-TF |
| Project: Universal AC-MOT | CURRENT_VERSION | Version: V5-TF |
| Project: Universal AC-MOT | FINAL_TARGET | Version: V5-TF |
| Project: Universal AC-MOT | BASELINE_ONLY | Version: V4 |
| Component: Universal AC Controller | HAS_CANDIDATE_RULE | Rule: F5 scene-adaptive resolution R-res |
| Rule: F5R random resolution control | CONTROLS_FOR | Rule: F5 scene-adaptive resolution R-res |
| Rule: F5 scene-adaptive resolution R-res | BELONGS_TO | Component: Compute Latency Constraint |
| Decision: D-018 final target V5-TF | SELECTS | Version: V5-TF |
| Decision: D-018 final target V5-TF | DEMOTES_TO_ABLATION | Version: V4 |
| Decision: D-018 final target V5-TF | RECORDED_IN | Protocol: Amendment 7 |
| Decision: D-019 scene-state control R-res | SELECTS | Rule: F5 scene-adaptive resolution R-res |
| Decision: D-019 scene-state control R-res | RECORDED_IN | Protocol: Amendment 7 |
| Decision: D-020 constant audit rule | GOVERNS | Experiment: E39 constant audit |
| Decision: D-022 E36 selects F3 | SELECTS | Rule: F3 motion-aware association |
| Decision: D-022 E36 selects F3 | REJECTS | Rule: F5 scene-adaptive resolution R-res |
| Decision: D-022 E36 selects F3 | SUPPORTED_BY | Experiment: E36 V5-TF dev validation |
| Decision: D-023 E39 C5 blockers | SUPPORTED_BY | Experiment: E39 constant audit |
| Decision: D-023 E39 C5 blockers | REQUIRES | Experiment: E41 constant-free family |
| Decision: D-024 live replay parity | SUPPORTED_BY | Experiment: E40 F3 live replay parity |
| Constraint: C0 final target V5-TF | APPLIES_TO | Version: V5-TF |
| Constraint: C0 final target V5-TF | APPLIES_TO | Version: V4 |
| Constraint: C0 final target V5-TF | DERIVED_FROM | Decision: D-018 final target V5-TF |
| Protocol: Amendment 7 | GOVERNS | Version: V5-TF |
| Protocol: Amendment 7 | GOVERNS | Experiment: E39 constant audit |
| Experiment: E39 constant audit | EVALUATES | Version: V5-TF |
| Project: Universal AC-MOT | LATEST_FROZEN_VERSION | Version: V4 |
| Project: Universal AC-MOT | NEXT_EXPERIMENT | Experiment: E41 constant-free family |
| Version: V1 | SUPERSEDES | Version: Legacy AC-MOT |
| Version: V3 | SUPERSEDES | Version: V1 |
| Version: V4 | SUPERSEDES | Version: V3 |
| Version: V5-TF | SUPERSEDES | Version: V5 learned controller |
| Version: V5-TF | COMPARED_AGAINST | Version: V4 |
| Version: V5-TF | COMPARED_AGAINST | Version: V3 |
| Version: V5-TF | COMPARED_AGAINST | Version: V5 learned controller |
| Version: Legacy AC-MOT | HAS_ARCHITECTURE | Architecture: Original SCI architecture |
| Version: V4 | HAS_ARCHITECTURE | Architecture: V4 compute-budget architecture |
| Version: V5-TF | HAS_ARCHITECTURE | Architecture: V5-TF target architecture |
| Version: Legacy AC-MOT | USES_COMPONENT | Component: Legacy SCI |
| Version: V3 | USES_COMPONENT | Component: Legacy SCI |
| Version: V3 | USES_COMPONENT | Component: Z-logit Leader Gate |
| Version: V4 | USES_COMPONENT | Component: Z-logit Leader Gate |
| Version: V4 | USES_COMPONENT | Component: ECDF Normaliser |
| Version: V4 | USES_COMPONENT | Component: Compute Latency Constraint |
| Version: V5-TF | USES_COMPONENT | Component: Scene State Analyzer |
| Version: V5-TF | USES_COMPONENT | Component: Online Self-Calibration |
| Version: V5-TF | USES_COMPONENT | Component: Universal AC Controller |
| Version: V5-TF | USES_COMPONENT | Component: Compute Latency Constraint |
| Version: V5-TF | USES_COMPONENT | Component: Detector Adapter |
| Version: V5-TF | USES_COMPONENT | Component: ECDF Normaliser |
| Version: V5-TF | USES_COMPONENT | Component: Otsu-3 Candidate Bands |
| Version: V5-TF | USES_COMPONENT | Component: Tracker Adapter |
| Component: Universal AC Controller | HAS_CANDIDATE_RULE | Rule: F1 Otsu window bands |
| Component: Universal AC Controller | HAS_CANDIDATE_RULE | Rule: F2 Otsu frame bands |
| Component: Universal AC Controller | HAS_CANDIDATE_RULE | Rule: F3 motion-aware association |
| Rule: F3 motion-aware association | USES_CUE | Cue: img_motion |
| Decision: D-016 F4 dropped | REJECTS | Rule: F4 online z-gate |
| Decision: D-005 remove legacy SCI | REJECTS | Component: Legacy SCI |
| Decision: D-005 remove legacy SCI | REJECTS | Cue: crowd count n/30 |
| Decision: D-005 remove legacy SCI | REJECTS | Cue: tiny object area |
| Decision: D-005 remove legacy SCI | REJECTS | Cue: img_edges |
| Decision: D-005 remove legacy SCI | REJECTS | Cue: img_brightness |
| Decision: D-005 remove legacy SCI | REJECTS | Cue: img_blur |
| Decision: D-005 remove legacy SCI | SUPPORTED_BY | Experiment: E24-E26 SCI audit |
| Decision: D-010 cue status | REJECTS | Cue: img_edges |
| Decision: D-010 cue status | REJECTS | Cue: img_brightness |
| Decision: D-010 cue status | REJECTS | Cue: img_blur |
| Decision: D-010 cue status | KEEPS_CANDIDATE | Cue: det_gap |
| Decision: D-010 cue status | KEEPS_CANDIDATE | Cue: det_count |
| Decision: D-010 cue status | KEEPS_CANDIDATE | Cue: trk_survival |
| Decision: D-010 cue status | KEEPS_CANDIDATE | Cue: trk_match |
| Decision: D-010 cue status | KEEPS_CANDIDATE | Cue: img_motion |
| Decision: D-010 cue status | KEEPS_CANDIDATE | Cue: img_motion_resp |
| Decision: D-010 cue status | SUPPORTED_BY | Experiment: E33 S2 cue utility |
| Decision: D-009 keep scene adaptation V4 is ablation | SELECTS | Version: V5-TF |
| Decision: D-009 keep scene adaptation V4 is ablation | DEMOTES_TO_ABLATION | Version: V4 |
| Decision: D-009 keep scene adaptation V4 is ablation | EXPLAINS | Architecture: V4 compute-budget architecture |
| Decision: D-011 learned V5 not adopted | SUPPORTED_BY | Experiment: E34-E35 S3 learned controller |
| Decision: D-015 training-free final AC | SUPERSEDES | Decision: D-012 40/16 train split |
| Decision: D-015 training-free final AC | SELECTS | Component: Online Self-Calibration |
| Decision: D-015 training-free final AC | REJECTS | Version: V5 learned controller |
| Decision: D-015 training-free final AC | SUPPORTED_BY | Experiment: E34-E35 S3 learned controller |
| Decision: D-015 training-free final AC | RECORDED_IN | Protocol: Amendment 6 |
| Decision: D-015 training-free final AC | RECORDED_IN | Commit: 3684684 |
| Decision: D-014 fidelity gate constants | RECORDED_IN | Protocol: Amendment 5f |
| Decision: D-016 F4 dropped | RECORDED_IN | Commit: 71faf44 |
| Constraint: C1 training-free final AC layer | APPLIES_TO | Version: V5-TF |
| Constraint: C3 causality | APPLIES_TO | Version: V5-TF |
| Constraint: C5 no category E constants | APPLIES_TO | Version: V5-TF |
| Constraint: C6 AC before detector | APPLIES_TO | Version: V5-TF |
| Constraint: C7 Optuna research-only | APPLIES_TO | Version: V5 learned controller |
| Constraint: C8 protected evaluations | APPLIES_TO | Split: VisDrone2019-MOT-train confirmation-16 |
| Constraint: C8 protected evaluations | APPLIES_TO | Detector: Faster R-CNN ResNet50-FPN v2 |
| Constraint: C8 protected evaluations | APPLIES_TO | Tracker: BoT-SORT |
| Constraint: C8 protected evaluations | APPLIES_TO | Split: UAVDT test |
| Constraint: C9 T4 official timing | APPLIES_TO | Experiment: E38 T4 fidelity gate |
| Constraint: C11 honest reporting | APPLIES_TO | Version: V5-TF |
| Constraint: C1 training-free final AC layer | DERIVED_FROM | Decision: D-015 training-free final AC |
| Constraint: C5 no category E constants | FLAGS | Parameter: OTSU_BINS 64 |
| Constraint: C5 no category E constants | FLAGS | Parameter: Otsu window 10 frames |
| Constraint: C5 no category E constants | EXCLUDES_FROM_V5_TF | Parameter: Z_REF 0.75 |
| Constraint: C5 no category E constants | EXCLUDES | Parameter: V4 tau 0.75 |
| Constraint: C5 no category E constants | EXCLUDES | Parameter: V4 sensitivity 0.4 |
| Constraint: C5 no category E constants | EXCLUDES | Parameter: V4 association offset 0.10 |
| Constraint: C5 no category E constants | EXCLUDES | Parameter: V4 NMS 0.45 |
| Constraint: C5 no category E constants | EXCLUDES | Parameter: V4 tracker 45/0.86 |
| Parameter: RobustHistory window 100 | BELONGS_TO | Component: Online Self-Calibration |
| Parameter: RobustHistory warm-up 5 | BELONGS_TO | Component: Online Self-Calibration |
| Parameter: OTSU_BINS 64 | BELONGS_TO | Component: Otsu-3 Candidate Bands |
| Parameter: Otsu window 10 frames | BELONGS_TO | Component: Otsu-3 Candidate Bands |
| Parameter: Z_REF 0.75 | BELONGS_TO | Component: Scene State Analyzer |
| Parameter: F3 cap 0.95 | BELONGS_TO | Rule: F3 motion-aware association |
| Parameter: native tracker defaults 30/0.8 | BELONGS_TO | Component: Tracker Adapter |
| Parameter: V4 tau 0.75 | BELONGS_TO | Component: Z-logit Leader Gate |
| Parameter: V4 NMS 0.45 | BELONGS_TO | Version: V4 |
| Parameter: V4 sensitivity 0.4 | BELONGS_TO | Version: V4 |
| Parameter: V4 association offset 0.10 | BELONGS_TO | Version: V4 |
| Parameter: V4 tracker 45/0.86 | BELONGS_TO | Version: V4 |
| Parameter: fidelity gate thresholds | BELONGS_TO | Protocol: Amendment 5f |
| Dataset: VisDrone2019-MOT | HAS_SPLIT | Split: VisDrone2019-MOT-val |
| Dataset: VisDrone2019-MOT | HAS_SPLIT | Split: VisDrone2019-MOT-train development-40 |
| Dataset: VisDrone2019-MOT | HAS_SPLIT | Split: VisDrone2019-MOT-train confirmation-16 |
| Dataset: VisDrone2019-MOT | HAS_SPLIT | Split: VisDrone2019-MOT-test-dev |
| Dataset: UAVDT | HAS_SPLIT | Split: UAVDT test |
| Experiment: E31 V4 test-dev held-out | EVALUATES | Version: V4 |
| Experiment: E31 V4 test-dev held-out | USES | Split: VisDrone2019-MOT-test-dev |
| Experiment: E31 V4 test-dev held-out | USES | Detector: YOLOv8n |
| Experiment: E31 V4 test-dev held-out | USES | Detector: RT-DETR-L |
| Experiment: E31 V4 test-dev held-out | USES | Tracker: ByteTrack |
| Experiment: E31 V4 test-dev held-out | USES | Tracker: BoT-SORT |
| Experiment: E31 V4 test-dev held-out | PRODUCES | Result: V4 test-dev pooled metrics |
| Result: V4 test-dev pooled metrics | REPORTS | Metric: HOTA |
| Result: V4 test-dev pooled metrics | REPORTS | Metric: IDF1 |
| Result: V4 test-dev pooled metrics | REPORTS | Metric: MOTA |
| Result: V4 test-dev pooled metrics | REPORTS | Metric: IDS |
| Experiment: E24-E26 SCI audit | EVALUATES | Component: Legacy SCI |
| Experiment: E24-E26 SCI audit | USES | Split: VisDrone2019-MOT-val |
| Experiment: E32 S1 headroom | USES | Split: VisDrone2019-MOT-val |
| Experiment: E33 S2 cue utility | USES | Split: VisDrone2019-MOT-val |
| Experiment: E34-E35 S3 learned controller | EVALUATES | Version: V5 learned controller |
| Experiment: E34-E35 S3 learned controller | USES | Split: VisDrone2019-MOT-val |
| Experiment: E36 V5-TF dev validation | EVALUATES | Version: V5-TF |
| Experiment: E36 V5-TF dev validation | USES | Split: VisDrone2019-MOT-train development-40 |
| Experiment: E36 V5-TF dev validation | USES | Detector: YOLOv8n |
| Experiment: E36 V5-TF dev validation | USES | Detector: RT-DETR-L |
| Experiment: E36 V5-TF dev validation | USES | Tracker: ByteTrack |
| Experiment: E36 V5-TF dev validation | PRODUCES | Result: V5-TF E36 family choice |
| Experiment: E39 constant audit | PRODUCES | Result: V5-TF E39 constant audit |
| Experiment: E39 constant audit | USES | Split: VisDrone2019-MOT-train development-40 |
| Experiment: E40 F3 live replay parity | EVALUATES | Rule: F3 motion-aware association |
| Experiment: E40 F3 live replay parity | USES | Split: VisDrone2019-MOT-train development-40 |
| Failure: F5 size-state resolution below random | DERIVED_FROM | Experiment: E36 V5-TF dev validation |
| Experiment: E37 discovery S1-S3 on train | USES | Split: VisDrone2019-MOT-train development-40 |
| Experiment: E38 T4 fidelity gate | USES | Split: VisDrone2019-MOT-train development-40 |
| Failure: V1 candidate explosion | DERIVED_FROM | Version: V1 |
| Failure: V2 density budgets | DERIVED_FROM | Version: V2 variants |
| Failure: V2f closed-loop trust windup | DERIVED_FROM | Version: V2 variants |
| Failure: J2 max-min objective degenerate | DERIVED_FROM | Version: V1 |
| Failure: V1 temperature non-invariance | DERIVED_FROM | Version: V1 |
| Failure: legacy SCI below random | DERIVED_FROM | Experiment: E24-E26 SCI audit |
| Failure: MOTA-aligned learning cost | DERIVED_FROM | Experiment: E34-E35 S3 learned controller |
| Failure: learned controller overfitting by scarcity | DERIVED_FROM | Experiment: E34-E35 S3 learned controller |
| Failure: unstable gate tau cue | DERIVED_FROM | Experiment: E34-E35 S3 learned controller |
| Failure: V4 RT-DETR below shared static | DERIVED_FROM | Experiment: E31 V4 test-dev held-out |
| Experiment: E17 signal audit | REJECTS | Cue: temporal persistence |
| Experiment: E17 signal audit | USES | Split: VisDrone2019-MOT-val |
| Tag: v1.0.0-acmot-frozen | POINTS_TO | Commit: a6c1fa4 |
| Tag: universal-acmot-v1-freeze | POINTS_TO | Commit: e56c2f3 |
| Tag: universal-acmot-v3-freeze | POINTS_TO | Commit: c1e799d |
| Tag: universal-acmot-v4-freeze | POINTS_TO | Commit: fc003bf |
| Commit: a6c1fa4 | FREEZES | Version: Legacy AC-MOT |
| Commit: e56c2f3 | FREEZES | Version: V1 |
| Commit: c1e799d | FREEZES | Version: V3 |
| Commit: fc003bf | FREEZES | Version: V4 |
| Commit: 71faf44 | IMPLEMENTS | Version: V5-TF |
| Tag: universal-acmot-v5tf-freeze | PLANNED_FOR | Version: V5-TF |
| Protocol: Amendment 6 | GOVERNS | Version: V5-TF |
| Protocol: Amendment 4 | GOVERNS | Version: V4 |
| Protocol: Amendment 5f | GOVERNS | Experiment: E38 T4 fidelity gate |
| Protocol: Amendment 5d | GOVERNS | Split: VisDrone2019-MOT-train confirmation-16 |
| Version: V4 | TESTED_WITH | Detector: YOLOv8n |
| Version: V4 | TESTED_WITH | Detector: RT-DETR-L |
| Version: V4 | TESTED_WITH | Tracker: ByteTrack |
| Version: V4 | TESTED_WITH | Tracker: BoT-SORT |
| Lock: TESTDEV_LOCK_V4 | LOCKS | Split: VisDrone2019-MOT-test-dev |
| Lock: TRANSFER_LOCK_FASTERRCNN | LOCKS | Detector: Faster R-CNN ResNet50-FPN v2 |
| Lock: TRANSFER_LOCK_UAVDT | LOCKS | Split: UAVDT test |
| Lock: TRAIN_SPLIT_V5 | DEFINES_SPLIT | Split: VisDrone2019-MOT-train development-40 |
| Lock: TRAIN_SPLIT_V5 | DEFINES_SPLIT | Split: VisDrone2019-MOT-train confirmation-16 |
| Families: detector and tracker families tested | TESTED_DETECTOR | Detector: YOLOv8n |
| Families: detector and tracker families tested | TESTED_DETECTOR | Detector: RT-DETR-L |
| Families: detector and tracker families tested | CACHED_NOT_EVALUATED | Detector: Faster R-CNN ResNet50-FPN v2 |
| Families: detector and tracker families tested | TESTED_TRACKER | Tracker: ByteTrack |
| Families: detector and tracker families tested | TESTED_TRACKER | Tracker: BoT-SORT |
| Failure: Failures not to repeat | INCLUDES | Failure: V1 candidate explosion |
| Failure: Failures not to repeat | INCLUDES | Failure: V2 density budgets |
| Failure: Failures not to repeat | INCLUDES | Failure: V2f closed-loop trust windup |
| Failure: Failures not to repeat | INCLUDES | Failure: J2 max-min objective degenerate |
| Failure: Failures not to repeat | INCLUDES | Failure: V1 temperature non-invariance |
| Failure: Failures not to repeat | INCLUDES | Failure: legacy SCI below random |
| Failure: Failures not to repeat | INCLUDES | Failure: MOTA-aligned learning cost |
| Failure: Failures not to repeat | INCLUDES | Failure: learned controller overfitting by scarcity |
| Failure: Failures not to repeat | INCLUDES | Failure: unstable gate tau cue |
| Failure: Failures not to repeat | INCLUDES | Failure: V4 RT-DETR below shared static |
| Failure: Failures not to repeat | INCLUDES | Cue: temporal persistence |
| Component: Legacy SCI | HAD_CUE | Cue: crowd count n/30 |
| Component: Legacy SCI | HAD_CUE | Cue: tiny object area |
| Component: Legacy SCI | HAD_CUE | Cue: img_edges |
| Component: Legacy SCI | HAD_CUE | Cue: img_brightness |
| Component: Legacy SCI | HAD_CUE | Cue: img_blur |

## Code links
| Entity | Relation | Code file |
|---|---|---|
| Component: Online Self-Calibration | IMPLEMENTED_IN | online_calibration.py |
| Component: Otsu-3 Candidate Bands | IMPLEMENTED_IN | online_calibration.py |
| Component: Otsu-3 Candidate Bands | IMPLEMENTED_IN | universal_policy_pipeline.py |
| Component: Universal AC Controller | IMPLEMENTED_IN | universal_policy_pipeline.py |
| Component: Z-logit Leader Gate | IMPLEMENTED_IN | universal_policy_pipeline.py |
| Component: ECDF Normaliser | IMPLEMENTED_IN | adapters/detectors/online_normalizer.py |
| Component: Scene State Analyzer | IMPLEMENTED_IN | scene_state.py |
| Component: Scene State Analyzer | IMPLEMENTED_IN | tools/visual_cues.py |
| Component: Compute Latency Constraint | IMPLEMENTED_IN | universal_acmot.py |
| Component: Detector Adapter | IMPLEMENTED_IN | adapters/detectors/base.py |
| Component: Tracker Adapter | IMPLEMENTED_IN | adapters/trackers/base.py |
| Component: Legacy SCI | IMPLEMENTED_IN | core.py |
| Version: V5 learned controller | IMPLEMENTED_IN | v5_controller.py |
| Version: V5 learned controller | IMPLEMENTED_IN | tools/v5_train.py |
| Experiment: E32 S1 headroom | IMPLEMENTED_IN | tools/v5_s1_headroom.py |
| Experiment: E33 S2 cue utility | IMPLEMENTED_IN | tools/v5_s2_cues.py |
| Experiment: E34-E35 S3 learned controller | IMPLEMENTED_IN | tools/v5_s3_nested.py |
| Experiment: E36 V5-TF dev validation | IMPLEMENTED_IN | tools/v5tf_dev.py |
| Experiment: E39 constant audit | IMPLEMENTED_IN | tools/v5tf_dev.py |
| Rule: F5 scene-adaptive resolution R-res | IMPLEMENTED_IN | universal_policy_pipeline.py |
| Rule: F5R random resolution control | IMPLEMENTED_IN | universal_policy_pipeline.py |
| Experiment: E38 T4 fidelity gate | IMPLEMENTED_IN | tools/fidelity_gate.py |
| Experiment: E31 V4 test-dev held-out | IMPLEMENTED_IN | tools/heldout_run.py |
| Constraint: C9 T4 official timing | IMPLEMENTED_IN | tools/t4_benchmark.py |
| Detector: YOLOv8n | IMPLEMENTED_IN | adapters/detectors/yolov8.py |
| Detector: RT-DETR-L | IMPLEMENTED_IN | adapters/detectors/rtdetr.py |
| Detector: Faster R-CNN ResNet50-FPN v2 | IMPLEMENTED_IN | adapters/detectors/fasterrcnn.py |
| Tracker: ByteTrack | IMPLEMENTED_IN | adapters/trackers/bytetrack.py |
| Tracker: BoT-SORT | IMPLEMENTED_IN | adapters/trackers/botsort.py |
