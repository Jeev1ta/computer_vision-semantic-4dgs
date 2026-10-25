# Critical project review and defensible roadmap

## Short verdict

The repository contains meaningful engineering work around a 4D-GS baseline, 2D SAM/CLIP extraction, semantic training hooks, a static multi-view label-transfer pipeline, and an editing prototype. The checked-in evidence does **not** currently support the presentation's claims of a verified end-to-end dynamic semantic editor, 84.3% 3D mIoU, 96.2% temporal consistency, 112.5 FPS, or full ARAP manipulation. The reports previously contained fixed values; the Phase 4 notebook uses a mock model; the browser demo is illustrative rather than a trained 4D-GS renderer/editor. Treat the current state as a prototype, not a validated research result.

The PPT did not reveal explicit pending/in-progress rows: its milestone table labels every item “Done.” That status is inconsistent with the repository evidence. This document reclassifies the work honestly and provides a completion plan.

## What is real, what is not yet demonstrated

| Area | Evidence-based status | Claim to avoid for now |
|---|---|---|
| Dynamic RGB reconstruction | 4D-GS codebase and training/rendering path are present. | “Reproduced and verified end-to-end” without run logs, checkpoint, scene, and rendered outputs. |
| SAM 2 / CLIP preprocessing | Code for per-frame masks/features and prompt labels is present. | Stable object identity over time or independent 3D supervision. Per-frame mask classification can change labels. |
| Semantic 4D training | Gaussian semantic attributes, semantic raster passes, and loss hooks exist. | A measured 3D semantic mIoU; 2D label-map agreement alone is not a 3D metric. |
| Multi-view semantic voting | A separate static COLMAP/PLY pipeline exists. | Evidence that it validates the dynamic 4D semantic model; these are different paths and tasks. |
| Editing | Class selection and local anchor-weighted point transforms exist. The editor now rejects unknown prompts instead of silently selecting class 1. | Arbitrary natural-language editing, physically coherent 4D edits, or ARAP optimization. The current weighted transform is not an ARAP energy solve; covariance handling and time behavior need validation. |
| UI | A synthetic/illustrative browser demo and a static-pipeline dashboard are present. | A live editor operating on a trained dynamic checkpoint. |
| Quantitative results | Previous report values were hard-coded and are withdrawn. | Any prior headline score until regenerated from saved predictions and labels. |

## Novelty: a stronger and honest framing

**SAM 2 + CLIP + 4D-GS is a system integration, not by itself a defensible algorithmic novelty claim.** The components are known; simply attaching semantic channels to a differentiable renderer is likely incremental. Do not claim “first” without a current, systematic literature review and careful scope.

A technically stronger contribution to pursue is:

> **Temporally consistent, prompt-queryable semantic fields on canonical dynamic Gaussians, with confidence-aware distillation and edits that remain attached to an object across time.**

This is a research direction, not a result already proven by this code. Make it concrete by implementing and testing:

1. **Canonical semantic identity:** predict/store object or part embeddings on canonical Gaussians; use the deformation map only to move geometry. Define how semantic attributes behave when points are duplicated/pruned.
2. **Confidence-aware 2D supervision:** weight SAM/CLIP pseudo-label losses by mask stability, prompt margin, visibility, and view agreement. Ignore uncertain/occluded pixels rather than forcing a noisy hard label.
3. **Temporal/view consistency:** associate masks across frames/views using SAM 2 tracks or explicit 3D reprojection; add a consistency objective on matched canonical Gaussians. Report it separately from label accuracy.
4. **Query-to-region selection:** encode the user's text with the same CLIP model used for feature distillation; compare normalized embeddings with calibrated thresholds. Persist encoder/version, preprocessing, prompt/class mapping, and feature normalization metadata. Unknown queries must return “no match,” not a guessed class.
5. **Time-aware editing:** select a canonical semantic region, apply a parameterized transform to its canonical positions, then render/deform at several timestamps. Implement a genuine local rigidity objective if claiming ARAP, or accurately call the current method anchor-weighted local transform blending.

A realistic novelty statement until those experiments exist is: **“We are building a 4D-GS prototype that transfers SAM 2/CLIP semantic supervision to Gaussian attributes and explores semantic-region editing.”**

## Completion plan (dependency order)

### P0 — Credibility and reproducibility (must do first)
- Keep evaluation evidence-first. The evaluator now refuses missing data, pairs filenames, checks single-channel maps/shapes, and reports only measured label IoU.
- Establish one small, fixed scene and a held-out time/view split with manually checked labels. Save the exact prompt/class mapping and all output artifacts.
- Run baseline and semantic model on the same machine, resolution, cameras, and timestamps. Keep runtime warm-up/synchronization and measurement procedure explicit.
- Produce a clean one-command manifest of environment, dataset, checkpoint, command, logs, and outputs. Never use the training pseudo-labels as the only ground truth.

### P1 — Prove the semantic training path
- Train baseline and semantic model on one scene from a clean checkpoint.
- Verify semantic tensor dimensionality and class/CLIP mode against metadata; store class IDs and names unambiguously.
- Inspect novel-view and novel-time semantic renders, not only training views.
- Add ablations: baseline RGB only; SAM labels only; CLIP feature loss only; combined; combined plus temporal consistency.
- Report per-class IoU and macro averages on held-out manual labels, confidence intervals across scenes/seeds, and failure cases.

### P2 — Make editing real and time-consistent
- Test loading exact PLY + deformation + semantic sidecars; verify Gaussian counts and semantic dimension match.
- Select a region from supported class logits or genuine compatible CLIP text embeddings. Remove spatial heuristic fallback and reject unsupported prompts (now enforced in the editor API).
- Define how covariance and rotation are transformed. Test that edited regions remain coherent over sampled timestamps and that non-selected regions remain unchanged.
- Add round-trip tests for export/reload, reset, empty selections, and sidecar persistence.
- Implement ARAP energy minimization only if it is a core promised contribution; otherwise rename the method and update the PPT.

### P3 — Build defensible research evaluation
- **Semantic:** held-out manually annotated mask IoU, per-class IoU, class-balanced accuracy, and unknown-query behavior.
- **Temporal:** evaluate labels on corresponding 3D points / optical-flow correspondences; report flicker separately for static background and moving objects. Raw same-pixel adjacent-frame agreement is invalid under motion.
- **Appearance:** PSNR/SSIM/LPIPS on held-out RGB with metric settings documented.
- **Speed/memory:** report hardware, image size, warm-up, synchronized timing, semantic-channel count, peak VRAM, and preprocessing costs separately from render-only FPS.
- **Editing:** selection precision/recall and temporal attachment; measure geometry/appearance preservation and edit latency on an actual loaded scene.
- **Baselines:** compare under the same data/split/hardware. Separate related-work numbers copied from papers from measurements performed by this project.

## Presentation changes recommended

Replace “100% done,” “first,” “zero flicker,” “full ARAP,” and all unverified metrics with the status and evidence boundaries above. Separate three deliverables clearly: (1) dynamic 4D-GS semantic training, (2) static multi-view geometric label voting, and (3) illustrative WebGL demo. Present the actual trained-scene output and limitations. A modest, reproducible result is more credible than a broad unsupported claim.
