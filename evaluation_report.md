# Evaluation status: not yet verified

The previous version of this report listed PSNR, SSIM, LPIPS, FPS, semantic mIoU, per-class IoU, temporal consistency, and ARAP claims without a traceable evaluation run. Those numbers have been withdrawn pending reproduction; they must not be cited as measured project results.

The checked-in evaluator now computes pixel-space semantic label-map IoU only, from prediction and ground-truth PNGs matched by filename. It does not establish 3D accuracy, temporal consistency, visual quality, rendering speed, or edit quality.

## Evidence required before reporting results

- Exact scene, train/test split, timestamps, checkpoint, label mapping, and evaluation frame list.
- Held-out manual semantic labels (not the same SAM/CLIP predictions used for training).
- Baseline 4D-GS and semantic model measured with identical cameras, resolution, hardware, and timing procedure.
- Appearance metrics from rendered RGB against held-out RGB; state image range and metric implementation.
- Temporal metrics evaluated on corresponding points/pixels using motion/flow or known synthetic correspondences; raw adjacent-frame pixel agreement is not sufficient.
- Editing evaluation at multiple timestamps, with selection precision/recall and geometry/appearance consistency before and after edits.
- Saved commands, environment versions, logs, predictions, ground truth, and machine-readable metric outputs.
