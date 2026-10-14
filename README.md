# Reducing False Positives on Male Chests in Nudity Detection via Pose-Guided Gender Verification

Source code accompanying the manuscript submitted to the *Journal of Electronic Imaging*.

Existing nudity detectors frequently misclassify muscular or gynecomastia male chests as female exposure.
This repository implements a multi-stage verification pipeline that uses the perceived gender of the
person who owns a detected chest as a **secondary** check to release benign male-chest detections.

```
Input frame
  └─ 1. Chest detection        (fine-tuned YOLOv8n  or  NudeNet)
       └─ 2. Pose estimation   (YOLOv8n-pose, 17 COCO keypoints)
            └─ 3. Face detection + pose-guided face-chest matching (YOLOv8l-face)
                 └─ 4. Gender classification (DeepFace, VGG-Face gender model)
                      └─ Final decision (Eq. 4)
```

**Final decision rule.** A chest flagged as *Sensitive* by the primary detector is released as *Normal*
only when the matched face is classified as *Male* with confidence `S >= tau` (`tau = 0.90`).
In every other case (female label, low confidence, or no matched face) the primary detector's
decision is kept, so the secondary stage can never add a block.

## Repository structure

| File | Description | Paper |
|------|-------------|-------|
| `static_image_evaluation.py` | Full pipeline (`BodybuildingExperiment`) for static-image evaluation with a YOLOv8 or NudeNet primary detector; also contains data preparation utilities (Unsplash collection, NudeNet annotation filtering, baseline inference). | Sec. 3, Sec. 4.1-4.2 |
| `pose_guided_matching.py` | Pose-guided face-chest matching: IoA-based skeleton assignment and the shoulder-line constraint (`alpha = 0.6`, `beta_min = 0.2`, `beta_max = 1.2`). Run it directly for a synthetic sanity check. | Sec. 3.3, Eqs. (1)-(3) |
| `table1_multi_person_matching.py` | Multi-person face-chest matching experiment comparing detection confidence-based, center distance-based, and pose-guided assignment. | Sec. 3.3, Table 1, Fig. 2 |
| `fig3_confusion_matrix_baseline.py` | Confusion matrices of the YOLOv8 / NudeNet baselines. | Fig. 3 |
| `fig3_confusion_matrix_proposed.py` | Confusion matrices of the proposed method at `tau = 0.90`. | Fig. 3 |
| `fig3_confusion_matrix_yolo_combined.py` | Female / male / combined confusion matrices for YOLOv8 + proposed method. | Fig. 3 (supplementary) |
| `table3_threshold_sweep_yolo_nudenet.py` | Threshold sweep plots for YOLOv8 and NudeNet pipelines. | Table 3 |
| `table3_threshold_sweep_yolo.py` | Threshold sweep plots and metrics for the YOLOv8 pipeline. | Table 3 (supplementary) |
| `collect_wwe_videos.py` | Collection script for WWE broadcast videos used in the video evaluation. | Sec. 4.3.1 |

## Requirements

- Python 3.9
- NVIDIA GPU recommended (experiments used a GeForce GTX 1660 SUPER)

```bash
pip install -r requirements.txt
```

Main packages: `ultralytics`, `nudenet`, `deepface`, `opencv-python`, `torch`, `tensorflow`,
`numpy`, `pandas`, `matplotlib`, `seaborn`.

## Model weights

Weights are not included in this repository. Place them in the project root:

| File | Description |
|------|-------------|
| `best.pt` | YOLOv8n chest detector fine-tuned on the NudeNet Detector Auto-Generated Data (200 epochs, 640x640, AdamW, batch 16, lr 0.01, 8:2 split). Available from Noh SeongUn (chips36@naver.com) upon reasonable request. |
| `yolov8n-pose.pt` | Ultralytics YOLOv8n-pose (downloaded automatically by `ultralytics`). |
| `yolov8l-face.pt` | YOLOv8l-face detector. |

NudeNet and DeepFace download their own pretrained weights on first use.

## Usage

### Static-image evaluation (Sec. 4.2)

Edit `INPUT_DIR`, `OUTPUT_DIR`, and `detector_type` (`'yolo'` or `'nudenet'`) in the
`__main__` block of `static_image_evaluation.py`, then run:

```bash
python static_image_evaluation.py
```

Images are sorted into `man/`, `woman/`, and `unknown/` sub-folders of `OUTPUT_DIR`, and summary
statistics (baseline vs. final decisions, processing time) are printed.

`static_image_evaluation.py` reads the Unsplash API key from an environment variable at import time:

```bash
export UNSPLASH_ACCESS_KEY="<your-unsplash-access-key>"
```

### Multi-person matching experiment (Table 1)

```bash
python table1_multi_person_matching.py
```

Results for each assignment strategy are saved to `map1_first_array/`, `map2_min_distance/`,
and `map3_pose_proposed/` under the output folder.

### Figures

```bash
python fig3_confusion_matrix_baseline.py
python fig3_confusion_matrix_proposed.py
python table3_threshold_sweep_yolo_nudenet.py
```

## Results (independent test set, `tau = 0.90`)

| Model | Method | Acc. (%) | Prec. (%) | Rec. (%) | F1 | FP |
|-------|--------|---------:|----------:|---------:|---:|---:|
| YOLOv8 | Baseline | 74.92 | 67.04 | 99.67 | 0.8016 | 147 |
| YOLOv8 | Proposed | **95.25** | 93.31 | 97.67 | **0.9544** | **21** |
| NudeNet | Baseline | 74.91 | 67.39 | 98.94 | 0.8017 | 135 |
| NudeNet | Proposed | **91.64** | 93.38 | 90.07 | **0.9170** | **18** |

## Data availability

Datasets are not redistributed in this repository.

- **Detector training / female-exposure images:** NudeNet Detector Auto-Generated Data (public).
- **Male-chest images:** collected via the Unsplash developer API (`download_images()` in `static_image_evaluation.py`).
- **Generative-AI and multi-person images:** generated with Google Gemini.
- **Video clips:** Unsplash image-to-video clips (Gemini, Veo 3.1) and WWE broadcast segments.
  WWE footage cannot be redistributed due to licensing restrictions.

Evaluation metadata, data-split seeds, and annotation guidelines are available from
Noh SeongUn (chips36@naver.com) upon reasonable request.

## Ethical considerations

- The gender signal is a face-based **perceived binary estimate**, used only as a conservative
  secondary check to reduce over-blocking, not as a representation of gender identity.
- Face crops are processed in memory and are not written to disk.
- Face-based gender classifiers show accuracy disparities across demographic groups; the gender
  stage should be audited before any deployment.

## Citation

```bibtex
@article{noh_choi_male_chest_fp,
  title   = {Reducing False Positives on Male Chests in Nudity Detection via Pose-Guided Gender Verification},
  author  = {Noh, Seongun and Choi, Daeseon},
  journal = {Journal of Electronic Imaging},
  note    = {Under review}
}
```

## Contact

Noh SeongUn — chips36@naver.com
