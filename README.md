# Ground–Satellite Smoke Detection

Lightweight YOLO detectors for wildfire smoke, a zero-shot cross-domain test, and simulated ground–satellite late fusion.

**"Ground–Satellite Smoke Detection: Lightweight YOLO Detectors, Cross-Domain Evaluation, and Calibrated Late Fusion"** (Akaash M K, Deepsikha K V, Dr. Arivoli A, Vellore Institute of Technology)

> **Important:** All fusion results are *simulated* from independently sampled ground and satellite images. They are not synchronized ground–satellite observations, so they show trade-offs, not operational benefit.

## What this repo does

1. **Branch A (terrestrial):** trains YOLO11n for fire and smoke on the Roboflow Fire-Smoke-Detection-Yolov11 v2 dataset.
2. **Branch B (satellite):** trains smoke-only detectors on FASDD_RS (remote-sensing subset of FASDD) in four configurations (E1–E4), 3 seeds each.
3. **Cross-domain test:** applies the terrestrial detector directly to satellite test images (zero-shot).
4. **Calibrated late fusion (simulation):** Platt-scales each detector's image-level confidence, then compares satellite-only, ground-only, AND, OR, Noisy-OR and a logistic-regression stacker under different event prevalence and sensor visibility.

## Repository structure

.
├── notebooks/
│   ├── 01_terrestrial_yolo11n_branch_a.ipynb
│   └── 02_satellite_fasdd_rs_crossdomain_fusion.ipynb
├── exploratory/
│   └── ts_satfire_temporal_unet_dropped.ipynb
├── requirements.txt
└── README.md

### Notebooks

| Notebook | Contents |
|---|---|
| `01_terrestrial_yolo11n_branch_a` | Roboflow dataset download, polygon-to-box conversion, YOLO11n training (640 px, seed 42), validation/test evaluation, checkpoint backup. The later cells audit TS-SatFire data and are not used in the paper. |
| `02_satellite_fasdd_rs_crossdomain_fusion` | FASDD_RS conversion to YOLO format, E1–E4 training, zero-shot evaluation of Branch A, Platt calibration, AUROC, fusion simulation, bootstrap CIs, latency benchmarking, qualitative figures, table export. |
| `exploratory/ts_satfire_temporal_unet_dropped` | Earlier attempt at a satellite branch using a temporal U-Net on TS-SatFire VIIRS time series. Dropped because of labelling problems; kept for transparency. |

Run 01 first (it produces the Branch A weights used by 02).

## Datasets (not included)

- **Terrestrial:** Fire-Smoke-Detection-Yolov11, version 2 (Roboflow Universe, S. Gamall). 9,156 / 872 / 435 train/val/test images.
- **Satellite:** FASDD_RS, RGB remote-sensing subset of the Flame and Smoke Detection Dataset (Wang et al., Geo-Spat. Inf. Sci., 2025). 1,112 / 741 / 370 images, smoke annotations only.
- **Exploratory only:** TS-SatFire (Zhao et al., Sci. Data, 2025).

Please follow each dataset's own license and citation requirements.

## Setup

The notebooks were run on Kaggle (NVIDIA T4). Dataset paths such as `/kaggle/input/...` must be changed if you run them elsewhere.

pip install ultralytics roboflow scikit-learn rasterio opencv-python pandas matplotlib

The Roboflow API key is read from a Kaggle secret named `ROBOFLOW_API_KEY`. Never commit your own key.

## Key results

**Terrestrial detector (Branch A, test set)**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---|---|---|---|
| All | 76.53 | 70.22 | 75.92 | 44.55 |
| Fire | 78.06 | 77.97 | 82.74 | 48.28 |
| Smoke | 74.99 | 62.46 | 69.10 | 40.82 |

**Satellite detector (Branch B, mean ± SD over 3 seeds, test set)**

| Config | mAP@50 | mAP@50–95 |
|---|---|---|
| E1: YOLO11n, 640 px, COCO init | 67.93 ± 1.30 | 39.58 ± 0.80 |
| E2: YOLO11n, 1024 px, COCO init | 70.74 ± 0.56 | 40.70 ± 0.85 |
| E3: YOLO11n, 640 px, Branch A init | 67.71 ± 1.51 | 39.70 ± 0.97 |
| E4: YOLOv8n, 640 px, COCO init | 67.66 ± 1.20 | 40.43 ± 0.90 |

**Zero-shot transfer (terrestrial detector on satellite test images)**
mAP@50 of 10.70% at 640 px and 6.74% at 1024 px, versus 67.93% / 70.74% for the satellite-trained detectors. This comparison also reflects dataset and annotation differences, not domain shift alone.

**Simulated fusion** (50% prevalence, 0.7/0.7 visibility, 20 draws)

| Strategy | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|
| Satellite-only | 93.0 | 70.7 | 80.3 | 5.4 |
| Ground-only | 87.7 | 65.5 | 75.0 | 9.3 |
| AND | 94.4 | 54.0 | 68.7 | 3.2 |
| OR | 91.0 | 86.4 | 88.7 | 8.6 |
| Noisy-OR | 90.0 | 88.2 | 89.1 | 9.9 |
| LR stacker | 87.1 | 87.1 | 87.1 | 13.0 |

**Efficiency (T4, batch size 1):** Branch A 640 px 9.5 ms, Branch B E1 640 px 10.5 ms, Branch B E2 1024 px 19.0 ms (about 2.6 M parameters each).

## Limitations

- Fusion is simulated from independent image samples (no temporal alignment, no correlated errors); Noisy-OR assumes conditional independence.
- With a missing sensor replaced by a negative image, the reduced-visibility AUROC gain (0.838 → 0.928) is partly built into the simulation.
- At 10% prevalence there is no consistent F1 advantage for fusion; thresholds were tuned at 50% prevalence and not re-tuned.
- The terrestrial test set has no background-only images, so smoke-free negatives are mainly fire-only images.
- Branch A was trained once (seed 42), so no variance is reported for it.

## Citation

@misc{akaash2026groundsatellite,
  title  = {Ground--Satellite Smoke Detection: Lightweight YOLO Detectors, Cross-Domain Evaluation, and Calibrated Late Fusion},
  author = {Akaash M K and Deepsikha K V and Arivoli A},
  year   = {2026},
  note   = {Vellore Institute of Technology}
}

## License

Code: MIT (see LICENSE). Datasets and pretrained weights keep their original licenses.

## Contact

Akaash M K — akaash.mk2023@vitstudent.ac.in
