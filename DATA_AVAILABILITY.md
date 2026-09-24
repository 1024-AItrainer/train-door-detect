# Data Availability and Reproducibility Scope

## What this repository provides

This repository releases the **lightweight detector only** used in the paper:

> *A lightweight real-time visual localization framework for dynamic train door perception in high-speed railway systems*

Included for independent method-level checking:

- Paper modules: `ultralytics/train_door_detect/modules/` (SPPSA-StarNet, CFDPN, DESD)
- Model config: `ultralytics/train_door_detect/cfg/models/train_door_detect_model.yaml`
- Scripts: `train.py`, `val.py`, `predict.py`
- Dependencies: `requirements.txt`
- Base framework: Ultralytics YOLO **8.3.225** (research fork)

## What cannot be released

Because this work was conducted under a **commercial collaborative project** with railway operational partners:

- Raw high-speed-railway station videos
- Industrial annotations / full proprietary detection dataset
- Complete on-site measurement / localization / PSD control software stack

cannot be publicly released.

Consequently, **industrial train-door metrics and Section-5 localization numbers cannot be regenerated from public data alone**.

## How to reproduce detector training publicly

Use any YOLO-format detection dataset (e.g., VisDrone) to verify that the released package trains and validates:

```bash
pip install -r requirements.txt
pip install -e .
python train.py --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 1 --batch 2
python val.py --weights runs/train_door_detect/weights/best.pt
```

See `README.md` for full instructions.
