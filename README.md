# Train-Door Detector (Paper Release)

Lightweight detector released for peer review of:

> *A lightweight real-time visual localization framework for dynamic train door perception in high-speed railway systems*

This repository provides the **detection network only** (SPPSA-StarNet + CFDPN + DESD), with installation and run scripts for reproducibility.

## Relationship to Ultralytics YOLO (official)

This package is a **research fork** of the official [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) codebase.

| Item                        | Value                                                                                        |
|-----------------------------|----------------------------------------------------------------------------------------------|
| Upstream project            | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)                        |
| Base version in this repo   | **`ultralytics==8.3.225`** (`ultralytics/__init__.py`)                                       |
| License of Ultralytics code | AGPL-3.0 (retain upstream license notices)                                                   |
| Our contribution            | Paper modules under `ultralytics/train_door_detect/` and wiring in `ultralytics/nn/tasks.py` |

We rely on Ultralytics for training / validation / inference APIs (`YOLO.train`, `YOLO.val`, `YOLO.predict`). Please cite Ultralytics YOLO when using this fork, in addition to
citing our paper.

## What is included / not included

See also [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) and [`REPRODUCE.md`](REPRODUCE.md) for the release boundary and a short reviewer checklist.


**Included (open for review)**

- Detector architecture YAML: `ultralytics/train_door_detect/cfg/models/train_door_detect_model.yaml`
- Paper modules:
    - **SPPSA-StarNet**: `modules/StarNet.py`, `modules/SPPSA.py`
    - **CFDPN**: `modules/CFDPN.py` (`SAG`, `ChannelAlign`, `LightFusion`)
    - **DESD** (NMS-free head): `modules/DESD.py`
- Scripts: `train.py`, `val.py`, `predict.py`
- Public-data demo config: VisDrone (`ultralytics/cfg/datasets/VisDrone.yaml`)

**Not included (commercial / operational constraints)**

- Industrial high-speed-railway train-door images and labels
- Full visual localization / speed-measurement / alignment pipeline
- Deployment packages used in the field prototype

For method-level reproducibility of the **detector**, use the VisDrone demo (or any YOLO-format detection dataset). Industrial numbers in the paper were obtained on the proprietary
train-door dataset described in the manuscript.

## Model (paper names ↔ code)

| Paper         | Code / YAML token                    | Path                                     |
|---------------|--------------------------------------|------------------------------------------|
| SPPSA-StarNet | `StarNet`, `SPPSA`                   | `ultralytics/train_door_detect/modules/` |
| CFDPN         | `SAG`, `ChannelAlign`, `LightFusion` | `.../modules/CFDPN.py`                   |
| DESD          | `DESD`                               | `.../modules/DESD.py`                    |

Scale `n` (default): about **1.2M parameters**, **4.3 GFLOPs** at 640×640 (Ultralytics `model.info()`; leaf-module layer count may differ from older Ultralytics prints).

## Requirements

- Python ≥ 3.8 (3.9+ recommended)
- CUDA-capable GPU recommended for training
- See `requirements.txt`

Install PyTorch for your CUDA version first (see [pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
cd ultralytics-for-train_door-detect
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs this fork as the local `ultralytics` package (version **8.3.225** with paper modules).

Optional check:

```bash
python -c "from ultralytics import YOLO; m=YOLO('ultralytics/train_door_detect/cfg/models/train_door_detect_model.yaml'); m.info()"
```

## Quick start (reproducible public demo)

### 1) Train (VisDrone)

VisDrone downloads automatically on first run (large; needs disk space and network).

```bash
python train.py --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 100 --batch 8 --imgsz 640
```

Useful flags: `--device 0`, `--workers 4`, `--project runs --name train_door_detect`.

Weights are saved under `runs/train_door_detect/weights/best.pt` (default naming).

### 2) Validate

```bash
python val.py --weights runs/train_door_detect/weights/best.pt --data ultralytics/cfg/datasets/VisDrone.yaml
```

### 3) Predict

```bash
python predict.py --weights runs/train_door_detect/weights/best.pt --source path/to/images_or_video
```

### Train on a custom YOLO-format dataset

Prepare a dataset YAML (same format as Ultralytics), then:

```bash
python train.py --data path/to/your_data.yaml --epochs 300 --batch 8
```

## Minimal Python API

```python
from ultralytics import YOLO

model = YOLO("ultralytics/train_door_detect/cfg/models/train_door_detect_model.yaml")
model.train(data="ultralytics/cfg/datasets/VisDrone.yaml", epochs=100, imgsz=640, batch=8, pretrained=False)

model = YOLO("runs/train_door_detect/weights/best.pt")
model.val(data="ultralytics/cfg/datasets/VisDrone.yaml")
model.predict(source="path/to/image.jpg", save=True)
```

## Citation

If you use this code, please cite our paper (Multimedia Systems) and the Ultralytics YOLO project:

```bibtex
@software{ultralytics_yolo,
  author  = {Jocher, Glenn and Qiu, Jing and others},
  title   = {Ultralytics YOLO},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## License

- Ultralytics source in this repository remains under **AGPL-3.0** (upstream license).
- Paper-specific modules are released together with this fork for peer review under the same repository terms unless a separate notice is added.

## Contact

For reviewer access questions related to this manuscript, please contact the corresponding author listed in the paper / response letter.
