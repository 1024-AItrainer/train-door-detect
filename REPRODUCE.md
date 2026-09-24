# Quick reproducibility checklist (for reviewers)

1. Python >= 3.8, install matching PyTorch for your CUDA/CPU.
2. `pip install -r requirements.txt`
3. `pip install -e .`  (installs this fork as local `ultralytics==8.3.225`)
4. Sanity-check model build:

```bash
python -c "from ultralytics import YOLO; m=YOLO('ultralytics/train_door_detect/cfg/models/train_door_detect_model.yaml'); m.info()"
```

5. Smoke train on a public detection split (VisDrone downloads on first use):

```bash
python train.py --data ultralytics/cfg/datasets/VisDrone.yaml --epochs 1 --batch 2 --workers 2
```

6. Validate / predict with the saved weights using `val.py` / `predict.py`.

Expected: model builds with ~1.0–1.2M parameters / ~4.3 GFLOPs at scale `n` (exact print may vary slightly by Ultralytics counting mode).
