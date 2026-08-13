"""Train the paper detector (SPPSA-StarNet + CFDPN + DESD).

Default demo uses the public VisDrone detection split for reproducibility
(industrial train-door data and the localization pipeline are not released).
"""

import argparse
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train train-door detector (paper)")
    p.add_argument(
        "--model",
        default="ultralytics/train_door_detect/cfg/models/train_door_detect_model.yaml",
        help="Model YAML (paper architecture)",
    )
    p.add_argument(
        "--data",
        default="ultralytics/cfg/datasets/VisDrone.yaml",
        help="Dataset YAML (VisDrone by default; replace with your YOLO-format data)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="", help="e.g. 0 or cpu; empty = auto")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="train_door_detect")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.model)
    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=False,
        amp=args.amp,
    )
    if args.device != "":
        kwargs["device"] = args.device
    model.train(**kwargs)
