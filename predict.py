"""Run inference with a trained checkpoint."""

import argparse
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Predict with train-door detector (paper)")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    p.add_argument("--source", required=True, help="Image / folder / video path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="", help="e.g. 0 or cpu; empty = auto")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="predict")
    p.add_argument("--save", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.weights)
    kwargs = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save=args.save,
    )
    if args.device != "":
        kwargs["device"] = args.device
    model.predict(**kwargs)
