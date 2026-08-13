"""Validate a trained checkpoint on a YOLO-format detection dataset."""

import argparse
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Validate train-door detector (paper)")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--data",
        default="ultralytics/cfg/datasets/VisDrone.yaml",
        help="Dataset YAML",
    )
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="", help="e.g. 0 or cpu; empty = auto")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.weights)
    kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
    )
    if args.device != "":
        kwargs["device"] = args.device
    model.val(**kwargs)
