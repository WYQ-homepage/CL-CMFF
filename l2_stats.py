"""Print three pre-fusion SALI L2 statistics per requested data split.

Example:
    python l2_stats.py --config configs/Vaihingen/Vaihingen_T.yaml \
        --weight weight/Vaihingen_T/<run>/stage3/best_model.pth --split test

Use a checkpoint for the full fusion path (all mode). Runs inference only,
with training-set normalization and no augmentation. Writes CSV to stdout.
"""
import argparse
from contextlib import redirect_stdout
import csv
import importlib
import json
import os
from pathlib import Path
import sys


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def collect_stats(model, loader, device, torch):
    # The module list is ordered by execution, starting at the deepest scale.
    modules = (model.fuse[2], model.fuse[1], model.fuse[0])
    batch_stats = {}

    def capture(level):
        def hook(module, inputs):
            if level in batch_stats:
                raise RuntimeError(f"SALI-{level + 1} ran more than once in a batch")
            norms = torch.stack([
                torch.linalg.vector_norm(feature, dtype=torch.float64)
                for feature in inputs
            ])
            optical, auxiliary = norms.unbind()
            ratio = 100 * auxiliary / (optical + auxiliary)
            batch_stats[level] = torch.stack((optical, auxiliary, ratio))
        return hook

    handles = [module.register_forward_pre_hook(capture(level))
               for level, module in enumerate(modules)]
    totals = torch.zeros((3, 3), device=device, dtype=torch.float64)
    samples = 0
    batches = 0
    try:
        with torch.inference_mode():
            for optical, auxiliary, _ in loader:
                batch_stats.clear()
                model(optical.to(device, non_blocking=True),
                      auxiliary.to(device, non_blocking=True))
                if len(batch_stats) != 3:
                    raise RuntimeError("Expected all three SALI inputs in full fusion mode")
                totals += torch.stack([batch_stats[level] for level in range(3)])
                samples += len(optical)
                batches += 1
    finally:
        for handle in handles:
            handle.remove()

    if batches == 0:
        raise ValueError("The selected split contains no samples")
    if not torch.isfinite(totals).all():
        raise ValueError("Non-finite SALI statistics or a batch with both feature norms zero")
    return samples, batches, (totals / batches).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset/model YAML")
    parser.add_argument("--weight", required=True, help="Full fusion checkpoint (.pth)")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--data-root", help="Override data.root_dir from the YAML")
    parser.add_argument("--batch-size", type=positive_int, default=2,
                        help="Inference batch size, including the last batch (default: 2)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--gpu", default="0", help="Physical CUDA GPU ID")
    parser.add_argument("--normalization", type=Path,
                        help="Optional train-set DatasetStats JSON to reuse")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    rows = []
    with redirect_stdout(sys.stderr):
        import torch
        from torch.utils.data import DataLoader

        from model.core import Model
        from utils import TrainConfig

        config = TrainConfig.from_yaml(args.config)
        name = config.data.name
        if name not in {"Vaihingen", "Potsdam", "WHU"}:
            parser.error(f"Unsupported dataset: {name}")
        root = args.data_root or config.data.root_dir
        pipeline = importlib.import_module(f"data_process.{name}")
        if args.normalization:
            stats = pipeline.DatasetStats(**json.loads(args.normalization.read_text()))
        elif config.data.calculate_stats:
            stats = pipeline.StatisticsCalculator(root, split="train").compute()
        else:
            stats = pipeline.DatasetStats()

        device = torch.device("cuda:0")
        model = Model.from_config(config.model, model_use="all", load_pretrained=False)
        state = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).eval()
        dataset_class = getattr(pipeline, f"{name}Dataset")
        for split in splits:
            dataset = dataset_class(root, split=split, stats=stats)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=False)
            samples, batches, means = collect_stats(model, loader, device, torch)
            for scale, (optical, auxiliary, ratio) in enumerate(means, start=1):
                rows.append((name, split, f"SALI-{scale}", samples, batches,
                             f"{optical:.6f}", f"{auxiliary:.6f}", f"{ratio:.6f}"))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("dataset", "split", "layer", "samples", "batches",
                     "O_L2", "X_L2", "RatioX_percent"))
    writer.writerows(rows)


if __name__ == "__main__":
    main()
