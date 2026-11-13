import os
import time
import math
import copy
import random
import datetime
import builtins
import argparse
import gc
import shutil
import logging
import hashlib
import json
import tempfile
from pathlib import Path
import sys
import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 引入 PyTorch 原生 Scheduler
from torch.optim.lr_scheduler import LambdaLR

# --- 自定义模块引入 ---
from model.core import Model
from model.Adapter import TargetAdapter, FeatureAdapter
from model.Loss import ModelLoss, FeatureLoss
from utils import TrainConfig
from val import validate
from reproducibility import make_data_generator, seed_worker, set_loader_epoch
# ==================== Training recovery ====================
FORMAT_VERSION = 1


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cpu_tree(value):
    """Copy tensor state to RAM, including nested optimizer state."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = type(value)((key, cpu_tree(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name("." + path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def select_run(weight_root, config, continue_training=False):
    """Select the newest unfinished run; never overwrite an existing run."""
    root = Path(weight_root)
    root.mkdir(parents=True, exist_ok=True)
    if continue_training:
        candidates = sorted((p for p in root.iterdir() if p.is_dir()
                             and not (p / "training_complete.json").exists()
                             and (p / "tmp" / "latest.pth").is_file()),
                            key=lambda p: p.name, reverse=True)
        if candidates:
            return str(candidates[0]), True
    run = root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(exist_ok=False)
    (run / "tmp").mkdir()
    return str(run), False


def capture_rng(loaders):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()
                    if loader is not None and loader.generator is not None},
    }


def restore_rng(state, loaders):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
    for name, rng in state["loaders"].items():
        loader = loaders.get(name)
        if loader is not None and loader.generator is not None:
            loader.generator.set_state(rng)


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


class TrainingResume:
    def __init__(self, weight_dir, config, resuming=False):
        self.weight_dir = Path(weight_dir)
        self.tmp = self.weight_dir / "tmp"
        self.tmp.mkdir(exist_ok=True)
        self.config = config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.group = dist.new_group(backend="gloo") if self.world_size > 1 else None
        self.checkpoint = None
        self.stage = 1
        if resuming:
            state = torch.load(self.tmp / "latest.pth", map_location="cpu", weights_only=False)
            if state.get("version") != FORMAT_VERSION:
                raise ValueError("Unsupported recovery checkpoint version")
            if state["config_digest"] != config_digest(config):
                raise ValueError("Recovery config differs from this run; use the saved YAML in the weight directory")
            if state["world_size"] != self.world_size:
                raise ValueError("Resume requires the original torchrun process count")
            stage = state["stage"]
            if stage not in (1, 2, 3):
                raise ValueError("Invalid recovery stage")
            total = getattr(config.phases, f"phase{stage}").scheduler.total_epochs
            if not 0 <= state["next_epoch"] <= total:
                raise ValueError("Invalid recovery epoch")
            self.checkpoint = state
            self.stage = stage

    def has_checkpoint(self, stage):
        return self.checkpoint is not None and self.checkpoint["stage"] == stage

    def restore(self, stage, modules, optimizer, scheduler, scaler, loaders):
        if not self.has_checkpoint(stage):
            return 0, -math.inf, {}, None
        state = self.checkpoint
        if set(state["modules"]) != set(modules):
            raise ValueError("Recovery module set does not match this training stage")
        for name, module in modules.items():
            unwrap(module).load_state_dict(state["modules"][name], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if (scheduler is None) != (state["scheduler"] is None):
            raise ValueError("Recovery scheduler setting differs")
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"][self.rank], loaders)
        result = state["next_epoch"], state["best_miou"], state["best_components"], state["best_epoch"]
        self.checkpoint = None
        return result

    def _primary_action(self, action):
        error = [None]
        if self.rank == 0:
            try:
                action()
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if self.world_size > 1:
            dist.broadcast_object_list(error, src=0, group=self.group, device=torch.device("cpu"))
        if error[0] is not None:
            raise RuntimeError(error[0])

    def save(self, stage, next_epoch, modules, optimizer, scheduler, scaler,
             best_miou, best_components, best_epoch, loaders):
        local_rng = capture_rng(loaders)
        states = [None] * self.world_size if self.rank == 0 else None
        if self.world_size > 1:
            dist.gather_object(local_rng, states, dst=0, group=self.group)
        else:
            states = [local_rng]

        def write():
            payload = {
                "version": FORMAT_VERSION, "config": self.config,
                "config_digest": config_digest(self.config), "world_size": self.world_size,
                "stage": stage, "next_epoch": next_epoch,
                "modules": {name: cpu_tree(unwrap(module).state_dict()) for name, module in modules.items()},
                "optimizer": cpu_tree(optimizer.state_dict()),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(), "rng": states,
                "best_miou": best_miou, "best_components": best_components, "best_epoch": best_epoch,
            }
            pending = self.tmp / ".latest.pth.pending"
            with pending.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, self.tmp / "latest.pth")
            atomic_json(self.tmp / "progress.json", {"stage": stage, "completed_epochs": next_epoch,
                                                      "config_digest": payload["config_digest"]})
        self._primary_action(write)

    def load_best(self, stage, names):
        directory = self.weight_dir / f"stage{stage}"
        manifest = json.loads((directory / "best_manifest.json").read_text(encoding="utf-8"))
        result = []
        for name in names:
            path = directory / name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != manifest["components"][name]["sha256"]:
                raise ValueError(f"Previous-stage weight checksum mismatch: {path}")
            result.append(torch.load(path, map_location="cpu", weights_only=True))
        return tuple(result)

    def finish(self):
        def complete():
            state = torch.load(self.tmp / "latest.pth", map_location="cpu", weights_only=False)
            total = self.config.phases.phase3.scheduler.total_epochs
            if state["stage"] != 3 or state["next_epoch"] != total or not math.isfinite(state["best_miou"]):
                raise RuntimeError("Cannot clean recovery files before all stages have completed")
            if self.tmp.is_symlink() or self.tmp.resolve().parent != self.weight_dir.resolve():
                raise RuntimeError("Refusing to remove a recovery directory outside this run")
            atomic_json(self.weight_dir / "training_complete.json", {
                "completed_at": datetime.datetime.now().isoformat(),
                "config_digest": config_digest(self.config), "final_stage": 3,
            })
            shutil.rmtree(self.tmp)
        self._primary_action(complete)

    def close(self):
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None


def cpu_state_dict(module):
    """Snapshot parameters and buffers on CPU without allocating a GPU copy."""
    state = module.state_dict()
    for name, value in state.items():
        state[name] = (value.detach().to(device="cpu", copy=True)
                       if torch.is_tensor(value) else copy.deepcopy(value))
    return state


def broadcast_cpu_weights(weights):
    """Keep both weight tensors and large serialization buffers off CUDA."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return weights
    cpu_group = dist.new_group(backend="gloo")
    try:
        objects = [weights if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(objects, src=0, group=cpu_group,
                                   device=torch.device("cpu"))
        return objects[0]
    finally:
        dist.destroy_process_group(cpu_group)


def save_stage_weights(weight_dir, stage, epoch, val_miou, states):
    """Write separate plain state_dict files; publish their manifest last."""
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    if "best_model.pth" not in states:
        raise ValueError("stage weights must include best_model.pth")
    if any(Path(name).name != name or not name.endswith(".pth") for name in states):
        raise ValueError("component filenames must be plain .pth filenames")
    stage_dir = Path(weight_dir) / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending_", dir=stage_dir) as temporary:
        pending = Path(temporary)
        components = {}
        for filename, state in states.items():
            path = pending / filename
            torch.save(state, path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            components[filename] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        metadata = {
            "stage": stage,
            "epoch": epoch,
            "selection": "best validation mIoU after the configured save threshold",
            "val_miou_percent": float(val_miou),
            "mode": {1: "f1", 2: "f2", 3: "all"}[stage],
            "format": "separate plain state_dict files",
            "components": components,
        }
        (pending / "best_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for filename in states:
            os.replace(pending / filename, stage_dir / filename)
        os.replace(pending / "best_manifest.json", stage_dir / "best_manifest.json")
    return str(stage_dir)


# ==================== LogManager ====================
class LogManager:
    def __init__(self, log_name, level=logging.INFO):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        self.formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

    def set_log_file(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "train_log.log")
        
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Log file handler added. Logging to: {log_file_path}")


def make_ddp(module, **kwargs):
    """Match convolution gradient layouts before DDP sees them.

    Singleton kernel/channel axes allow multiple contiguous strides. Autograd may
    retain channels-last strides while DDP buckets use the parameter strides.
    Copy only mismatched weight gradients, preserving their exact values.
    """
    for child in module.modules():
        if not isinstance(child, nn.Conv2d):
            continue
        weight = child.weight
        if not weight.requires_grad or 1 not in weight.shape:
            continue
        expected_stride = weight.stride()

        def align_gradient(gradient, stride=expected_stride):
            if gradient.stride() == stride:
                return gradient
            aligned = torch.empty_strided(
                gradient.size(), stride, dtype=gradient.dtype, device=gradient.device,
            )
            return aligned.copy_(gradient)

        weight.register_hook(align_gradient)
    return DDP(module, **kwargs)


# ==================== DDP Setup & Helpers ====================

def setup_ddp(configured_device_ids=None):
    if configured_device_ids is not None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not configured_device_ids or len(set(configured_device_ids)) != len(configured_device_ids):
            raise ValueError("device_ids must contain unique GPUs")
        if world_size != len(configured_device_ids):
            raise ValueError(f"WORLD_SIZE={world_size} does not match device_ids={configured_device_ids}; use train.py or matching torchrun process count")
        visible_devices = ",".join(map(str, configured_device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=device,
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=device,
        )
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_print(is_master):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master or kwargs.pop("force", False):
            builtin_print(*args, **kwargs)

    builtins.print = print


def reduce_tensor(tensor):
    if dist.is_initialized() and dist.get_world_size() > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
    return tensor


def seed_everything(seed: int):
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = seed + rank
    random.seed(seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_grad(module, requires_grad=True):
    for param in module.parameters():
        param.requires_grad = requires_grad


# ==================== Scheduler Helper ====================

def get_scheduler(optimizer, sched_cfg):
    """Linear warmup followed by cosine decay; step once at each epoch end.

    Epoch zero starts at 1% of the base LR when warmup is enabled.
    After warmup_epochs completed epochs the base LR is reached; after
    total_epochs completed epochs the LR reaches zero.
    """
    if not sched_cfg.use_lr_scheduler:
        return None
    total_epochs = sched_cfg.total_epochs
    warmup_epochs = sched_cfg.warmup_epochs if sched_cfg.use_warmup else 0
    if total_epochs <= 0 or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("Require 0 <= warmup_epochs < total_epochs")

    def lr_factor(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return 0.01 + 0.99 * epoch / warmup_epochs
        progress = min(max((epoch - warmup_epochs) / (total_epochs - warmup_epochs), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_factor)


def get_model_parameters(target_model):
    real_model = target_model.module if isinstance(target_model, DDP) else target_model
    param_groups = []

    fuse_param_ids = set()
    if hasattr(real_model, "fuse") and real_model.fuse is not None:
        for p in real_model.fuse.parameters():
            fuse_param_ids.add(id(p))

    encode_params = (
        [p for p in real_model.encode.parameters() if id(p) not in fuse_param_ids]
        if hasattr(real_model, "encode")
        else []
    )
    decode_params = (
        [p for p in real_model.decode.parameters() if id(p) not in fuse_param_ids]
        if hasattr(real_model, "decode")
        else []
    )

    if not encode_params and not decode_params:
        all_params = [p for p in real_model.parameters() if id(p) not in fuse_param_ids]
        param_groups.append({"params": all_params, "lr": None})
        param_groups.append({"params": [], "lr": None})
    else:
        param_groups.append({"params": encode_params, "lr": None})
        param_groups.append({"params": decode_params, "lr": None})

    if fuse_param_ids:
        fuse_params = list(real_model.fuse.parameters())
        param_groups.append({"params": fuse_params, "lr": None})

    return param_groups


def create_ema_model(model):
    """Snapshot the student only after DDP has synchronized its parameters."""
    source = model.module if isinstance(model, DDP) else model
    teacher = copy.deepcopy(source)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def update_ema_model(ema_model, model, momentum=0.999):
    source_model = model.module if isinstance(model, DDP) else model

    with torch.no_grad():
        for ema_param, src_param in zip(
            ema_model.parameters(), source_model.parameters()
        ):
            if ema_param.data.device != src_param.data.device:
                src_data = src_param.data.to(device=ema_param.data.device)
            else:
                src_data = src_param.data
            ema_param.data.mul_(momentum).add_(src_data, alpha=1 - momentum)

        for ema_buffer, src_buffer in zip(ema_model.buffers(), source_model.buffers()):
            if ema_buffer.data.device != src_buffer.data.device:
                src_data = src_buffer.data.to(device=ema_buffer.data.device)
            else:
                src_data = src_buffer.data
            ema_buffer.data.copy_(src_data)


# ==================== Training Phases ====================


def train_phase1(
    config,
    train_loader,
    val_loader,
    logger,
    device,
    train_sampler,
    weight_dir,
    use_bf16=False,
    resume_manager=None,
):
    phase_cfg = config.phases.phase1
    current_ema_momentum = float(phase_cfg.ema.momentum)
    if not 0.0 <= current_ema_momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")
    sched_cfg = phase_cfg.scheduler
    loss_cfg = phase_cfg.loss
    optim_cfg = phase_cfg.optimizer

    model_cfg = config.model
    save_start_epoch = sched_cfg.save_start_epoch

    if is_main_process():
        logger.info(f">>> Phase 1 <<<")

    model = Model.from_config(
        model_cfg, model_use="f1",
        load_pretrained=not (resume_manager is not None and resume_manager.has_checkpoint(1)),
    ).to(device)

    adapter = TargetAdapter(model_cfg.num_classes, channels=model.feature_channels[2:]).to(device)

    model.switch_to_f1()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = make_ddp(model, device_ids=[device], output_device=device)

    adapter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(adapter)
    adapter = make_ddp(adapter, device_ids=[device], output_device=device)

    ema_model = create_ema_model(model)
    ema_adapter = create_ema_model(adapter)

    optimizer_params = get_model_parameters(model)[:2]
    optimizer_params.append({"params": list(adapter.parameters()), "lr": None})

    initial_lrs = [optim_cfg.encode_lr, optim_cfg.decode_lr, optim_cfg.distill_proj_lr]
    optimizer_params = [g for g in optimizer_params if len(g["params"]) > 0]
    current_lrs = initial_lrs[: len(optimizer_params)]
    for i, lr in enumerate(current_lrs):
        optimizer_params[i]["lr"] = lr
    optimizer = optim.AdamW(optimizer_params, weight_decay=optim_cfg.weight_decay)

    # [Scheduler] 初始化 PyTorch 调度器
    scheduler = get_scheduler(optimizer, sched_cfg)
    if is_main_process():
        logger.info(f"Phase 1 Scheduler: enabled={sched_cfg.use_lr_scheduler}, epoch-level cosine, warmup_epochs={sched_cfg.warmup_epochs if sched_cfg.use_warmup else 0}, total_epochs={sched_cfg.total_epochs}, min_lr=0")

    scaler = torch.amp.GradScaler(enabled=use_bf16)

    train_loss_func = ModelLoss(
        main_weight=loss_cfg.main_weight,
        aux_weight=loss_cfg.aux_weight,
        ce_weight=loss_cfg.ce_weight,
        kd_weight=loss_cfg.kd_weight,
        ignore_index=model_cfg.ignore_index,
        soft_temperature=loss_cfg.distill_temp,
        aux_weights=[0.75, 0.5, 0.25, 0.1],
        use_weight=True,
    ).to(device)

    best_miou = -math.inf
    best_state = (None, None)

    checkpoint_modules = {"model": model, "adapter": adapter, "ema_model": ema_model, "ema_adapter": ema_adapter}
    checkpoint_loaders = {"train": train_loader, "val": val_loader}
    start_epoch, best_components, best_epoch = 0, {}, None
    if resume_manager is not None:
        start_epoch, best_miou, best_components, best_epoch = resume_manager.restore(
            1, checkpoint_modules, optimizer, scheduler, scaler, checkpoint_loaders,
        )
        if start_epoch and is_main_process():
            logger.info(f"Resuming Phase 1 after {start_epoch} completed epochs")
            if best_components:
                save_stage_weights(weight_dir, 1, best_epoch, best_miou, best_components)
        if best_components:
            best_state = (best_components["best_model.pth"], best_components["best_target_adapter.pth"])

    for epoch in range(start_epoch, sched_cfg.total_epochs):
        set_loader_epoch(train_loader, train_sampler, config.system.seed,
                         rank=dist.get_rank(), stage=1, epoch=epoch)
        model.train()
        adapter.train()

        train_loop = (
            tqdm(train_loader, desc=f"P1 Ep{epoch+1}", leave=False, ncols=120)
            if is_main_process()
            else train_loader
        )
        epoch_loss = 0.0

        for batch_idx, (rgb, x, labels) in enumerate(train_loop):
            inputs_1, inputs_2 = rgb.to(device, non_blocking=True), x.to(
                device, non_blocking=True
            )
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_bf16,
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
            ):
                with torch.no_grad():
                    t_main = ema_model(inputs_1, inputs_2)
                    t_feats = ema_model.get_distill_features()
                    teacher_outputs = ema_adapter([t_main] + t_feats)

                s_main = model(inputs_1, inputs_2)
                s_feats = model.module.get_distill_features()
                student_outputs = adapter([s_main] + s_feats)

                loss = train_loss_func(student_outputs, labels, [teacher_outputs])

            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Update EMA only when GradScaler did not skip the optimizer update.
            optimizer_updated = scaler.get_scale() >= previous_scale
            if optimizer_updated:
                update_ema_model(ema_model, model, current_ema_momentum)
                update_ema_model(ema_adapter, adapter, current_ema_momentum)

            reduced_loss = reduce_tensor(loss.detach())
            epoch_loss += reduced_loss.item()
            if is_main_process():
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_dec = optimizer.param_groups[1]["lr"]
                train_loop.set_postfix(
                    loss=reduced_loss.item(), enc=f"{lr_enc:.2e}", dec=f"{lr_dec:.2e}"
                )
        

        if is_main_process():
            train_loss_avg = epoch_loss / len(train_loader)
            if epoch >= save_start_epoch:
                miou, macro_f1, iou_per_cls = validate(
                    model.module,
                    val_loader,
                    device,
                    model_cfg.num_classes,
                    model_cfg.valid_classes,
                    True,
                )
                logger.info(
                    f"P1 Ep{epoch+1} | Loss: {train_loss_avg:.4f} | Val mIoU: {miou:.2f}% | EMA m: {current_ema_momentum:.5f}"
                )

                if math.isfinite(miou) and miou > best_miou:
                    best_miou = miou
                    best_state = (
                        cpu_state_dict(model.module),
                        cpu_state_dict(adapter.module),
                    )
                    best_epoch = epoch + 1
                    best_components = {
                        "best_model.pth": best_state[0],
                        "best_target_adapter.pth": best_state[1],
                        "best_ema_model.pth": cpu_state_dict(ema_model),
                        "best_ema_target_adapter.pth": cpu_state_dict(ema_adapter),
                    }
                    stage_dir = save_stage_weights(
                        weight_dir, stage=1, epoch=epoch + 1, val_miou=miou,
                        states=best_components,
                    )
                    logger.info(f"Model improved. Saved matched Stage 1 weights to {stage_dir}")
            else:
                logger.info(
                    f"P1 Ep{epoch+1} | Loss: {train_loss_avg:.4f} | EMA m: {current_ema_momentum:.5f} | Validation skipped before save_start_epoch"
                )

        # All ranks advance once after the completed epoch.
        if scheduler is not None:
            scheduler.step()
        if resume_manager is not None:
            resume_manager.save(
                1, epoch + 1, checkpoint_modules, optimizer, scheduler, scaler,
                best_miou, best_components, best_epoch, checkpoint_loaders,
            )
        dist.barrier()

    require_best_metric(best_miou, 1)
    del model, adapter, ema_model, ema_adapter, optimizer, scaler
    torch.cuda.empty_cache()

    best_state = broadcast_cpu_weights(best_state)
    return best_state[0], best_state[1]


def train_phase2(
    config,
    p1_weights,
    train_loader,
    val_loader,
    logger,
    device,
    train_sampler,
    weight_dir,
    use_bf16=False,
    resume_manager=None,
):
    phase_cfg = config.phases.phase2
    sched_cfg = phase_cfg.scheduler
    loss_cfg = phase_cfg.loss
    optim_cfg = phase_cfg.optimizer

    model_cfg = config.model
    save_start_epoch = sched_cfg.save_start_epoch

    if is_main_process():
        logger.info(f">>> Phase 2 <<<")

    # Init Student (f2) & Teacher (f1)
    model = Model.from_config(model_cfg, model_use="f1", load_pretrained=False).to(device)
    adapter = TargetAdapter(model_cfg.num_classes, channels=model.feature_channels[2:]).to(device)
    model.load_state_dict(p1_weights[0])
    adapter.load_state_dict(p1_weights[1])

    model.switch_to_f2()
    model.init_light_weights()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    adapter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(adapter)

    teacher_model = Model.from_config(model_cfg, model_use="f1", load_pretrained=False).to(device)
    teacher_adapter = TargetAdapter(model_cfg.num_classes, channels=teacher_model.feature_channels[2:]).to(device)
    teacher_model.load_state_dict(p1_weights[0])
    teacher_adapter.load_state_dict(p1_weights[1])

    teacher_model.eval()
    teacher_adapter.eval()

    feature_adapter = FeatureAdapter(
        channels=model.feature_channels, num_head=model.feature_heads,
        drop_path_rate=model.drop_path_rate,
    ).to(device)
    feature_adapter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_adapter)

    optimizer_params = get_model_parameters(model)[:2]
    optimizer_params.append({"params": list(adapter.parameters()), "lr": None})
    optimizer_params.append({"params": list(feature_adapter.parameters()), "lr": None})

    optimizer_params = [g for g in optimizer_params if len(g["params"]) > 0]

    feature_adapter_lr = optim_cfg.feature_adapter_lr
    initial_lrs = [
        optim_cfg.encode_lr,
        optim_cfg.decode_lr,
        optim_cfg.distill_proj_lr,
        feature_adapter_lr,
    ]

    current_lrs = initial_lrs[: len(optimizer_params)]
    for i, lr in enumerate(current_lrs):
        optimizer_params[i]["lr"] = lr

    optimizer = optim.AdamW(optimizer_params, weight_decay=optim_cfg.weight_decay)
    
    # [Scheduler] 初始化 PyTorch 调度器
    scheduler = get_scheduler(optimizer, sched_cfg)
    if is_main_process():
        logger.info(f"Phase 2 Scheduler: enabled={sched_cfg.use_lr_scheduler}, epoch-level cosine, warmup_epochs={sched_cfg.warmup_epochs if sched_cfg.use_warmup else 0}, total_epochs={sched_cfg.total_epochs}, min_lr=0")

    scaler = torch.amp.GradScaler(enabled=use_bf16)

    train_loss_func = ModelLoss(
        main_weight=loss_cfg.main_weight,
        aux_weight=loss_cfg.aux_weight,
        ce_weight=loss_cfg.ce_weight,
        kd_weight=loss_cfg.kd_weight,
        ignore_index=model_cfg.ignore_index,
        soft_temperature=loss_cfg.distill_temp,
        aux_weights=[0.75, 0.5, 0.25, 0.1],
        use_weight=True,
    ).to(device)
    feature_loss = FeatureLoss().to(device)

    best_miou = -math.inf
    best_state = (None, None, None)

    set_grad(adapter, requires_grad=True)
    model.phase2_main()

    model_ddp = make_ddp(model, device_ids=[device], output_device=device)
    adapter_ddp = make_ddp(adapter, device_ids=[device], output_device=device)
    feature_adapter_ddp = make_ddp(
        feature_adapter, device_ids=[device], output_device=device
    )

    checkpoint_modules = {"model": model_ddp, "adapter": adapter_ddp, "feature_adapter": feature_adapter_ddp}
    checkpoint_loaders = {"train": train_loader, "val": val_loader}
    start_epoch, best_components, best_epoch = 0, {}, None
    if resume_manager is not None:
        start_epoch, best_miou, best_components, best_epoch = resume_manager.restore(
            2, checkpoint_modules, optimizer, scheduler, scaler, checkpoint_loaders,
        )
        if start_epoch and is_main_process():
            logger.info(f"Resuming Phase 2 after {start_epoch} completed epochs")
            if best_components:
                save_stage_weights(weight_dir, 2, best_epoch, best_miou, best_components)
        if best_components:
            best_state = (best_components["best_model.pth"], best_components["best_target_adapter.pth"], best_components["best_feature_adapter.pth"])

    for epoch in range(start_epoch, sched_cfg.total_epochs):
        set_loader_epoch(train_loader, train_sampler, config.system.seed,
                         rank=dist.get_rank(), stage=2, epoch=epoch)
        model_ddp.train()
        adapter_ddp.train()
        feature_adapter_ddp.train()

        train_loop = (
            tqdm(train_loader, desc=f"P2 Main Ep{epoch+1}", leave=False, ncols=120)
            if is_main_process()
            else train_loader
        )
        epoch_loss = 0.0

        for batch_idx, (rgb, x, labels) in enumerate(train_loop):
            inputs_1, inputs_2 = rgb.to(device, non_blocking=True), x.to(
                device, non_blocking=True
            )
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_bf16,
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
            ):
                with torch.no_grad():
                    t_main = teacher_model(inputs_1, inputs_2)
                    t_distill = teacher_model.get_distill_features()
                    t_align = teacher_model.get_align_features()
                    teacher_outputs = teacher_adapter([t_main] + t_distill)

                s_main = model_ddp(inputs_1, inputs_2)
                s_distill = model_ddp.module.get_distill_features()
                s_align = model_ddp.module.get_align_features()
                student_outputs = adapter_ddp([s_main] + s_distill)

                teacher_features_adapted = feature_adapter_ddp(t_align, s_align)
                loss_feat = feature_loss(s_align, [teacher_features_adapted])
                loss_main = train_loss_func(student_outputs, labels, [teacher_outputs])

                loss = (loss_cfg.feature_weight * loss_feat) + loss_main

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            reduced_loss = reduce_tensor(loss.detach())
            epoch_loss += reduced_loss.item()
            if is_main_process():
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_dec = optimizer.param_groups[1]["lr"]
                train_loop.set_postfix(
                    loss=reduced_loss.item(), enc=f"{lr_enc:.2e}", dec=f"{lr_dec:.2e}"
                )
        

        if is_main_process():
            train_loss_avg = epoch_loss / len(train_loader)
            if epoch >= save_start_epoch:
                miou, macro_f1, iou_per_cls = validate(
                    model,
                    val_loader,
                    device,
                    model_cfg.num_classes,
                    model_cfg.valid_classes,
                    True,
                )
                logger.info(
                    f"P2 Main Ep{epoch+1} | Loss: {train_loss_avg:.4f} | Val mIoU: {miou:.2f}%"
                )

                if math.isfinite(miou) and miou > best_miou:
                    best_miou = miou
                    best_state = (
                        cpu_state_dict(model),
                        cpu_state_dict(adapter),
                        cpu_state_dict(feature_adapter),
                    )
                    best_epoch = epoch + 1
                    best_components = {
                        "best_model.pth": best_state[0],
                        "best_target_adapter.pth": best_state[1],
                        "best_feature_adapter.pth": best_state[2],
                    }
                    stage_dir = save_stage_weights(
                        weight_dir, stage=2, epoch=epoch + 1, val_miou=miou,
                        states=best_components,
                    )
                    logger.info(f"Model improved. Saved matched Stage 2 weights to {stage_dir}")
            else:
                logger.info(
                    f"P2 Main Ep{epoch+1} | Loss: {train_loss_avg:.4f} | Validation skipped before save_start_epoch"
                )

        # All ranks advance once after the completed epoch.
        if scheduler is not None:
            scheduler.step()
        if resume_manager is not None:
            resume_manager.save(
                2, epoch + 1, checkpoint_modules, optimizer, scheduler, scaler,
                best_miou, best_components, best_epoch, checkpoint_loaders,
            )
        dist.barrier()

    require_best_metric(best_miou, 2)
    del model_ddp, adapter_ddp, feature_adapter_ddp
    del (
        model,
        adapter,
        teacher_model,
        teacher_adapter,
        feature_adapter,
        optimizer,
        scaler,
    )
    torch.cuda.empty_cache()

    return broadcast_cpu_weights(best_state)


def train_phase3(
    config,
    p1_weights,
    p2_weights,
    p2_fa_weights,
    train_loader,
    val_loader,
    logger,
    weight_dir,
    device,
    train_sampler,
    use_bf16=False,
    resume_manager=None,
):
    phase_cfg = config.phases.phase3
    sched_cfg = phase_cfg.scheduler
    loss_cfg = phase_cfg.loss
    optim_cfg = phase_cfg.optimizer
    model_cfg = config.model

    if is_main_process():
        logger.info(f">>> Phase 3 <<<")

    # Init Models
    model = Model.from_config(model_cfg, model_use="all", load_pretrained=False).to(device)
    adapter = TargetAdapter(model_cfg.num_classes, channels=model.feature_channels[2:]).to(device)
    model.load_state_dict(p2_weights[0])
    adapter.load_state_dict(p2_weights[1])
    model.switch_to_all()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    adapter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(adapter)

    # Teachers (冻结模式)
    teacher_1 = Model.from_config(model_cfg, model_use="f1", load_pretrained=False).to(device)
    teacher_1.load_state_dict(p1_weights[0])
    teacher_1.eval()
    t_adapter_1 = TargetAdapter(model_cfg.num_classes, channels=teacher_1.feature_channels[2:]).to(device)
    t_adapter_1.load_state_dict(p1_weights[1])
    t_adapter_1.eval()

    teacher_2 = Model.from_config(model_cfg, model_use="f2", load_pretrained=False).to(device)
    teacher_2.load_state_dict(p2_weights[0])
    teacher_2.eval()
    t_adapter_2 = TargetAdapter(model_cfg.num_classes, channels=teacher_2.feature_channels[2:]).to(device)
    t_adapter_2.load_state_dict(p2_weights[1])
    t_adapter_2.eval()

    fa_1 = FeatureAdapter(
        channels=model.feature_channels, num_head=model.feature_heads,
        drop_path_rate=model.drop_path_rate,
    ).to(device)
    fa_1.load_state_dict(p2_fa_weights)
    fa_2 = FeatureAdapter(
        channels=model.feature_channels, num_head=model.feature_heads,
        drop_path_rate=model.drop_path_rate,
    ).to(device)
    fa_2.load_state_dict(p2_fa_weights)
    fa_1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fa_1)
    fa_2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fa_2)

    optimizer_params = get_model_parameters(model)
    optimizer_params.append({"params": list(adapter.parameters()), "lr": None})
    optimizer_params.append({"params": list(fa_1.parameters()), "lr": None})
    optimizer_params.append({"params": list(fa_2.parameters()), "lr": None})
    optimizer_params = [g for g in optimizer_params if len(g["params"]) > 0]

    feature_adapter_lr = optim_cfg.feature_adapter_lr
    fuse_lr = optim_cfg.fuse_lr
    initial_lrs = [
        optim_cfg.encode_lr,
        optim_cfg.decode_lr,
        fuse_lr,
        optim_cfg.distill_proj_lr,
        feature_adapter_lr,
        feature_adapter_lr,
    ]

    current_lrs = initial_lrs[: len(optimizer_params)]
    for i, lr in enumerate(current_lrs):
        optimizer_params[i]["lr"] = lr

    optimizer = optim.AdamW(optimizer_params, weight_decay=optim_cfg.weight_decay)
    
    # [Scheduler] 初始化 PyTorch 调度器
    scheduler = get_scheduler(optimizer, sched_cfg)
    if is_main_process():
        logger.info(f"Phase 3 Scheduler: enabled={sched_cfg.use_lr_scheduler}, epoch-level cosine, warmup_epochs={sched_cfg.warmup_epochs if sched_cfg.use_warmup else 0}, total_epochs={sched_cfg.total_epochs}, min_lr=0")

    scaler = torch.amp.GradScaler(enabled=use_bf16)

    train_loss_func = ModelLoss(
        main_weight=loss_cfg.main_weight,
        aux_weight=loss_cfg.aux_weight,
        ce_weight=loss_cfg.ce_weight,
        kd_weight=loss_cfg.kd_weight,
        ignore_index=model_cfg.ignore_index,
        soft_temperature=loss_cfg.distill_temp,
        aux_weights=[0.75, 0.5, 0.25, 0.1],
        use_weight=True,
    ).to(device)
    feature_loss = FeatureLoss().to(device)

    best_miou = -math.inf

    set_grad(adapter, requires_grad=True)
    set_grad(fa_1, requires_grad=True)
    set_grad(fa_2, requires_grad=True)
    freeze_shallow = getattr(phase_cfg, "freeze_shallow", True)
    if not isinstance(freeze_shallow, bool):
        raise ValueError("phases.phase3.freeze_shallow must be a YAML boolean")
    model.phase3_main(freeze_shallow=freeze_shallow)
    if is_main_process():
        logger.info(f"Phase 3 freeze_shallow={freeze_shallow}; backbone={model.backbone}; "
                    f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model_ddp = make_ddp(model, device_ids=[device], output_device=device)
    adapter_ddp = make_ddp(adapter, device_ids=[device], output_device=device)
    fa_1_ddp = make_ddp(fa_1, device_ids=[device], output_device=device)
    fa_2_ddp = make_ddp(fa_2, device_ids=[device], output_device=device)

    checkpoint_modules = {"model": model_ddp, "adapter": adapter_ddp, "fa_1": fa_1_ddp, "fa_2": fa_2_ddp}
    checkpoint_loaders = {"train": train_loader, "val": val_loader}
    start_epoch, best_components, best_epoch = 0, {}, None
    if resume_manager is not None:
        start_epoch, best_miou, best_components, best_epoch = resume_manager.restore(
            3, checkpoint_modules, optimizer, scheduler, scaler, checkpoint_loaders,
        )
        if start_epoch and is_main_process():
            logger.info(f"Resuming Phase 3 after {start_epoch} completed epochs")
            if best_components:
                save_stage_weights(weight_dir, 3, best_epoch, best_miou, best_components)

    for epoch in range(start_epoch, sched_cfg.total_epochs):
        set_loader_epoch(train_loader, train_sampler, config.system.seed,
                         rank=dist.get_rank(), stage=3, epoch=epoch)
        model_ddp.train()
        adapter_ddp.train()
        fa_1_ddp.train()
        fa_2_ddp.train()

        train_loop = (
            tqdm(train_loader, desc=f"P3 Main Ep{epoch+1}", leave=False, ncols=120)
            if is_main_process()
            else train_loader
        )
        epoch_loss = 0.0

        for batch_idx, (rgb, x, labels) in enumerate(train_loop):
            inputs_1, inputs_2 = rgb.to(device, non_blocking=True), x.to(
                device, non_blocking=True
            )
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_bf16,
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
            ):
                with torch.no_grad():
                    t1_main = teacher_1(inputs_1, inputs_2)
                    t2_main = teacher_2(inputs_1, inputs_2)
                    t1_distill = teacher_1.get_distill_features()
                    t2_distill = teacher_2.get_distill_features()

                    out_t1 = t_adapter_1([t1_main] + t1_distill)
                    out_t2 = t_adapter_2([t2_main] + t2_distill)
                    t1_align = teacher_1.get_align_features()
                    t2_align = teacher_2.get_align_features()

                s_main = model_ddp(inputs_1, inputs_2)
                s_distill = model_ddp.module.get_distill_features()
                student_out = adapter_ddp([s_main] + s_distill)
                s_align = model_ddp.module.get_align_features()

                t1_feat = fa_1_ddp(t1_align, s_align)
                t2_feat = fa_2_ddp(t2_align, s_align)

                loss = (
                    loss_cfg.feature_weight * feature_loss(s_align, [t1_feat, t2_feat])
                ) + train_loss_func(student_out, labels, [out_t1, out_t2])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            reduced_loss = reduce_tensor(loss.detach())
            epoch_loss += reduced_loss.item()
            if is_main_process():
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_dec = optimizer.param_groups[1]["lr"]
                train_loop.set_postfix(
                    loss=reduced_loss.item(), enc=f"{lr_enc:.2e}", dec=f"{lr_dec:.2e}"
                )
        

        if is_main_process():
            train_loss_avg = epoch_loss / len(train_loader)
            if epoch >= sched_cfg.save_start_epoch:
                miou, macro_f1, iou_per_cls = validate(
                    model,
                    val_loader,
                    device,
                    model_cfg.num_classes,
                    model_cfg.valid_classes,
                    True,
                )
                logger.info(
                    f"P3 Main Ep{epoch+1} | Loss: {train_loss_avg:.4f} | Val mIoU: {miou:.2f}%"
                )

                if math.isfinite(miou) and miou > best_miou:
                    best_miou = miou
                    best_epoch = epoch + 1
                    best_components = {
                        "best_model.pth": cpu_state_dict(model),
                        "best_target_adapter.pth": cpu_state_dict(adapter),
                        "best_feature_adapter_1.pth": cpu_state_dict(fa_1),
                        "best_feature_adapter_2.pth": cpu_state_dict(fa_2),
                    }
                    stage_dir = save_stage_weights(
                        weight_dir, stage=3, epoch=epoch + 1, val_miou=miou,
                        states=best_components,
                    )
                    logger.info(f"Model improved. Saved matched Stage 3 weights to {stage_dir}")
            else:
                logger.info(
                    f"P3 Main Ep{epoch+1} | Loss: {train_loss_avg:.4f} | Validation skipped before save_start_epoch"
                )

        # All ranks advance once after the completed epoch.
        if scheduler is not None:
            scheduler.step()
        if resume_manager is not None:
            resume_manager.save(
                3, epoch + 1, checkpoint_modules, optimizer, scheduler, scaler,
                best_miou, best_components, best_epoch, checkpoint_loaders,
            )
        dist.barrier()

    require_best_metric(best_miou, 3)
    del model_ddp, adapter_ddp, fa_1_ddp, fa_2_ddp, scaler
    torch.cuda.empty_cache()
    dist.barrier()


def require_best_metric(best_miou, stage):
    result = [best_miou if is_main_process() else None]
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if not math.isfinite(result[0]):
        raise RuntimeError(f"Stage {stage} has no finite best validation metric/checkpoint")


def validate_training_config(config):
    for name in ("phase1", "phase2", "phase3"):
        phase = getattr(config.phases, name)
        epochs = phase.scheduler.total_epochs
        start = phase.scheduler.save_start_epoch
        if epochs <= 0 or not 0 <= start < epochs:
            raise ValueError(f"{name}: require total_epochs > 0 and 0 <= save_start_epoch < total_epochs")


def run(config, config_path, continue_training=False):
    validate_training_config(config)
    configured_gpus = config.system.device_ids
    rank, local_rank, world_size = setup_ddp(configured_device_ids=configured_gpus)
    setup_print(is_main_process())
    seed_everything(config.system.seed)

    device = torch.device(f"cuda:{local_rank}")

    use_bf16 = config.system.use_bf16
    
    # [Init LogManager]
    logger = None
    log_manager = None
    if is_main_process():
        log_manager = LogManager(config.system.log_name, level=logging.DEBUG)
        logger = log_manager.get_logger()
        
        gpu_name = torch.cuda.get_device_name(local_rank)
        logger.info(
            f"System: Rank {rank} | Logical GPU {local_rank} (Physical Name: {gpu_name})"
        )
        logger.info(f"Precision: BF16 Enabled: {use_bf16}")

        if use_bf16 and not torch.cuda.is_bf16_supported():
            logger.warning(
                "BF16 enabled in config but current hardware might not fully support it (Requires Ampere+)."
            )

    selection = None
    if is_main_process():
        try:
            directory, resuming = select_run(
                Path("weight") / config.system.weight_dir_name, config, continue_training,
            )
            selection = {"directory": directory, "resuming": resuming, "error": None}
        except Exception as exc:
            selection = {"error": str(exc)}
    selection = broadcast_cpu_weights(selection)
    if selection["error"]:
        raise RuntimeError(selection["error"])
    weight_dir = selection["directory"]
    resume_manager = TrainingResume(weight_dir, config, resuming=selection["resuming"])
    if is_main_process():
        for stage in (1, 2, 3):
            os.makedirs(os.path.join(weight_dir, f"stage{stage}"), exist_ok=True)
        if not selection["resuming"]:
            target_config_path = os.path.join(weight_dir, os.path.basename(config_path))
            shutil.copy(config_path, target_config_path)
            logger.info(f"Config file backed up to: {target_config_path}")
        log_manager.set_log_file(weight_dir)
        if selection["resuming"]:
            logger.info(f"Continuing unfinished run: {weight_dir}; Phase {resume_manager.stage}")
        elif continue_training:
            logger.info("No unfinished epoch checkpoint found; starting a new run")

    if config.data.name == "Vaihingen":
        from data_process.Vaihingen import PipLine
    if config.data.name == "Potsdam":
        from data_process.Potsdam import PipLine
    if config.data.name == "WHU":
        from data_process.WHU import PipLine
    if config.data.name == "Levir_CS":
        from data_process.Levir_CS import PipLine

    data_manager = PipLine(
        root_dir=config.data.root_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=getattr(config.data, "pin_memory", True),
        calculate_stats=config.data.calculate_stats,
        seed=config.system.seed,
    )

    train_dataset = data_manager.get_loader("train", augment=True, shuffle=True).dataset
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=config.system.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        generator=make_data_generator(config.system.seed, rank=rank),
        worker_init_fn=seed_worker,
        persistent_workers=False,
        num_workers=config.data.num_workers,
        pin_memory=data_manager.pin_memory,
        drop_last=True,
    )

    if is_main_process():
        val_loader = data_manager.get_loader("val", augment=False, shuffle=False)
    else:
        val_loader = None

    if resume_manager.stage <= 1:
        p1_model_weights, p1_adapter_weights = train_phase1(
            config, train_loader, val_loader, logger, device, train_sampler,
            weight_dir, use_bf16=use_bf16, resume_manager=resume_manager,
        )
    else:
        p1_model_weights, p1_adapter_weights = resume_manager.load_best(
            1, ("best_model.pth", "best_target_adapter.pth"),
        )
    if resume_manager.stage <= 2:
        p2_weights = train_phase2(
            config, [p1_model_weights, p1_adapter_weights], train_loader,
            val_loader, logger, device, train_sampler, weight_dir,
            use_bf16=use_bf16, resume_manager=resume_manager,
        )
    else:
        p2_weights = resume_manager.load_best(
            2, ("best_model.pth", "best_target_adapter.pth", "best_feature_adapter.pth"),
        )
    train_phase3(
        config, [p1_model_weights, p1_adapter_weights], p2_weights[:2], p2_weights[2],
        train_loader, val_loader, logger, weight_dir, device, train_sampler,
        use_bf16=use_bf16, resume_manager=resume_manager,
    )
    resume_manager.finish()
    resume_manager.close()
    if is_main_process():
        logger.info("All training stages completed; tmp recovery directory removed")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/Potsdam/Potsdam_T.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--continue", dest="continue_training", action="store_true",
        help="Resume the newest unfinished run from its latest completed epoch",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    run(config, args.config, continue_training=args.continue_training)
