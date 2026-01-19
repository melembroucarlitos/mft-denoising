#!/usr/bin/env python3
"""
batch_diagnostics.py

Batch size and gradient noise diagnostics for trained models.

Usage examples:
  # Sweep batches and report progress/sec:
  python batch_diagnostics.py --checkpoint experiments/exp_*/checkpoint_epoch_5.pth --mode sweep --batches 128,256,512,1024 --steps 400 --base_batch 256 --base_lr 1e-3 --lr_scale sqrt --amp

  # Gradient-noise diagnostic at a given minibatch size:
  python batch_diagnostics.py --checkpoint experiments/exp_*/checkpoint_epoch_5.pth --mode gradnoise --minibatch 256 --K 8 --pre_steps 200 --base_lr 1e-3 --amp

  # Both:
  python batch_diagnostics.py --checkpoint experiments/exp_*/checkpoint_epoch_5.pth --mode both --batches 128,256,512,1024 --minibatch 256 --K 8 --steps 300 --pre_steps 200 --amp
"""

import argparse
import math
import time
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import sys

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mft_denoising.nn import TwoLayerNet
from mft_denoising.data import TwoHotStream, DataConfig
from mft_denoising.losses import create_loss_function
from mft_denoising.config import ExperimentConfig, ModelConfig, TrainingConfig, LossConfig


# -----------------------------
# USER HOOKS: Adapted for TwoLayerNet
# -----------------------------

def build_model(config: ExperimentConfig, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Build and load model from checkpoint.
    
    Args:
        config: Experiment configuration
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model on specified device
    """
    # #region agent log
    import json
    with open('/home/ubuntu/code/mft-denoising/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_diagnostics.py:56","message":"build_model entry","data":{"checkpoint_path":str(checkpoint_path),"device":str(device),"d":config.data.d,"hidden_size":config.model.hidden_size},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    
    model = TwoLayerNet(
        input_size=config.data.d,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=1.0,  # Doesn't matter, loading from checkpoint
        decoder_initialization_scale=1.0,
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle compiled model checkpoints (strip _orig_mod. prefix if present)
    # When models are saved after torch.compile(), state_dict keys have _orig_mod. prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_key = k[len('_orig_mod.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    
    # Move to device
    if device.type == "cuda":
        model = model.cuda()
        # Compile if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"[warn] torch.compile failed, continuing uncompiled: {e}")
    else:
        model = model.to(device)
    
    return model


def make_batch(batch_size: int, device: torch.device, data_config: DataConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data on the specified device.
    
    Args:
        batch_size: Batch size
        device: Device to generate on
        data_config: Data configuration
    
    Returns:
        (x, x_star) tuple where x is noisy input and x_star is clean target
    """
    # Create a stream for generating data
    stream = TwoHotStream(data_config)
    
    # Generate batch (returns CPU tensors)
    x, x_star = stream.sample_batch(batch_size)
    
    # Move to device
    x = x.to(device, non_blocking=True)
    x_star = x_star.to(device, non_blocking=True)
    
    return (x, x_star)


def loss_fn(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], loss_function) -> torch.Tensor:
    """
    Compute scalar loss given model + batch.
    
    Args:
        model: Model to evaluate
        batch: (x, x_star) tuple
        loss_function: Loss function from create_loss_function
    
    Returns:
        Scalar loss tensor
    """
    x, x_star = batch
    output = model(x)
    total_loss, _, _ = loss_function(output=output, label=x_star)
    return total_loss


# -----------------------------
# END USER HOOKS
# -----------------------------


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_enable_tf32(enable: bool):
    """Enable TF32 if requested."""
    if enable:
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def make_optimizer(model: nn.Module, lr: float, wd: float, fused: bool):
    """Create optimizer (AdamW with optional fused kernel)."""
    kwargs = dict(lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    if fused:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)


@dataclass
class SweepResult:
    """Results from a batch size sweep."""
    batch_size: int
    lr: float
    steps: int
    seconds: float
    steps_per_sec: float
    logloss_drop_per_sec: float
    final_loss: float


def lr_scaled(base_lr: float, base_batch: int, batch: int, mode: str) -> float:
    """Scale learning rate based on batch size."""
    if mode == "none":
        return base_lr
    scale = batch / base_batch
    if mode == "sqrt":
        return base_lr * math.sqrt(scale)
    if mode == "linear":
        return base_lr * scale
    raise ValueError(f"Unknown lr_scale {mode}")


@torch.no_grad()
def quick_eval_loss(model: nn.Module, device: torch.device, data_config: DataConfig, 
                   loss_function, batch_size: int, iters: int = 10) -> float:
    """Quick evaluation loss estimate."""
    model.eval()
    losses = []
    for _ in range(iters):
        batch = make_batch(batch_size, device, data_config)
        loss = loss_fn(model, batch, loss_function)
        losses.append(loss.detach().float().item())
    model.train()
    return sum(losses) / len(losses)


def run_sweep_one(
    batch_size: int,
    steps: int,
    warmup_steps: int,
    base_lr: float,
    base_batch: int,
    lr_scale_mode: str,
    wd: float,
    amp: bool,
    amp_dtype: str,
    fused_opt: bool,
    tf32: bool,
    compile_model: bool,
    seed: int,
    device: torch.device,
    config: ExperimentConfig,
    model: nn.Module,
    loss_function,
) -> SweepResult:
    """Run a single batch size sweep."""
    set_seeds(seed)
    maybe_enable_tf32(tf32)

    # Use provided model (already loaded and on device)
    if compile_model and not hasattr(model, '_orig_mod'):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile failed, continuing uncompiled: {e}")

    lr = lr_scaled(base_lr, base_batch, batch_size, lr_scale_mode)
    opt = make_optimizer(model, lr=lr, wd=wd, fused=fused_opt)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    # Warmup for timing stability
    model.train()
    for _ in range(min(10, steps // 10)):
        batch = make_batch(batch_size, device, config.data)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp, dtype=autocast_dtype):
            loss = loss_fn(model, batch, loss_function)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # Real run (time + log losses)
    log_losses = []
    t0 = time.perf_counter()

    for s in range(steps):
        batch = make_batch(batch_size, device, config.data)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp, dtype=autocast_dtype):
            loss = loss_fn(model, batch, loss_function)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        if s >= warmup_steps:
            log_losses.append(loss.detach().float().item())

    t1 = time.perf_counter()
    elapsed = t1 - t0

    steps_per_sec = steps / elapsed
    final_loss = log_losses[-1] if log_losses else float('nan')
    
    # Compute log loss drop per second
    if len(log_losses) > 1:
        initial_log_loss = math.log(log_losses[0])
        final_log_loss = math.log(log_losses[-1])
        logloss_drop = initial_log_loss - final_log_loss
        logloss_drop_per_sec = logloss_drop / elapsed
    else:
        logloss_drop_per_sec = float('nan')

    return SweepResult(
        batch_size=batch_size,
        lr=lr,
        steps=steps,
        seconds=elapsed,
        steps_per_sec=steps_per_sec,
        logloss_drop_per_sec=logloss_drop_per_sec,
        final_loss=final_loss,
    )


@dataclass
class GradNoiseResult:
    """Results from gradient noise diagnostic."""
    minibatch: int
    K: int
    grad_norm_mean: float
    grad_norm_std: float
    grad_noise_ratio: float


def run_gradnoise(
    minibatch: int,
    K: int,
    pre_steps: int,
    base_lr: float,
    wd: float,
    amp: bool,
    amp_dtype: str,
    fused_opt: bool,
    tf32: bool,
    seed: int,
    device: torch.device,
    config: ExperimentConfig,
    model: nn.Module,
    loss_function,
) -> GradNoiseResult:
    """Run gradient noise diagnostic."""
    set_seeds(seed)
    maybe_enable_tf32(tf32)

    opt = make_optimizer(model, lr=base_lr, wd=wd, fused=fused_opt)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    # Pre-steps to stabilize
    model.train()
    for _ in range(pre_steps):
        batch = make_batch(minibatch, device, config.data)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp, dtype=autocast_dtype):
            loss = loss_fn(model, batch, loss_function)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # Collect K gradient samples
    grad_norms = []
    for k in range(K):
        batch = make_batch(minibatch, device, config.data)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp, dtype=autocast_dtype):
            loss = loss_fn(model, batch, loss_function)
        scaler.scale(loss).backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        scaler.step(opt)
        scaler.update()

    grad_norms = torch.tensor(grad_norms)
    grad_norm_mean = grad_norms.mean().item()
    grad_norm_std = grad_norms.std().item()
    grad_noise_ratio = grad_norm_std / grad_norm_mean if grad_norm_mean > 0 else float('nan')

    return GradNoiseResult(
        minibatch=minibatch,
        K=K,
        grad_norm_mean=grad_norm_mean,
        grad_norm_std=grad_norm_std,
        grad_noise_ratio=grad_noise_ratio,
    )


def load_config_from_checkpoint(checkpoint_path: Path) -> Optional[ExperimentConfig]:
    """Try to load config.json from experiment directory."""
    # Check parent directory (experiment dir) for config.json
    exp_dir = checkpoint_path.parent
    config_path = exp_dir / "config.json"
    
    if config_path.exists():
        try:
            return ExperimentConfig.load_json(config_path)
        except Exception as e:
            print(f"[warn] Failed to load config.json: {e}")
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description='Batch diagnostics for trained models')
    
    # Checkpoint and config
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pth)')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['sweep', 'gradnoise', 'both'], default='sweep',
                       help='Diagnostic mode')
    
    # Sweep mode args
    parser.add_argument('--batches', type=str, default='128,256,512,1024',
                       help='Comma-separated batch sizes to test')
    parser.add_argument('--steps', type=int, default=400,
                       help='Number of steps per batch size')
    parser.add_argument('--base_batch', type=int, default=256,
                       help='Base batch size for LR scaling')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                       help='Base learning rate')
    parser.add_argument('--lr_scale', type=str, choices=['none', 'sqrt', 'linear'], default='sqrt',
                       help='LR scaling mode')
    parser.add_argument('--warmup_steps', type=int, default=10,
                       help='Warmup steps before timing')
    
    # Gradient noise args
    parser.add_argument('--minibatch', type=int, default=256,
                       help='Minibatch size for gradient noise')
    parser.add_argument('--K', type=int, default=8,
                       help='Number of gradient samples')
    parser.add_argument('--pre_steps', type=int, default=200,
                       help='Pre-steps before gradient noise measurement')
    
    # Training options
    parser.add_argument('--amp', action='store_true',
                       help='Use mixed precision')
    parser.add_argument('--amp_dtype', type=str, choices=['fp16', 'bf16'], default='fp16',
                       help='Mixed precision dtype')
    parser.add_argument('--fused', action='store_true',
                       help='Use fused optimizer')
    parser.add_argument('--tf32', action='store_true',
                       help='Enable TF32')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile')
    parser.add_argument('--wd', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model/data config (if no config.json found)
    parser.add_argument('--d', type=int, default=None,
                       help='Input dimension (overrides config)')
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Hidden size (overrides config)')
    parser.add_argument('--sparsity', type=int, default=None,
                       help='Sparsity (overrides config)')
    parser.add_argument('--noise_variance', type=float, default=None,
                       help='Noise variance (overrides config)')
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Try to load config from experiment directory
    config = load_config_from_checkpoint(checkpoint_path)
    
    # If no config found, create minimal config from args or defaults
    if config is None:
        print("[info] No config.json found, using defaults/CLI args")
        d = args.d if args.d is not None else 1024
        hidden_size = args.hidden_size if args.hidden_size is not None else 512
        sparsity = args.sparsity if args.sparsity is not None else 2
        noise_variance = args.noise_variance if args.noise_variance is not None else 0.005
        
        config = ExperimentConfig(
            model=ModelConfig(hidden_size=hidden_size),
            training=TrainingConfig(),
            loss=LossConfig(),
            data=DataConfig(
                d=d,
                sparsity=sparsity,
                noise_variance=noise_variance,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
    else:
        # Override with CLI args if provided
        if args.d is not None:
            config.data.d = args.d
        if args.hidden_size is not None:
            config.model.hidden_size = args.hidden_size
        if args.sparsity is not None:
            config.data.sparsity = args.sparsity
        if args.noise_variance is not None:
            config.data.noise_variance = args.noise_variance
    
    # Determine device
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Model config: d={config.data.d}, hidden_size={config.model.hidden_size}")
    print(f"Device: {device}")
    
    # Build model and load checkpoint
    model = build_model(config, checkpoint_path, device)
    
    # Create loss function
    loss_function = create_loss_function(config.loss)
    
    # Run diagnostics
    if args.mode in ['sweep', 'both']:
        print("\n" + "=" * 80)
        print("BATCH SIZE SWEEP")
        print("=" * 80)
        
        batch_sizes = [int(b.strip()) for b in args.batches.split(',')]
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            # Reload model from checkpoint for each batch size to ensure fair comparison
            model = build_model(config, checkpoint_path, device)
            result = run_sweep_one(
                batch_size=batch_size,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                base_lr=args.base_lr,
                base_batch=args.base_batch,
                lr_scale_mode=args.lr_scale,
                wd=args.wd,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                fused_opt=args.fused,
                tf32=args.tf32,
                compile_model=args.compile,
                seed=args.seed,
                device=device,
                config=config,
                model=model,
                loss_function=loss_function,
            )
            results.append(result)
            print(f"  Steps/sec: {result.steps_per_sec:.2f}")
            print(f"  Final loss: {result.final_loss:.6f}")
            print(f"  Log loss drop/sec: {result.logloss_drop_per_sec:.6f}")
        
        # Print summary table
        print("\n" + "=" * 80)
        print("SWEEP RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Batch':<8} {'LR':<12} {'Steps/sec':<12} {'Final Loss':<12} {'LogLoss Drop/sec':<18}")
        print("-" * 80)
        for r in results:
            print(f"{r.batch_size:<8} {r.lr:<12.6f} {r.steps_per_sec:<12.2f} {r.final_loss:<12.6f} {r.logloss_drop_per_sec:<18.6f}")
    
    if args.mode in ['gradnoise', 'both']:
        print("\n" + "=" * 80)
        print("GRADIENT NOISE DIAGNOSTIC")
        print("=" * 80)
        
        result = run_gradnoise(
            minibatch=args.minibatch,
            K=args.K,
            pre_steps=args.pre_steps,
            base_lr=args.base_lr,
            wd=args.wd,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            fused_opt=args.fused,
            tf32=args.tf32,
            seed=args.seed,
            device=device,
            config=config,
            model=model,
            loss_function=loss_function,
        )
        
        print(f"Minibatch size: {result.minibatch}")
        print(f"Gradient samples (K): {result.K}")
        print(f"Gradient norm mean: {result.grad_norm_mean:.6f}")
        print(f"Gradient norm std: {result.grad_norm_std:.6f}")
        print(f"Gradient noise ratio (std/mean): {result.grad_noise_ratio:.6f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
