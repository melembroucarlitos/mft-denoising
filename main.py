import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple, List
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

from mft_denoising.data import create_dataloaders
from mft_denoising.nn import TwoLayerNet
from mft_denoising.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
)
from mft_denoising.data import DataConfig
from mft_denoising.losses import create_loss_function
from mft_denoising.optimizers import create_optimizer, SGLD
from mft_denoising.experiment import ExperimentTracker
from mft_denoising.clustering import cluster_encoder_weights, sample_frozen_encoder



def train(
    model,
    train_loader,
    test_loader,
    config: ExperimentConfig,
    tracker: Optional[ExperimentTracker] = None,
):
    """
    Train model using configured optimizer (SGLD or ADAM) with configured loss function.
    
    Args:
        model: TwoLayerNet model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Experiment configuration
        tracker: Optional experiment tracker for logging
    
    Returns:
        Trained model
    """
    # Validate device - fall back to CPU if CUDA is requested but not available
    device_str = config.data.device
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        use_cuda = False
    
    device = torch.device(device_str)
    
    # Move model to CUDA if available (using .cuda() for better performance)
    if use_cuda:
        model = model.cuda()
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
    else:
        model = model.to(device)
    
    # Create optimizer based on config
    optimizer = create_optimizer(model, config.training, use_cuda=use_cuda)
    
    # Create loss function based on config
    loss_fn = create_loss_function(config.loss)
    
    # Create GradScaler for mixed precision training (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    
    # Create learning rate scheduler with warmup (only for standard optimizers, not SGLD)
    scheduler = None
    if config.training.enable_warmup and isinstance(optimizer, torch.optim.Optimizer):
        total_steps = config.training.epochs * len(train_loader)
        warmup_steps = int(config.training.warmup_fraction * total_steps)
        
        def lr_fn(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
        print(f"Learning rate scheduler enabled: {warmup_steps} warmup steps, {total_steps} total steps")
    
    # Start tracking if provided
    if tracker is not None:
        tracker.start()
    
    # Track global step for scheduler
    global_step = 0
    
    for epoch in range(config.training.epochs):
        model.train()
        total_train_loss = 0
        total_train_scaled_loss = 0
        total_train_loss_on_w = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Use non_blocking=True for CUDA data transfer
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            # Use set_to_none=True for zero_grad when available
            if hasattr(optimizer, 'zero_grad'):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            # Mixed precision training for CUDA (only for standard optimizers, not SGLD)
            if use_cuda and isinstance(optimizer, torch.optim.Optimizer):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(data)
                    
                    # Compute loss using configured loss function
                    total_scaled_loss, loss_on_w, loss_off = loss_fn(
                        output=output,
                        label=target
                    )
                    
                    # Add L2 regularization for encoder and decoder
                    encoder_l2 = torch.sum(model.fc1.weight ** 2)
                    decoder_l2 = torch.sum(model.fc2.weight ** 2)
                    
                    # Total loss = scaled loss + regularization
                    loss = (total_scaled_loss 
                           + config.training.encoder_regularization * encoder_l2 
                           + config.training.decoder_regularization * decoder_l2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision for CPU or SGLD optimizer
                output = model(data)
                
                # Compute loss using configured loss function
                total_scaled_loss, loss_on_w, loss_off = loss_fn(
                    output=output,
                    label=target
                )
                
                # Add L2 regularization for encoder and decoder
                encoder_l2 = torch.sum(model.fc1.weight ** 2)
                decoder_l2 = torch.sum(model.fc2.weight ** 2)
                
                # Total loss = scaled loss + regularization
                loss = (total_scaled_loss 
                       + config.training.encoder_regularization * encoder_l2 
                       + config.training.decoder_regularization * decoder_l2)
                
                loss.backward()
                optimizer.step()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                global_step += 1
            
            total_train_loss += loss.item()
            total_train_scaled_loss += total_scaled_loss.item()
            total_train_loss_on_w += loss_on_w.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr'] if isinstance(optimizer, torch.optim.Optimizer) else config.training.learning_rate
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-Loss: {loss_on_w.item():.6f}, Off-Loss: {loss_off.item():.6f}, LR: {current_lr:.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_scaled_loss = total_train_scaled_loss / len(train_loader)   
        avg_loss_on_w = total_train_loss_on_w / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        test_scaled_loss = test_metrics["scaled_loss"]
        
        print(f'Epoch {epoch+1}/{config.training.epochs}: Train Loss: {avg_loss:.6f}, Train Scaled Loss: {avg_scaled_loss:.6f}, Train On-Loss: {avg_loss_on_w:.6f}, Train Off-Loss: {avg_loss_off:.6f}, Test Scaled Loss: {test_scaled_loss:.6f}')
        
        # Log epoch to tracker
        if tracker is not None:
            train_metrics = {
                "loss": avg_loss,
                "scaled_loss": avg_scaled_loss,
                "loss_on": avg_loss_on_w,
                "loss_off": avg_loss_off,
            }
            tracker.log_epoch(epoch + 1, train_metrics, test_metrics, model=model)
            
            # Save encoder-decoder weight pairs data or plot directly at each epoch
            if config.save_plots:
                if config.plot_during_training:
                    # Plot directly during training (allows viewing plots while training)
                    pairs_plot_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{epoch+1:04d}.png')
                    plot_encoder_decoder_pairs(model=model, save_path=pairs_plot_path)
                else:
                    # Save data for later plotting (faster, no matplotlib blocking)
                    pairs_data_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{epoch+1:04d}.npz')
                    save_encoder_decoder_pairs(model=model, save_path=pairs_data_path)
    
    return model


def train_stage2_decoder_only(
    model,
    train_loader,
    test_loader,
    config: ExperimentConfig,
    tracker: Optional[ExperimentTracker] = None,
) -> Tuple[torch.nn.Module, float]:
    """
    Train only the decoder (unembedding) layer with frozen encoder.
    
    Similar to train() but:
    - Only optimizes fc2 (decoder) parameters
    - Encoder (fc1) is frozen (requires_grad=False)
    - Only applies decoder regularization (encoder reg = 0)
    - Uses stage2_epochs if specified, else config.training.epochs
    
    Args:
        model: TwoLayerNet model with frozen encoder
        train_loader: Training data loader
        test_loader: Test data loader
        config: Experiment configuration
        tracker: Optional experiment tracker for logging
    
    Returns:
        Trained model (decoder only)
    """
    # Validate device - fall back to CPU if CUDA is requested but not available
    device_str = config.data.device
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        use_cuda = False
    
    device = torch.device(device_str)
    
    # Move model to CUDA if available (using .cuda() for better performance)
    if use_cuda:
        model = model.cuda()
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
    else:
        model = model.to(device)
    
    # Ensure encoder is frozen
    model.fc1.weight.requires_grad = False
    if model.fc1.bias is not None:
        model.fc1.bias.requires_grad = False
    
    # Create optimizer - only optimize decoder (fc2) parameters
    decoder_params = [p for p in model.fc2.parameters() if p.requires_grad]
    
    if config.training.optimizer_type == "sgld":
        optimizer = SGLD(decoder_params, lr=config.training.learning_rate, 
                        temperature=config.training.temperature)
    elif config.training.optimizer_type == "adam":
        # Use AdamW with fused=True on CUDA for better performance
        if use_cuda:
            weight_decay = getattr(config.training, 'weight_decay', 0.0)
            optimizer = torch.optim.AdamW(
                decoder_params,
                lr=config.training.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
                fused=True
            )
        else:
            optimizer = torch.optim.Adam(decoder_params, lr=config.training.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {config.training.optimizer_type}")
    
    # Create loss function based on config
    loss_fn = create_loss_function(config.loss)
    
    # Create GradScaler for mixed precision training (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    
    # Create learning rate scheduler with warmup (only for standard optimizers, not SGLD)
    scheduler = None
    if config.training.enable_warmup and isinstance(optimizer, torch.optim.Optimizer):
        stage2_epochs = config.training.stage2_epochs if config.training.stage2_epochs is not None else config.training.epochs
        total_steps = stage2_epochs * len(train_loader)
        warmup_steps = int(config.training.warmup_fraction * total_steps)
        
        def lr_fn(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
        print(f"Stage 2 learning rate scheduler enabled: {warmup_steps} warmup steps, {total_steps} total steps")
    
    # Determine epochs for stage 2
    stage2_epochs = config.training.stage2_epochs if config.training.stage2_epochs is not None else config.training.epochs
    
    print(f"Starting Stage 2 training (decoder only) for {stage2_epochs} epochs...")
    
    best_test_loss = float('inf')
    
    # Track global step for scheduler
    global_step = 0
    
    for epoch in range(stage2_epochs):
        model.train()
        total_train_loss = 0
        total_train_scaled_loss = 0
        total_train_loss_on_w = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Use non_blocking=True for CUDA data transfer
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            # Use set_to_none=True for zero_grad when available
            if hasattr(optimizer, 'zero_grad'):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            # Mixed precision training for CUDA (only for standard optimizers, not SGLD)
            if use_cuda and isinstance(optimizer, torch.optim.Optimizer):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(data)
                    
                    # Compute loss using configured loss function
                    total_scaled_loss, loss_on_w, loss_off = loss_fn(
                        output=output,
                        label=target
                    )
                    
                    # Only decoder regularization (encoder is frozen, so encoder reg = 0)
                    decoder_l2 = torch.sum(model.fc2.weight ** 2)
                    
                    # Total loss = scaled loss + decoder regularization only
                    loss = (total_scaled_loss 
                           + config.training.decoder_regularization * decoder_l2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision for CPU or SGLD optimizer
                output = model(data)
                
                # Compute loss using configured loss function
                total_scaled_loss, loss_on_w, loss_off = loss_fn(
                    output=output,
                    label=target
                )
                
                # Only decoder regularization (encoder is frozen, so encoder reg = 0)
                decoder_l2 = torch.sum(model.fc2.weight ** 2)
                
                # Total loss = scaled loss + decoder regularization only
                loss = (total_scaled_loss 
                       + config.training.decoder_regularization * decoder_l2)
                
                loss.backward()
                optimizer.step()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                global_step += 1
            
            total_train_loss += loss.item()
            total_train_scaled_loss += total_scaled_loss.item()
            total_train_loss_on_w += loss_on_w.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr'] if isinstance(optimizer, torch.optim.Optimizer) else config.training.learning_rate
                print(f'Stage 2 - Epoch: {epoch+1}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-Loss: {loss_on_w.item():.6f}, Off-Loss: {loss_off.item():.6f}, LR: {current_lr:.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_scaled_loss = total_train_scaled_loss / len(train_loader)   
        avg_loss_on_w = total_train_loss_on_w / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        test_scaled_loss = test_metrics["scaled_loss"]
        
        # Track best test loss
        if test_scaled_loss < best_test_loss:
            best_test_loss = test_scaled_loss
        
        print(f'Stage 2 - Epoch {epoch+1}/{stage2_epochs}: Train Loss: {avg_loss:.6f}, Train Scaled Loss: {avg_scaled_loss:.6f}, Train On-Loss: {avg_loss_on_w:.6f}, Train Off-Loss: {avg_loss_off:.6f}, Test Scaled Loss: {test_scaled_loss:.6f}')
        
        # Log epoch to tracker (if provided)
        if tracker is not None:
            train_metrics = {
                "loss": avg_loss,
                "scaled_loss": avg_scaled_loss,
                "loss_on": avg_loss_on_w,
                "loss_off": avg_loss_off,
            }
            tracker.log_epoch(epoch + 1 + config.training.epochs, train_metrics, test_metrics)  # Offset by stage 1 epochs
            
            # Save encoder-decoder weight pairs data or plot directly at each stage 2 epoch
            if config.save_plots:
                epoch_num = epoch + 1 + config.training.epochs  # Offset by stage 1 epochs
                if config.plot_during_training:
                    # Plot directly during training (allows viewing plots while training)
                    pairs_plot_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{epoch_num:04d}.png')
                    plot_encoder_decoder_pairs(model=model, save_path=pairs_plot_path)
                else:
                    # Save data for later plotting (faster, no matplotlib blocking)
                    pairs_data_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{epoch_num:04d}.npz')
                    save_encoder_decoder_pairs(model=model, save_path=pairs_data_path)
    
    return model, best_test_loss


def evaluate(model, test_loader, loss_fn, device):
    """Evaluate model and return metrics"""
    model.eval()
    test_total_scaled_loss = 0
    test_loss_on = 0
    test_loss_off = 0
    
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for data, target in test_loader:
            # Use non_blocking=True for CUDA data transfer
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            # Use mixed precision for evaluation on CUDA
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(data)
                    total_scaled_loss, loss_on_w, loss_off = loss_fn(
                        output=output,
                        label=target
                    )
            else:
                output = model(data)
                total_scaled_loss, loss_on_w, loss_off = loss_fn(
                    output=output,
                    label=target
                )
            
            test_total_scaled_loss += total_scaled_loss.item()
            test_loss_on += loss_on_w.item()
            test_loss_off += loss_off.item()
    
    n_batches = len(test_loader)
    return {
        "scaled_loss": test_total_scaled_loss / n_batches,
        "loss_on": test_loss_on / n_batches,
        "loss_off": test_loss_off / n_batches,
    }


def plot_encoder_weights_histogram(model, bins=50, figsize=(10, 6), save_path=None):
    """
    Plot histogram of encoder (first layer) weights.
    
    Args:
        model: TwoLayerNet instance
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Extract encoder weights (fc1.weight has shape: hidden_size x input_size)
    encoder_weights = model.fc1.weight.data.cpu().numpy().flatten()
    
    # Compute statistics
    mean = encoder_weights.mean()
    std = encoder_weights.std()
    median = np.median(encoder_weights)
    min_val = encoder_weights.min()
    max_val = encoder_weights.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(encoder_weights, bins=bins, density=True, 
                                      alpha=0.7, color='steelblue', edgecolor='black')
    
    # Check if there are weights with absolute value > 0.25 and adjust y-axis if needed
    threshold = 0.25
    large_weights_mask = np.abs(encoder_weights) > threshold
    ylim_adjusted = False
    
    if np.any(large_weights_mask):
        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Find bins corresponding to large weights
        large_weight_bins_mask = np.abs(bin_centers) > threshold
        
        if np.any(large_weight_bins_mask):
            # Find maximum density among large weight bins
            max_density_large = np.max(n[large_weight_bins_mask])
            
            # Set y-axis limit to focus on larger weights (with 10% margin)
            if max_density_large > 0:
                ax.set_ylim(0, max_density_large * 1.1)
                ylim_adjusted = True
    
    # Add vertical lines for mean and median
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    title = 'Distribution of Encoder Weights (fc1)'
    if ylim_adjusted:
        title += '\n(Weights near 0 truncated for clarity)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add interval count labels below x-axis
    interval_size = 0.2
    min_val_range = np.floor(min_val / interval_size) * interval_size
    max_val_range = np.ceil(max_val / interval_size) * interval_size
    intervals = np.arange(min_val_range, max_val_range + interval_size, interval_size)
    
    # Helper function to format count with 1-2 significant figures
    def format_count(n):
        if n < 100:
            return str(n)
        elif n < 1000:
            return str(round(n, -1))  # Round to nearest 10
        elif n < 10000:
            return f"{n/1000:.1f}k".rstrip('0').rstrip('.')
        else:
            return f"{n/1000:.0f}k"
    
    for i in range(len(intervals) - 1):
        interval_start = intervals[i]
        interval_end = intervals[i + 1]
        # Count values in this interval
        count = np.sum((encoder_weights >= interval_start) & (encoder_weights < interval_end))
        # Handle the last interval to include the endpoint
        if i == len(intervals) - 2:
            count = np.sum((encoder_weights >= interval_start) & (encoder_weights <= interval_end))
        
        if count > 0:
            interval_center = (interval_start + interval_end) / 2
            # Position label below x-axis (at y = -3% of y-range)
            y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            ax.text(interval_center, y_pos, format_count(count), 
                   ha='center', va='top', fontsize=8, 
                   color='gray', rotation=0)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean:.4f}\n'
    stats_text += f'Std: {std:.4f}\n'
    stats_text += f'Median: {median:.4f}\n'
    stats_text += f'Min: {min_val:.4f}\n'
    stats_text += f'Max: {max_val:.4f}\n'
    stats_text += f'Total weights: {len(encoder_weights):,}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax

def plot_network_outputs_histogram(model, data_loader=None, n_samples=1, bins=50, 
                                   figsize=(10, 6), save_path=None):
    """
    Plot histogram of neural network output values.
    
    Args:
        model: TwoLayerNet instance
        data_loader: DataLoader to sample from (if None, uses random inputs)
        n_samples: Number of samples to generate
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    model.eval()
    device = next(model.parameters()).device
    
    outputs_list = []
    
    with torch.no_grad():
        if data_loader is not None:
            # Use data from data_loader
            samples_collected = 0
            for data, _ in data_loader:
                data = data.to(device)
                output = model(data)
                outputs_list.append(output.cpu().numpy())
                samples_collected += data.size(0)
                if samples_collected >= n_samples:
                    break
        else:
            # Generate random inputs
            input_size = model.fc1.in_features
            # Generate in batches for efficiency
            batch_size = 128
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for _ in range(n_batches):
                current_batch_size = min(batch_size, n_samples - len(outputs_list) * batch_size)
                random_input = torch.randn(current_batch_size, input_size, device=device)
                output = model(random_input)
                outputs_list.append(output.cpu().numpy())
    
    # Concatenate all outputs and flatten
    all_outputs = np.concatenate(outputs_list, axis=0).flatten()
    
    # Limit to n_samples if we collected more
    if len(all_outputs) > n_samples * model.fc2.out_features:
        all_outputs = all_outputs[:n_samples * model.fc2.out_features]
    
    # Compute statistics
    mean = all_outputs.mean()
    std = all_outputs.std()
    median = np.median(all_outputs)
    min_val = all_outputs.min()
    max_val = all_outputs.max()
    
    # Count values in different ranges (for sparse prediction analysis)
    near_zero = np.sum((all_outputs >= -0.1) & (all_outputs <= 0.1))
    near_one = np.sum((all_outputs >= 0.9) & (all_outputs <= 1.1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(all_outputs, bins=bins, density=True, 
                                      alpha=0.7, color='coral', edgecolor='black')
    
    # Check if there are outputs with absolute value > 0.25 and adjust y-axis if needed
    threshold = 0.25
    large_outputs_mask = np.abs(all_outputs) > threshold
    ylim_adjusted = False
    
    if np.any(large_outputs_mask):
        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Find bins corresponding to large outputs
        large_output_bins_mask = np.abs(bin_centers) > threshold
        
        if np.any(large_output_bins_mask):
            # Find maximum density among large output bins
            max_density_large = np.max(n[large_output_bins_mask])
            
            # Set y-axis limit to focus on larger outputs (with 10% margin)
            if max_density_large > 0:
                ax.set_ylim(0, max_density_large * 1.1)
                ylim_adjusted = True
    
    # Add vertical lines for mean, median, 0, and 1
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    ax.axvline(0, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Target: 0')
    ax.axvline(1, color='purple', linestyle=':', linewidth=2, alpha=0.5, label='Target: 1')
    
    # Add labels and title
    ax.set_xlabel('Output Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    title = 'Distribution of Neural Network Outputs'
    if ylim_adjusted:
        title += '\n(Outputs near 0 truncated for clarity)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add interval count labels below x-axis
    interval_size = 0.2
    min_val_range = np.floor(min_val / interval_size) * interval_size
    max_val_range = np.ceil(max_val / interval_size) * interval_size
    intervals = np.arange(min_val_range, max_val_range + interval_size, interval_size)
    
    # Helper function to format count with 1-2 significant figures
    def format_count(n):
        if n < 100:
            return str(n)
        elif n < 1000:
            return str(round(n, -1))  # Round to nearest 10
        elif n < 10000:
            return f"{n/1000:.1f}k".rstrip('0').rstrip('.')
        else:
            return f"{n/1000:.0f}k"
    
    for i in range(len(intervals) - 1):
        interval_start = intervals[i]
        interval_end = intervals[i + 1]
        # Count values in this interval
        count = np.sum((all_outputs >= interval_start) & (all_outputs < interval_end))
        # Handle the last interval to include the endpoint
        if i == len(intervals) - 2:
            count = np.sum((all_outputs >= interval_start) & (all_outputs <= interval_end))
        
        if count > 0:
            interval_center = (interval_start + interval_end) / 2
            # Position label below x-axis (at y = -3% of y-range)
            y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            ax.text(interval_center, y_pos, format_count(count), 
                   ha='center', va='top', fontsize=8, 
                   color='gray', rotation=0)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean:.4f}\n'
    stats_text += f'Std: {std:.4f}\n'
    stats_text += f'Median: {median:.4f}\n'
    stats_text += f'Min: {min_val:.4f}\n'
    stats_text += f'Max: {max_val:.4f}\n'
    stats_text += f'Total values: {len(all_outputs):,}\n'
    stats_text += f'Near 0 (±0.1): {100*near_zero/len(all_outputs):.1f}%\n'
    stats_text += f'Near 1 (±0.1): {100*near_one/len(all_outputs):.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax

def save_encoder_decoder_pairs(model, save_path):
    """
    Save encoder-decoder weight pairs to disk for later plotting.
    
    Pairs encoder[i,j] with decoder[j,i] representing the same path:
    input j -> hidden i -> output j
    
    Args:
        model: TwoLayerNet instance
        save_path: Path to save the .npz file (will replace .png with .npz if needed)
    """
    # Extract encoder and decoder weights
    encoder_weights = model.fc1.weight.data.cpu().numpy()  # (hidden_size, input_size)
    decoder_weights = model.fc2.weight.data.cpu().numpy()  # (input_size, hidden_size)
    
    hidden_size, input_size = encoder_weights.shape
    
    # Pair weights: encoder[i, j] with decoder[j, i]
    encoder_pairs = []
    decoder_pairs = []
    
    for i in range(hidden_size):
        for j in range(input_size):
            encoder_pairs.append(encoder_weights[i, j])
            decoder_pairs.append(decoder_weights[j, i])
    
    encoder_pairs = np.array(encoder_pairs)
    decoder_pairs = np.array(decoder_pairs)
    
    # Save to numpy format
    # Convert PosixPath to string if needed
    save_path = str(save_path)
    if save_path.endswith('.png'):
        save_path = save_path.replace('.png', '.npz')
    elif not save_path.endswith('.npz'):
        save_path = save_path + '.npz'
    
    np.savez(save_path, encoder_pairs=encoder_pairs, decoder_pairs=decoder_pairs,
             hidden_size=hidden_size, input_size=input_size)
    
    return save_path


def plot_encoder_decoder_pairs(model=None, figsize=(10, 8), save_path=None, data_path=None, 
                                colored_pairs=None, colored_colors=None):
    """
    Plot 2D scatter plot of encoder-decoder weight pairs.
    
    Can plot from a model directly or from saved .npz data file.
    
    Args:
        model: TwoLayerNet instance (if provided, extracts weights from model)
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        data_path: Path to .npz file with saved weight pairs (alternative to model)
        colored_pairs: Optional list of (i, j) tuples to highlight with distinct colors
        colored_colors: Optional list of colors for colored_pairs (defaults to tab20 colormap)
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Load data from file or extract from model
    encoder_weights_full = None
    decoder_weights_full = None
    
    if data_path is not None:
        data = np.load(data_path)
        encoder_pairs = data['encoder_pairs']
        decoder_pairs = data['decoder_pairs']
        # For colored pairs, we need the full weight matrices, but they're not in .npz
        # So colored pairs won't work with data_path (would need to save full matrices)
    elif model is not None:
        # Extract encoder and decoder weights
        encoder_weights_full = model.fc1.weight.data.cpu().numpy()  # (hidden_size, input_size)
        decoder_weights_full = model.fc2.weight.data.cpu().numpy()  # (input_size, hidden_size)
        
        hidden_size, input_size = encoder_weights_full.shape
        
        # Pair weights: encoder[i, j] with decoder[j, i]
        # Vectorized: encoder[i,j] in row-major order pairs with decoder[j,i] via transpose
        encoder_pairs = encoder_weights_full.flatten()  # Row-major: [enc[0,0], enc[0,1], ..., enc[0,d-1], enc[1,0], ...]
        decoder_pairs = decoder_weights_full.T.flatten()  # Transpose then flatten: [dec[0,0], dec[1,0], ..., dec[d-1,0], dec[0,1], ...]
    else:
        raise ValueError("Either model or data_path must be provided")
    
    # Compute axis limits from all pairs first (before plotting colored points)
    xlim = (encoder_pairs.min(), encoder_pairs.max())
    ylim = (decoder_pairs.min(), decoder_pairs.max())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot (each point as a pixel) - background
    ax.scatter(encoder_pairs, decoder_pairs, s=0.5, alpha=0.3, color='blue', rasterized=True, zorder=1)
    
    # Plot colored points on top if provided
    if colored_pairs is not None and encoder_weights_full is not None and decoder_weights_full is not None:
        # Generate colors if not provided
        if colored_colors is None:
            cmap = plt.cm.tab20
            colored_colors = [cmap(i % 20) for i in range(len(colored_pairs))]
        
        # Extract values for colored pairs
        colored_encoder = []
        colored_decoder = []
        for i, j in colored_pairs:
            colored_encoder.append(encoder_weights_full[i, j])
            colored_decoder.append(decoder_weights_full[j, i])
        
        # Plot colored points with distinct colors, larger size, higher zorder
        for idx, (enc_val, dec_val, color) in enumerate(zip(colored_encoder, colored_decoder, colored_colors)):
            ax.scatter([enc_val], [dec_val], s=50, color=color, edgecolors='black', 
                      linewidths=1, zorder=10, alpha=0.8, label=f'Point {idx+1}' if idx < 10 else None)
    
    # Set axis limits (computed from all pairs, not affected by colored points)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Add labels and title
    ax.set_xlabel('Encoder Weight', fontsize=12)
    ax.set_ylabel('Decoder Weight', fontsize=12)
    ax.set_title('Encoder-Decoder Weight Pairs\n(encoder[i,j] vs decoder[j,i])', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Total pairs: {len(encoder_pairs):,}\n'
    stats_text += f'Encoder range: [{encoder_pairs.min():.4f}, {encoder_pairs.max():.4f}]\n'
    stats_text += f'Decoder range: [{decoder_pairs.min():.4f}, {decoder_pairs.max():.4f}]\n'
    stats_text += f'Encoder mean: {encoder_pairs.mean():.4f}\n'
    stats_text += f'Decoder mean: {decoder_pairs.mean():.4f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def generate_plots_from_saved_data(experiment_dir: Path, n_jobs: Optional[int] = None):
    """
    Generate plots from all saved .npz files in the experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory containing .npz files
        n_jobs: Number of parallel jobs (default: cpu_count())
    """
    # Find all epoch .npz files
    pattern = "encoder_decoder_pairs_epoch_*.npz"
    epoch_files = sorted(experiment_dir.glob(pattern))
    
    if not epoch_files:
        print("No encoder_decoder_pairs_epoch_*.npz files found to plot.")
        return
    
    print(f"\nGenerating plots from {len(epoch_files)} saved epoch files...")
    
    # Prepare arguments: (data_path, output_path)
    plot_args = []
    for data_file in epoch_files:
        # Convert .npz to .png
        output_file = data_file.with_suffix('.png')
        plot_args.append((str(data_file), str(output_file)))
    
    # Determine number of parallel jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    n_jobs = min(n_jobs, len(plot_args))  # Don't use more jobs than files
    
    # Helper function for multiprocessing
    def plot_single_file(args):
        data_path, output_path = args
        try:
            plot_encoder_decoder_pairs(data_path=data_path, save_path=output_path)
            return f"Generated {Path(output_path).name}"
        except Exception as e:
            return f"Error plotting {Path(output_path).name}: {e}"
    
    # Generate plots in parallel
    if n_jobs > 1 and len(plot_args) > 1:
        print(f"Using {n_jobs} parallel workers...")
        with Pool(n_jobs) as pool:
            results = pool.map(plot_single_file, plot_args)
        for result in results:
            print(f"  {result}")
    else:
        # Generate plots sequentially
        for data_path, output_path in plot_args:
            result = plot_single_file((data_path, output_path))
            print(f"  {result}")
    
    print(f"Generated {len(plot_args)} plot(s) from saved data.")


def main():
    # Load config from JSON file if provided, otherwise use defaults
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        config = ExperimentConfig.load_json(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        # Default configuration (can be used if no config file is provided)
        config = ExperimentConfig(
            model=ModelConfig(
                hidden_size=128,
                encoder_initialization_scale=1.0,
                decoder_initialization_scale=1.0,
            ),
            training=TrainingConfig(
                optimizer_type="sgld",  # or "adam"
                learning_rate=1e-4,
                temperature=0.0,  # Only used for SGLD
                epochs=1,
                batch_size=128,
                encoder_regularization=0.0,
                decoder_regularization=0.0,
            ),
            loss=LossConfig(
                loss_type="scaled_mse",
                lambda_on=10.0,
            ),
            data=DataConfig(
                d=32,
                sparsity=2,
                noise_variance=0.1,
                n_train=1000000,
                n_val=1000,
                seed=42,
            ),
            experiment_name="experiment",
            output_dir=None,  # Auto-generate
            save_model=True,
            save_plots=True,
        )
        print("Using default configuration (provide a JSON config file as argument to use custom config)")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config.data, batch_size=config.training.batch_size)
    
    # Create model
    model = TwoLayerNet(
        input_size=config.data.d,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=config.model.encoder_initialization_scale,
        decoder_initialization_scale=config.model.decoder_initialization_scale,
    )
    
    # Validate device before training
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        config.data.device = "cpu"  # Update config for consistency
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    print(f"Training with {config.training.optimizer_type.upper()} on device: {device_str}")
    print(f"Learning rate: {config.training.learning_rate}, "
          f"Temperature: {config.training.temperature}, "
          f"Encoder regularization: {config.training.encoder_regularization}, "
          f"Decoder regularization: {config.training.decoder_regularization}")
    
    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )
    
    print("\nTraining complete!")
    
    # Save plots
    if config.save_plots:
        encoder_plot_path = tracker.get_plot_path('encoder_weights_histogram.png')
        output_plot_path = tracker.get_plot_path('network_outputs_histogram.png')
        pairs_plot_path = tracker.get_plot_path('encoder_decoder_pairs.png')
        pairs_data_path = tracker.get_plot_path('encoder_decoder_pairs.npz')
        plot_encoder_weights_histogram(model=model, save_path=encoder_plot_path)
        plot_network_outputs_histogram(model=model, data_loader=test_loader, save_path=output_plot_path)
        # Save data for later plotting and also plot final model
        save_encoder_decoder_pairs(model=model, save_path=pairs_data_path)
        plot_encoder_decoder_pairs(model=model, save_path=pairs_plot_path)
        
        # Generate plots from all saved .npz epoch files if plot_during_training was False
        if not config.plot_during_training:
            generate_plots_from_saved_data(tracker.output_dir)
    
    # Stage 2 training: Frozen GMM encoder + decoder-only training
    stage2_clustering_params = None
    if config.training.two_stage_training:
        print("\n" + "=" * 80)
        print("Starting Stage 2: Frozen GMM Encoder + Decoder-Only Training")
        print("=" * 80)
        
        # Extract encoder weights
        encoder_weights = model.fc1.weight.data.cpu().numpy()
        encoder_shape = encoder_weights.shape
        
        print(f"Clustering encoder weights (shape: {encoder_shape}) into {config.training.num_clusters} Gaussian components...")
        
        # Cluster encoder weights
        gmm_params, mixture_probs = cluster_encoder_weights(
            encoder_weights, 
            n_clusters=config.training.num_clusters
        )
        
        # Store clustering parameters for logging
        stage2_clustering_params = {
            'gmm_params': gmm_params,
            'mixture_probs': mixture_probs.tolist(),
            'encoder_shape': list(encoder_shape)
        }
        
        # Print clustering results
        print("\nClustering results:")
        for i, (params, prob) in enumerate(zip(gmm_params, mixture_probs)):
            print(f"  Cluster {i+1}: mean={params['mean']:.6f}, variance={params['variance']:.6f}, "
                  f"mixture_prob={prob:.4f}")
        
        # Get device
        device = torch.device(device_str)
        
        # Evaluate initial (stage 1) model to get baseline loss
        loss_fn = create_loss_function(config.loss)
        initial_model_metrics = evaluate(model, test_loader, loss_fn, device)
        initial_test_loss = initial_model_metrics["scaled_loss"]
        print(f"\nInitial (Stage 1) model test loss: {initial_test_loss:.6f}")
        
        # Train multiple traces
        num_traces = config.training.num_traces
        print(f"\nTraining {num_traces} trace(s) with independently sampled frozen encoders...")
        
        trace_results = []
        best_model = None
        best_trace_loss = float('inf')
        
        for trace_idx in range(num_traces):
            print(f"\n--- Trace {trace_idx + 1}/{num_traces} ---")
            
            # Sample frozen encoder from GMM (each trace gets a new sample)
            frozen_encoder = sample_frozen_encoder(
                encoder_shape,
                gmm_params,
                mixture_probs,
                random_seed=config.data.seed + trace_idx  # Different seed for each trace
            )
            
            # Create new model with frozen encoder
            model_stage2 = TwoLayerNet(
                input_size=config.data.d,
                hidden_size=config.model.hidden_size,
                encoder_initialization_scale=1.0,
                decoder_initialization_scale=1.0,
            )
            
            # Move to device and compile if CUDA
            use_cuda_stage2 = device_str.startswith("cuda") and torch.cuda.is_available()
            if use_cuda_stage2:
                model_stage2 = model_stage2.cuda()
                # Compile model for better performance (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    model_stage2 = torch.compile(model_stage2)
            else:
                model_stage2 = model_stage2.to(device)
            
            # Set frozen encoder weights
            model_stage2.fc1.weight.data = frozen_encoder.to(device)
            model_stage2.fc1.weight.requires_grad = False
            if model_stage2.fc1.bias is not None:
                model_stage2.fc1.bias.requires_grad = False
            
            # Copy decoder weights from stage 1 (start from same decoder initialization)
            model_stage2.fc2.weight.data = model.fc2.weight.data.clone()
            if model_stage2.fc2.bias is not None and model.fc2.bias is not None:
                model_stage2.fc2.bias.data = model.fc2.bias.data.clone()
            
            # Optionally plot first trace's GMM-sampled encoder weights
            if config.save_plots and trace_idx == 0:
                original_plot_path = tracker.get_plot_path('encoder_weights_stage1.png')
                gmm_plot_path = tracker.get_plot_path('encoder_weights_gmm_frozen_trace1.png')
                plot_encoder_weights_histogram(model, save_path=original_plot_path)
                plot_encoder_weights_histogram(model_stage2, save_path=gmm_plot_path)
                print(f"Saved encoder weight histograms: original and GMM-sampled (trace 1)")
            
            # Train decoder only (stage 2)
            model_stage2, best_test_loss = train_stage2_decoder_only(
                model=model_stage2,
                train_loader=train_loader,
                test_loader=test_loader,
                config=config,
                tracker=tracker,
            )
            
            # Store trace results
            loss_diff = best_test_loss - initial_test_loss
            trace_results.append({
                'trace': trace_idx + 1,
                'best_test_loss': best_test_loss,
                'loss_difference': loss_diff
            })
            
            print(f"Trace {trace_idx + 1} - Best test loss: {best_test_loss:.6f}, "
                  f"Difference from initial: {loss_diff:+.6f}")
            
            # Track best trace
            if best_test_loss < best_trace_loss:
                best_trace_loss = best_test_loss
                best_model = model_stage2
        
        # Use best model for final plots/results
        if best_model is not None:
            model = best_model
        
        # Store trace results in final metrics
        stage2_clustering_params['trace_results'] = trace_results
        stage2_clustering_params['initial_test_loss'] = initial_test_loss
        stage2_clustering_params['best_trace_loss'] = best_trace_loss
        stage2_clustering_params['num_traces'] = num_traces
        
        print("\n" + "=" * 80)
        print("Stage 2 training complete!")
        print(f"Initial model test loss: {initial_test_loss:.6f}")
        print(f"Best trace test loss: {best_trace_loss:.6f}")
        print(f"Best improvement: {best_trace_loss - initial_test_loss:+.6f}")
        print("=" * 80)
        
        # Generate plots from all saved .npz epoch files if plot_during_training was False
        if config.save_plots and not config.plot_during_training:
            generate_plots_from_saved_data(tracker.output_dir)
    
    # Save final results
    final_metrics = {}
    if stage2_clustering_params is not None:
        final_metrics['stage2_clustering'] = stage2_clustering_params
    
    model_state = model.state_dict() if config.save_model else None
    tracker.save_results(final_metrics=final_metrics, model_state=model_state)

if __name__ == "__main__":
    main()