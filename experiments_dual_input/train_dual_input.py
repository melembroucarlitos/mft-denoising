import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mft_denoising.data import create_dual_input_dataloaders, DualInputDataConfig
from mft_denoising.nn import TwoLayerNet
from mft_denoising.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
)
from mft_denoising.losses import create_loss_function
from mft_denoising.optimizers import create_optimizer
from mft_denoising.experiment import ExperimentTracker


def plot_dual_input_encoder_decoder_pairs(
    model,
    d_1: int,
    d_2: int,
    save_path: Optional[Path] = None,
    figsize=(15, 5)
):
    """
    Plot encoder-decoder weight pairs for dual input types.
    Creates three plots: Type 1 (blue), Type 2 (red), and Overlay (both).
    
    Args:
        model: TwoLayerNet instance
        d_1: Dimension of input type 1
        d_2: Dimension of input type 2
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    # Extract weights
    encoder_weights = model.fc1.weight.data.cpu().numpy()  # (hidden_size, d_1 + d_2)
    decoder_weights = model.fc2.weight.data.cpu().numpy()  # (d_1 + d_2, hidden_size)
    
    hidden_size = encoder_weights.shape[0]
    
    # Split by type
    enc_1 = encoder_weights[:, :d_1]  # (hidden_size, d_1)
    enc_2 = encoder_weights[:, d_1:]  # (hidden_size, d_2)
    dec_1 = decoder_weights[:d_1, :]  # (d_1, hidden_size)
    dec_2 = decoder_weights[d_1:, :]  # (d_2, hidden_size)
    
    # Create weight pairs for each type (vectorized)
    # Type 1: encoder[i, j] pairs with decoder[j, i]
    encoder_pairs_1 = enc_1.flatten()  # Row-major flatten: [enc_1[0,0], enc_1[0,1], ..., enc_1[0,d_1-1], enc_1[1,0], ...]
    decoder_pairs_1 = dec_1.T.flatten()  # Transpose then flatten: [dec_1[0,0], dec_1[1,0], ..., dec_1[d_1-1,0], dec_1[0,1], ...]
    
    # Type 2: encoder[i, j] pairs with decoder[j, i]
    encoder_pairs_2 = enc_2.flatten()  # Row-major flatten
    decoder_pairs_2 = dec_2.T.flatten()  # Transpose then flatten
    
    # Create three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Type 1 (blue)
    axes[0].scatter(encoder_pairs_1, decoder_pairs_1, s=0.5, alpha=0.3, color='blue', rasterized=True)
    axes[0].set_xlabel('Encoder Weight', fontsize=12)
    axes[0].set_ylabel('Decoder Weight', fontsize=12)
    axes[0].set_title('Type 1 (Blue)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Type 2 (red)
    axes[1].scatter(encoder_pairs_2, decoder_pairs_2, s=0.5, alpha=0.3, color='red', rasterized=True)
    axes[1].set_xlabel('Encoder Weight', fontsize=12)
    axes[1].set_ylabel('Decoder Weight', fontsize=12)
    axes[1].set_title('Type 2 (Red)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Overlay
    # Compute axis limits from both types
    all_enc = np.concatenate([encoder_pairs_1, encoder_pairs_2])
    all_dec = np.concatenate([decoder_pairs_1, decoder_pairs_2])
    xlim = (all_enc.min(), all_enc.max())
    ylim = (all_dec.min(), all_dec.max())
    
    axes[2].scatter(encoder_pairs_1, decoder_pairs_1, s=0.5, alpha=0.3, color='blue', 
                   label='Type 1', rasterized=True)
    axes[2].scatter(encoder_pairs_2, decoder_pairs_2, s=0.5, alpha=0.3, color='red', 
                   label='Type 2', rasterized=True)
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    axes[2].set_xlabel('Encoder Weight', fontsize=12)
    axes[2].set_ylabel('Decoder Weight', fontsize=12)
    axes[2].set_title('Overlay (Type 1 + Type 2)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, axes


def evaluate(model, test_loader, loss_fn, device, d_1: int):
    """Evaluate model and return metrics"""
    model.eval()
    test_total_loss = 0
    test_loss_on_1 = 0
    test_loss_on_2 = 0
    test_loss_off = 0
    
    use_cuda = device.type == "cuda"
    
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(data)
                    total_loss, loss_on_1, loss_on_2, loss_off = loss_fn(
                        output=output,
                        label=target,
                        d_1=d_1
                    )
            else:
                output = model(data)
                total_loss, loss_on_1, loss_on_2, loss_off = loss_fn(
                    output=output,
                    label=target,
                    d_1=d_1
                )
            
            test_total_loss += total_loss.item()
            test_loss_on_1 += loss_on_1.item()
            test_loss_on_2 += loss_on_2.item()
            test_loss_off += loss_off.item()
    
    n_batches = len(test_loader)
    return {
        "loss": test_total_loss / n_batches,
        "loss_on_1": test_loss_on_1 / n_batches,
        "loss_on_2": test_loss_on_2 / n_batches,
        "loss_off": test_loss_off / n_batches,
    }


def train(
    model,
    train_loader,
    test_loader,
    config: ExperimentConfig,
    tracker: Optional[ExperimentTracker] = None,
    d_1: int = None,
):
    """
    Train model using configured optimizer with dual lambda loss function.
    
    Args:
        model: TwoLayerNet model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Experiment configuration
        tracker: Optional experiment tracker for logging
        d_1: Dimension of input type 1 (required for dual lambda loss)
    
    Returns:
        Trained model
    """
    if d_1 is None:
        raise ValueError("d_1 must be provided for dual input training")
    
    # Validate device
    device_str = config.data.device if hasattr(config.data, 'device') else "cpu"
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        use_cuda = False
    
    device = torch.device(device_str)
    
    # Move model to device
    if use_cuda:
        model = model.cuda()
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
    else:
        model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config.training, use_cuda=use_cuda)
    
    # Create loss function
    loss_fn = create_loss_function(config.loss)
    
    # Create GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    
    # Create learning rate scheduler with warmup
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
        tracker.start(model=model)
    
    global_step = 0
    
    for epoch in range(config.training.epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_on_1 = 0
        total_train_loss_on_2 = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            if hasattr(optimizer, 'zero_grad'):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            # Mixed precision training for CUDA
            if use_cuda and isinstance(optimizer, torch.optim.Optimizer):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(data)
                    
                    total_loss, loss_on_1, loss_on_2, loss_off = loss_fn(
                        output=output,
                        label=target,
                        d_1=d_1
                    )
                    
                    # Add L2 regularization
                    encoder_l2 = torch.sum(model.fc1.weight ** 2)
                    decoder_l2 = torch.sum(model.fc2.weight ** 2)
                    
                    loss = (total_loss 
                           + config.training.encoder_regularization * encoder_l2 
                           + config.training.decoder_regularization * decoder_l2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                output = model(data)
                
                total_loss, loss_on_1, loss_on_2, loss_off = loss_fn(
                    output=output,
                    label=target,
                    d_1=d_1
                )
                
                # Add L2 regularization
                encoder_l2 = torch.sum(model.fc1.weight ** 2)
                decoder_l2 = torch.sum(model.fc2.weight ** 2)
                
                loss = (total_loss 
                       + config.training.encoder_regularization * encoder_l2 
                       + config.training.decoder_regularization * decoder_l2)
                
                loss.backward()
                optimizer.step()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                global_step += 1
            
            total_train_loss += loss.item()
            total_train_loss_on_1 += loss_on_1.item()
            total_train_loss_on_2 += loss_on_2.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr'] if isinstance(optimizer, torch.optim.Optimizer) else config.training.learning_rate
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-1: {loss_on_1.item():.6f}, On-2: {loss_on_2.item():.6f}, Off: {loss_off.item():.6f}, LR: {current_lr:.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_loss_on_1 = total_train_loss_on_1 / len(train_loader)
        avg_loss_on_2 = total_train_loss_on_2 / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device, d_1)
        
        print(f'Epoch {epoch+1}/{config.training.epochs}: Train Loss: {avg_loss:.6f}, '
              f'Train On-1: {avg_loss_on_1:.6f}, Train On-2: {avg_loss_on_2:.6f}, Train Off: {avg_loss_off:.6f}, '
              f'Test Loss: {test_metrics["loss"]:.6f}')
        
        # Log epoch to tracker
        if tracker is not None:
            train_metrics = {
                "loss": avg_loss,
                "loss_on_1": avg_loss_on_1,
                "loss_on_2": avg_loss_on_2,
                "loss_off": avg_loss_off,
            }
            tracker.log_epoch(epoch + 1, train_metrics, test_metrics, model=model)
            
            # Save plots at each epoch
            if config.save_plots:
                pairs_plot_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{epoch+1:04d}.png')
                plot_dual_input_encoder_decoder_pairs(
                    model=model,
                    d_1=d_1,
                    d_2=config.data.d_2,
                    save_path=pairs_plot_path
                )
    
    return model


def main():
    # Load config from JSON file if provided
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        # Load config manually to handle DualInputDataConfig
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create config objects
        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        loss_config = LossConfig(**config_dict["loss"])
        data_config = DualInputDataConfig(**config_dict["data"])
        
        config = ExperimentConfig(
            model=model_config,
            training=training_config,
            loss=loss_config,
            data=data_config,
            experiment_name=config_dict.get("experiment_name", "dual_input_experiment"),
            output_dir=config_dict.get("output_dir", None),
            save_model=config_dict.get("save_model", True),
            save_plots=config_dict.get("save_plots", True),
            plot_during_training=config_dict.get("plot_during_training", False),
        )
        print(f"Loaded config from: {config_path}")
    else:
        print("Error: Please provide a config file as argument")
        sys.exit(1)
    
    # Create dataloaders
    train_loader, test_loader = create_dual_input_dataloaders(
        config.data, 
        batch_size=config.training.batch_size
    )
    
    # Create model
    model = TwoLayerNet(
        input_size=config.data.d_total,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=config.model.encoder_initialization_scale,
        decoder_initialization_scale=config.model.decoder_initialization_scale,
    )
    
    # Validate device
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        config.data.device = "cpu"
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    print(f"Training with {config.training.optimizer_type.upper()} on device: {device_str}")
    print(f"Input dimensions: d_1={config.data.d_1}, d_2={config.data.d_2}, total={config.data.d_total}")
    print(f"Lambda values: lambda_1={config.loss.lambda_1}, lambda_2={config.loss.lambda_2}")
    
    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
        d_1=config.data.d_1,
    )
    
    print("\nTraining complete!")
    
    # Save final plots
    if config.save_plots:
        pairs_plot_path = tracker.get_plot_path('encoder_decoder_pairs.png')
        plot_dual_input_encoder_decoder_pairs(
            model=model,
            d_1=config.data.d_1,
            d_2=config.data.d_2,
            save_path=pairs_plot_path
        )
    
    # Save final results
    model_state = model.state_dict() if config.save_model else None
    tracker.save_results(final_metrics={}, model_state=model_state)


if __name__ == "__main__":
    main()
