import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from main import (
    evaluate,
    plot_encoder_decoder_pairs,
    save_encoder_decoder_pairs,
    plot_encoder_weights_histogram,
    plot_network_outputs_histogram,
)


def train_two_stage(
    model,
    train_loader,
    test_loader,
    config: ExperimentConfig,
    tracker: Optional[ExperimentTracker] = None,
):
    """
    Train model in two stages: SGLD for t_1 epochs, then Adam for t_2 epochs.
    
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
    
    # Move model to CUDA if available
    if use_cuda:
        model = model.cuda()
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
    else:
        model = model.to(device)
    
    # Get stage-specific hyperparameters
    sgld_epochs = config.training.sgld_epochs if config.training.sgld_epochs is not None else config.training.epochs
    adam_epochs = config.training.adam_epochs if config.training.adam_epochs is not None else config.training.epochs
    
    sgld_lr = config.training.sgld_learning_rate if config.training.sgld_learning_rate is not None else config.training.learning_rate
    sgld_temp = config.training.sgld_temperature if config.training.sgld_temperature is not None else config.training.temperature
    adam_lr = config.training.adam_learning_rate if config.training.adam_learning_rate is not None else sgld_lr
    
    print(f"Two-stage training configuration:")
    print(f"  Stage 1 (SGLD): {sgld_epochs} epochs, lr={sgld_lr}, temperature={sgld_temp}")
    print(f"  Stage 2 (Adam): {adam_epochs} epochs, lr={adam_lr}")
    
    # Create loss function (shared between stages)
    loss_fn = create_loss_function(config.loss)
    
    # Start tracking if provided
    if tracker is not None:
        tracker.start(model=model)
    
    # ========== STAGE 1: SGLD Training ==========
    print("\n" + "=" * 80)
    print("STAGE 1: SGLD Training")
    print("=" * 80)
    
    # Create SGLD optimizer
    sgld_optimizer = SGLD(model.parameters(), lr=sgld_lr, temperature=sgld_temp)
    
    # Create GradScaler (SGLD doesn't use mixed precision, but we'll keep it for consistency)
    scaler = None  # SGLD doesn't support mixed precision
    
    # Track global step for continuous epoch numbering
    global_epoch = 0
    
    for epoch in range(sgld_epochs):
        model.train()
        total_train_loss = 0
        total_train_scaled_loss = 0
        total_train_loss_on_w = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            sgld_optimizer.zero_grad()
            
            # Standard precision for SGLD
            output = model(data)
            
            # Compute loss
            total_scaled_loss, loss_on_w, loss_off = loss_fn(
                output=output,
                label=target
            )
            
            # Add L2 regularization
            encoder_l2 = torch.sum(model.fc1.weight ** 2)
            decoder_l2 = torch.sum(model.fc2.weight ** 2)
            
            loss = (total_scaled_loss 
                   + config.training.encoder_regularization * encoder_l2 
                   + config.training.decoder_regularization * decoder_l2)
            
            loss.backward()
            sgld_optimizer.step()
            
            total_train_loss += loss.item()
            total_train_scaled_loss += total_scaled_loss.item()
            total_train_loss_on_w += loss_on_w.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                print(f'Stage 1 - Epoch: {epoch+1}/{sgld_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-Loss: {loss_on_w.item():.6f}, Off-Loss: {loss_off.item():.6f}, LR: {sgld_lr:.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_scaled_loss = total_train_scaled_loss / len(train_loader)   
        avg_loss_on_w = total_train_loss_on_w / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        test_scaled_loss = test_metrics["scaled_loss"]
        
        global_epoch += 1
        print(f'Stage 1 - Epoch {epoch+1}/{sgld_epochs} (Global: {global_epoch}): Train Loss: {avg_loss:.6f}, Train Scaled Loss: {avg_scaled_loss:.6f}, Train On-Loss: {avg_loss_on_w:.6f}, Train Off-Loss: {avg_loss_off:.6f}, Test Scaled Loss: {test_scaled_loss:.6f}')
        
        # Log epoch to tracker
        if tracker is not None:
            train_metrics = {
                "loss": avg_loss,
                "scaled_loss": avg_scaled_loss,
                "loss_on": avg_loss_on_w,
                "loss_off": avg_loss_off,
            }
            tracker.log_epoch(global_epoch, train_metrics, test_metrics, model=model)
            
            # Save encoder-decoder weight pairs data or plot directly at each epoch
            if config.save_plots:
                # Always plot during training (user requested plotting every epoch)
                pairs_plot_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{global_epoch:04d}.png')
                plot_encoder_decoder_pairs(
                    model=model, 
                    save_path=pairs_plot_path,
                    colored_pairs=tracker.colored_pairs
                )
                # Also save data for later use
                pairs_data_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{global_epoch:04d}.npz')
                save_encoder_decoder_pairs(model=model, save_path=pairs_data_path)
    
    # ========== STAGE 2: Adam Training ==========
    print("\n" + "=" * 80)
    print("STAGE 2: Adam Training")
    print("=" * 80)
    
    # Create Adam optimizer
    if use_cuda:
        weight_decay = getattr(config.training, 'weight_decay', 0.0)
        adam_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adam_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            fused=True
        )
    else:
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    
    # Create GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    
    # Create learning rate scheduler with warmup (only for Adam)
    scheduler = None
    if config.training.enable_warmup:
        total_steps = adam_epochs * len(train_loader)
        warmup_steps = int(config.training.warmup_fraction * total_steps)
        
        def lr_fn(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(adam_optimizer, lr_lambda=lr_fn)
        print(f"Learning rate scheduler enabled: {warmup_steps} warmup steps, {total_steps} total steps")
    
    global_step = 0
    
    for epoch in range(adam_epochs):
        model.train()
        total_train_loss = 0
        total_train_scaled_loss = 0
        total_train_loss_on_w = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            
            if isinstance(adam_optimizer, torch.optim.Optimizer):
                adam_optimizer.zero_grad(set_to_none=True)
            else:
                adam_optimizer.zero_grad()
            
            # Mixed precision training for CUDA
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(data)
                    
                    total_scaled_loss, loss_on_w, loss_off = loss_fn(
                        output=output,
                        label=target
                    )
                    
                    encoder_l2 = torch.sum(model.fc1.weight ** 2)
                    decoder_l2 = torch.sum(model.fc2.weight ** 2)
                    
                    loss = (total_scaled_loss 
                           + config.training.encoder_regularization * encoder_l2 
                           + config.training.decoder_regularization * decoder_l2)
                
                scaler.scale(loss).backward()
                scaler.step(adam_optimizer)
                scaler.update()
            else:
                # Standard precision for CPU
                output = model(data)
                
                total_scaled_loss, loss_on_w, loss_off = loss_fn(
                    output=output,
                    label=target
                )
                
                encoder_l2 = torch.sum(model.fc1.weight ** 2)
                decoder_l2 = torch.sum(model.fc2.weight ** 2)
                
                loss = (total_scaled_loss 
                       + config.training.encoder_regularization * encoder_l2 
                       + config.training.decoder_regularization * decoder_l2)
                
                loss.backward()
                adam_optimizer.step()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                global_step += 1
            
            total_train_loss += loss.item()
            total_train_scaled_loss += total_scaled_loss.item()
            total_train_loss_on_w += loss_on_w.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                current_lr = adam_optimizer.param_groups[0]['lr']
                print(f'Stage 2 - Epoch: {epoch+1}/{adam_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-Loss: {loss_on_w.item():.6f}, Off-Loss: {loss_off.item():.6f}, LR: {current_lr:.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_scaled_loss = total_train_scaled_loss / len(train_loader)   
        avg_loss_on_w = total_train_loss_on_w / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        test_scaled_loss = test_metrics["scaled_loss"]
        
        global_epoch += 1
        print(f'Stage 2 - Epoch {epoch+1}/{adam_epochs} (Global: {global_epoch}): Train Loss: {avg_loss:.6f}, Train Scaled Loss: {avg_scaled_loss:.6f}, Train On-Loss: {avg_loss_on_w:.6f}, Train Off-Loss: {avg_loss_off:.6f}, Test Scaled Loss: {test_scaled_loss:.6f}')
        
        # Log epoch to tracker
        if tracker is not None:
            train_metrics = {
                "loss": avg_loss,
                "scaled_loss": avg_scaled_loss,
                "loss_on": avg_loss_on_w,
                "loss_off": avg_loss_off,
            }
            tracker.log_epoch(global_epoch, train_metrics, test_metrics, model=model)
            
            # Save encoder-decoder weight pairs data or plot directly at each epoch
            if config.save_plots:
                # Always plot during training (user requested plotting every epoch)
                pairs_plot_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{global_epoch:04d}.png')
                plot_encoder_decoder_pairs(
                    model=model, 
                    save_path=pairs_plot_path,
                    colored_pairs=tracker.colored_pairs
                )
                # Also save data for later use
                pairs_data_path = tracker.get_plot_path(f'encoder_decoder_pairs_epoch_{global_epoch:04d}.npz')
                save_encoder_decoder_pairs(model=model, save_path=pairs_data_path)
    
    return model


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
        # Default configuration
        config = ExperimentConfig(
            model=ModelConfig(
                hidden_size=128,
                encoder_initialization_scale=1.0,
                decoder_initialization_scale=1.0,
            ),
            training=TrainingConfig(
                optimizer_type="sgld",  # Not used in two-stage, but kept for compatibility
                learning_rate=1e-4,
                temperature=0.0,
                epochs=1,  # Not used in two-stage
                batch_size=128,
                encoder_regularization=0.0,
                decoder_regularization=0.0,
                # Two-stage specific parameters
                sgld_epochs=50,
                adam_epochs=50,
                sgld_learning_rate=1e-4,
                sgld_temperature=0.1,
                adam_learning_rate=1e-4,
                colored_points_count=20,
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
            experiment_name="sgld_adam_experiment",
            output_dir=None,
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
    
    # Load checkpoint if specified
    if config.training.resume_from_checkpoint is not None:
        checkpoint_path = Path(config.training.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint state dict
        checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle compiled model checkpoints (strip _orig_mod. prefix if present)
        if any(k.startswith('_orig_mod.') for k in checkpoint_state.keys()):
            new_state_dict = {}
            for k, v in checkpoint_state.items():
                if k.startswith('_orig_mod.'):
                    new_key = k[len('_orig_mod.'):]
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            checkpoint_state = new_state_dict
        
        # Verify architecture compatibility
        expected_fc1_shape = (config.model.hidden_size, config.data.d)
        expected_fc2_shape = (config.data.d, config.model.hidden_size)
        
        if 'fc1.weight' not in checkpoint_state or 'fc2.weight' not in checkpoint_state:
            raise ValueError(f"Checkpoint missing required layers (fc1.weight or fc2.weight). "
                           f"Found keys: {list(checkpoint_state.keys())}")
        
        actual_fc1_shape = checkpoint_state['fc1.weight'].shape
        actual_fc2_shape = checkpoint_state['fc2.weight'].shape
        
        if actual_fc1_shape != expected_fc1_shape:
            raise ValueError(f"Architecture mismatch for fc1 (encoder): "
                           f"Expected {expected_fc1_shape}, got {actual_fc1_shape}. "
                           f"Checkpoint hidden_size={actual_fc1_shape[0]}, input_size={actual_fc1_shape[1]}, "
                           f"but config requires hidden_size={config.model.hidden_size}, d={config.data.d}")
        
        if actual_fc2_shape != expected_fc2_shape:
            raise ValueError(f"Architecture mismatch for fc2 (decoder): "
                           f"Expected {expected_fc2_shape}, got {actual_fc2_shape}. "
                           f"Checkpoint input_size={actual_fc2_shape[0]}, hidden_size={actual_fc2_shape[1]}, "
                           f"but config requires d={config.data.d}, hidden_size={config.model.hidden_size}")
        
        # Load state dict into model
        try:
            model.load_state_dict(checkpoint_state, strict=True)
            print(f"✓ Checkpoint loaded successfully. Architecture verified:")
            print(f"  Encoder (fc1): {actual_fc1_shape}")
            print(f"  Decoder (fc2): {actual_fc2_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint state dict: {e}") from e
    
    # Validate device before training
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        config.data.device = "cpu"
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    print(f"Two-stage training: SGLD then Adam on device: {device_str}")
    
    # Train
    model = train_two_stage(
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
        save_encoder_decoder_pairs(model=model, save_path=pairs_data_path)
        plot_encoder_decoder_pairs(
            model=model, 
            save_path=pairs_plot_path,
            colored_pairs=tracker.colored_pairs
        )
    
    # Save final results
    model_state = model.state_dict() if config.save_model else None
    tracker.save_results(final_metrics={}, model_state=model_state)


if __name__ == "__main__":
    main()
