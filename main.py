import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from mft_denoising.data import DataConfig, TwoHotStream, create_dataloaders
from mft_denoising.nn import TwoLayerNet


def scaled_loss(
    output: torch.Tensor, 
    label: torch.Tensor,  
    lambda_on: torch.Tensor,
    mask_on: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom loss function that separately weights on/off positions.
    
    Args:
        output: Model predictions (B, d)
        label: Clean sparse signal (B, d) - not used directly, but kept for compatibility
        mask_on: Binary mask of active positions (B, d)
        lambda_on: Weight for active position loss
    
    Returns:
        total_loss, loss_on_w, loss_off
    """
    if mask_on is None:
        mask_on = label
    
    err_on = (output - label) * mask_on
    err_off = output * (1.0 - mask_on)
    loss_on_w = lambda_on * err_on.pow(2).sum(dim=1).mean()
    loss_off = err_off.pow(2).sum(dim=1).mean()
    return loss_on_w + loss_off, loss_on_w, loss_off

class SGLD:
    """Stochastic Gradient Langevin Dynamics optimizer"""
    def __init__(self, params, lr=1e-2, temperature=1.0):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            temperature: Temperature parameter (noise_variance = 2 * lr * temperature)
        """
        self.params = list(params)
        self.lr = lr
        self.temperature = temperature
    
    def step(self):
        """Perform a single SGLD update step"""
        for p in self.params:
            if p.grad is None:
                continue
            
            # Standard gradient update
            d_p = p.grad.data
            
            # Add Langevin noise: noise ~ N(0, 2*lr*temperature)
            noise_std = np.sqrt(2 * self.lr * self.temperature)
            noise = torch.randn_like(p.data) * noise_std
            
            # SGLD update: θ_{t+1} = θ_t - lr * ∇L + noise
            p.data.add_(-self.lr * d_p + noise)
    
    def zero_grad(self):
        """Zero out gradients"""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()



def train_sgld(
    model, 
    train_loader, 
    test_loader, 
    epochs, 
    lr, 
    temperature,
    lambda_on,
    encoder_regularization, 
    decoder_regularization,
):
    """
    Train model using SGLD with separate encoder/decoder regularization.
    
    Args:
        model: TwoLayerNet model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        temperature: SGLD temperature (controls exploration)
        encoder_regularization: L2 regularization weight for encoder (fc1)
        decoder_regularization: L2 regularization weight for decoder (fc2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = SGLD(model.parameters(), lr=lr, temperature=temperature)
    criterion = nn.MSELoss()  # Use MSE for denoising regression task
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_scaled_loss = 0
        total_train_loss_on_w = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Compute MSE loss
            total_scaled_loss, loss_on_w, loss_off = scaled_loss(
                output=output,
                label=target,
                lambda_on=lambda_on
            )
            
            # Add L2 regularization for encoder and decoder
            encoder_l2 = torch.sum(model.fc1.weight ** 2)
            decoder_l2 = torch.sum(model.fc2.weight ** 2)
            
            # Total loss = MSE + regularization
            loss = total_scaled_loss + encoder_regularization * encoder_l2 + decoder_regularization * decoder_l2
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_scaled_loss += total_scaled_loss.item()
            total_train_loss_on_w += loss_on_w.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-Loss: {loss_on_w.item():.6f}, Off-Loss: {loss_off.item():.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_scaled_loss = total_train_scaled_loss / len(train_loader)   
        avg_loss_on_w = total_train_loss_on_w / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_total_scaled_loss = evaluate(model, test_loader, lambda_on, device)
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.6f}, Train Scaled Loss: {avg_scaled_loss:.6f}, Train On-Loss: {avg_loss_on_w:.6f}, Train Off-Loss: {avg_loss_off:.6f}, Test Scaled Loss: {test_total_scaled_loss:.6f}')
    
    return model


def evaluate(model, test_loader, lambda_on, device):
    """Evaluate model MSE loss"""
    model.eval()
    test_total_scaled_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_scaled_loss, loss_on_w, loss_off = scaled_loss(
                output=output,
                label=target,
                lambda_on=lambda_on
            )
            test_total_scaled_loss += total_scaled_loss.item()
    
    return test_total_scaled_loss / len(test_loader)


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
    
    # Add vertical lines for mean and median
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Encoder Weights (fc1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
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
    
    # Add vertical lines for mean, median, 0, and 1
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    ax.axvline(0, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Target: 0')
    ax.axvline(1, color='purple', linestyle=':', linewidth=2, alpha=0.5, label='Target: 1')
    
    # Add labels and title
    ax.set_xlabel('Output Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Neural Network Outputs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
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

def main():
    # Optimizer hyperparameters
    batch_size = 128
    epochs = 1
    lr = 1e-4
    temperature = 0  # Controls the amount of Langevin noise
    encoder_regularization = 0
    decoder_regularization = 0
    lambda_on = 10.0

    # Data hyperparameters
    data_dim = 32
    sparsity = 2
    noise_variance = 0.1
    n_train = 1000000
    n_val = 1000
    seed = 42
    cfg = DataConfig(d=data_dim, sparsity=sparsity, noise_variance=noise_variance, n_train=n_train, n_val=n_val, seed=seed)
    train_loader, test_loader = create_dataloaders(cfg, batch_size=batch_size)
    
    # Create model
    hidden_size = 128
    model = TwoLayerNet(
        input_size=data_dim, 
        hidden_size=hidden_size, 
        encoder_initialization_scale=1.0, 
        decoder_initialization_scale=1.0
    )
    
    print("Training with SGLD...")
    print(f"Learning rate: {lr}, Temperature: {temperature}, Encoder regularization: {encoder_regularization}, Decoder regularization: {decoder_regularization}")
    
    # Train
    model = train_sgld(model=model, 
                       train_loader=train_loader, 
                       test_loader=test_loader, 
                       epochs=epochs, 
                       lr=lr, 
                       temperature=temperature, 
                       lambda_on=lambda_on,
                       encoder_regularization=encoder_regularization, 
                       decoder_regularization=decoder_regularization,)

    print("\nTraining complete!")
    plot_encoder_weights_histogram(model=model, save_path='encoder_weights_histogram.png')
    plot_network_outputs_histogram(model=model, data_loader=test_loader, save_path='network_outputs_histogram.png')

if __name__ == "__main__":
    main()