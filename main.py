import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from mft_denoising.data import DataConfig, TwoHotStream, create_dataloaders
from mft_denoising.nn import TwoLayerNet


class SGLD:
    """Stochastic Gradient Langevin Dynamics optimizer"""
    def __init__(self, params, lr=1e-2, noise_scale=1.0):
        self.params = list(params)
        self.lr = lr
        self.noise_scale = noise_scale
    
    def step(self):
        """Perform a single SGLD update step"""
        for p in self.params:
            if p.grad is None:
                continue
            
            # Standard gradient update
            d_p = p.grad.data
            
            # Add Langevin noise: N(0, 2*lr)
            noise = torch.randn_like(p.data) * np.sqrt(2 * self.lr * self.noise_scale)
            
            # SGLD update: θ_{t+1} = θ_t - lr * ∇L + noise
            p.data.add_(-self.lr * d_p + noise)
    
    def zero_grad(self):
        """Zero out gradients"""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


def train_sgld(model, train_loader, test_loader, epochs=10, lr=1e-3, noise_scale=1.0):
    """Train model using SGLD"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = SGLD(model.parameters(), lr=lr, noise_scale=noise_scale)
    criterion = nn.MSELoss(reduction='sum')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = train_loss / len(train_loader)            
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.6f}')
        # Evaluate
        test_loss = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1}/{epochs}: Test Loss: {test_loss:.6f}')
    
    return model


def evaluate(model, test_loader, device):
    """Evaluate model MSE loss"""
    model.eval()
    test_loss = 0
    criterion = nn.MSELoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)



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

def main():
    # Optimizer hyperparameters
    batch_size = 128
    epochs = 1
    lr = 1e-3
    noise_scale = 0.01  # Controls the amount of Langevin noise
    
    # Data hyperparameters
    data_dim = 3000
    sparsity = 2
    noise_variance = 0.03
    n_train = 100000
    n_val = 10000
    seed = 42
    cfg = DataConfig(d=data_dim, sparsity=sparsity, noise_variance=noise_variance, n_train=n_train, n_val=n_val, seed=seed)
    train_loader, test_loader = create_dataloaders(cfg, batch_size=batch_size)
    
    # Create model
    model = TwoLayerNet(input_size=data_dim, hidden_size=256, output_size=data_dim)
    
    print("Training with SGLD...")
    print(f"Learning rate: {lr}, Noise scale: {noise_scale}")
    
    # Train
    model = train_sgld(model, train_loader, test_loader, 
                       epochs=epochs, lr=lr, noise_scale=noise_scale)
    
    print("\nTraining complete!")

    plot_encoder_weights_histogram(model, save_path='encoder_weights_histogram.png')

if __name__ == "__main__":
    main()