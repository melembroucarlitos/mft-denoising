import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from mft_denoising.data import DataConfig, TwoHotStream, create_dataloaders
from mft_denoising.nn import TwoLayerNet
from mft_denoising.plot import plot_encoder_weights_histogram, plot_network_outputs_histogram


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