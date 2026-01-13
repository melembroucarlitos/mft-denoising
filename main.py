import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


class TwoLayerNet(nn.Module):
    """Simple 2-layer neural network"""
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
    criterion = nn.CrossEntropyLoss()
    
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
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return model


def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def main():
    # Hyperparameters
    batch_size = 128
    epochs = 10
    lr = 1e-3
    noise_scale = 0.01  # Controls the amount of Langevin noise
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = TwoLayerNet(input_size=784, hidden_size=256, output_size=10)
    
    print("Training with SGLD...")
    print(f"Learning rate: {lr}, Noise scale: {noise_scale}")
    
    # Train
    model = train_sgld(model, train_loader, test_loader, 
                       epochs=epochs, lr=lr, noise_scale=noise_scale)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()