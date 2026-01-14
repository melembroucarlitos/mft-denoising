import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    """Simple 2-layer neural network for denoising"""
    def __init__(self, input_size=3000, hidden_size=256, 
                 encoder_initialization_scale=1.0, decoder_initialization_scale=1.0):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            encoder_initialization_scale: Scale factor for encoder weight initialization
            decoder_initialization_scale: Scale factor for decoder weight initialization
        """
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)  # Output same size as input
        
        # Apply custom initialization scales
        with torch.no_grad():
            # Encoder initialization
            nn.init.xavier_normal_(self.fc1.weight)
            self.fc1.weight.mul_(encoder_initialization_scale)
            if self.fc1.bias is not None:
                self.fc1.bias.zero_()
            
            # Decoder initialization
            nn.init.xavier_normal_(self.fc2.weight)
            self.fc2.weight.mul_(decoder_initialization_scale)
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()    
    def forward(self, x):
        # Handle both batched (B, d) and unbatched (d,) inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension: (d,) -> (1, d)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = x.view(x.size(0), -1)  # Flatten: (B, d) -> (B, d)
        x = F.tanh(self.fc1(x))**3
        x = self.fc2(x)  # No activation on output for regression
        
        # Remove batch dimension if input was unbatched
        if squeeze_output:
            x = x.squeeze(0)  # (1, d) -> (d,)
        return x

if __name__ == "__main__":
    model = TwoLayerNet(input_size=1000, hidden_size=256, output_size=1000)
    x = torch.randn(1000)
    print(model(x).shape)