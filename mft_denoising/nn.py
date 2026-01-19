"""
Two-layer neural network for sparse signal denoising.

Architecture:
    Input (d-dim) → fc1 (Encoder) → tanh³ → fc2 (Decoder) → Output (d-dim)

Key Concept - Weight Pairs:
    The i-th encoder-decoder pair is (fc1.weight[i, :], fc2.weight[:, i])
    These pairs can form "blobs" in 2D scatter plots, indicating clean
    mean field theory behavior and weight decoupling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    """
    Two-layer autoencoder-style network with cubic tanh activation.

    This architecture is designed to study encoder-decoder weight pair
    decoupling in the context of sparse signal denoising. The cubic
    nonlinearity (tanh³) is chosen for theoretical tractability in
    mean field analysis.

    Architecture:
        fc1 (encoder):  (d, h) weight matrix - each row is an encoder
        tanh³:          Cubic activation for smooth nonlinearity
        fc2 (decoder):  (h, d) weight matrix - each column is a decoder

    Weight Pairing:
        For hidden neuron i:
            encoder[i] = fc1.weight[i, :]  (d-dimensional)
            decoder[i] = fc2.weight[:, i]  (d-dimensional)
        These pairs (encoder[i], decoder[i]) form the units of analysis
        for studying weight decoupling and blob formation.
    """

    def __init__(self, input_size=3000, hidden_size=256,
                 encoder_initialization_scale=1.0, decoder_initialization_scale=1.0):
        """
        Initialize two-layer network with scaled weight initialization.

        Args:
            input_size: Dimension of input/output (d)
            hidden_size: Number of hidden neurons (h)
            encoder_initialization_scale: Multiplier for Xavier-initialized encoder weights
                Higher values → larger initial weights → faster initial learning
                Lower values → smaller initial weights → more conservative start
            decoder_initialization_scale: Multiplier for Xavier-initialized decoder weights
                Can differ from encoder to test asymmetric initialization

        Weight Initialization:
            1. Xavier normal initialization (variance scaled by fan-in/fan-out)
            2. Multiply by initialization_scale parameter
            3. Zero-initialize biases
        """
        super(TwoLayerNet, self).__init__()

        # Encoder: Maps input (d-dim) to hidden representation (h-dim)
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Decoder: Maps hidden representation (h-dim) back to input space (d-dim)
        self.fc2 = nn.Linear(hidden_size, input_size)

        # Apply custom initialization scales
        # Note: This happens AFTER default PyTorch initialization
        with torch.no_grad():
            # Encoder initialization
            # Xavier: weights ~ N(0, 2/(fan_in + fan_out))
            nn.init.xavier_normal_(self.fc1.weight)
            self.fc1.weight.mul_(encoder_initialization_scale)  # Scale by parameter
            if self.fc1.bias is not None:
                self.fc1.bias.zero_()  # Start with no bias

            # Decoder initialization
            nn.init.xavier_normal_(self.fc2.weight)
            self.fc2.weight.mul_(decoder_initialization_scale)
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()

    def forward(self, x):
        """
        Forward pass through the network.

        Computation:
            hidden = tanh³(x @ fc1.T + bias1)
            output = hidden @ fc2.T + bias2

        The cubic tanh activation (tanh³) provides:
            - Smooth differentiability
            - Bounded output range [-1, 1]
            - Odd symmetry: f(-x) = -f(x)
            - Theoretical tractability for mean field analysis

        Args:
            x: Input tensor, shape (B, d) or (d,)
               B = batch size, d = input dimension

        Returns:
            Denoised output, same shape as input
        """
        # Handle both batched (B, d) and unbatched (d,) inputs
        # This flexibility is useful for single-sample inference
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension: (d,) -> (1, d)
            squeeze_output = True
        else:
            squeeze_output = False

        # Encoder: project to hidden space
        x = x.view(x.size(0), -1)  # Flatten: (B, d) -> (B, d)
        x = F.tanh(self.fc1(x))**3  # Cubic activation

        # Decoder: project back to input space
        x = self.fc2(x)  # No activation on output (regression task)

        # Remove batch dimension if input was unbatched
        if squeeze_output:
            x = x.squeeze(0)  # (1, d) -> (d,)
        return x

if __name__ == "__main__":
    model = TwoLayerNet(input_size=1000, hidden_size=256, output_size=1000)
    x = torch.randn(1000)
    print(model(x).shape)