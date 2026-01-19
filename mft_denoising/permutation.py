"""
Weight pair permutation utilities.

Permutes encoder-decoder weight pairs (Enc_{i,k}, Dec_{k,i}) to create
new model initializations while preserving the pairing relationship.
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional
from pathlib import Path

from mft_denoising.nn import TwoLayerNet


def permute_weight_pairs(
    encoder_weights: torch.Tensor,
    decoder_weights: torch.Tensor,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly permute encoder-decoder weight pairs.
    
    For each position (i,k), the pair (Enc[i,k], Dec[k,i]) is permuted to a new
    position (i',k'), maintaining the pairing relationship:
    - Enc'[i,k] = Enc[i', k']
    - Dec'[k,i] = Dec[k', i']
    
    Args:
        encoder_weights: Encoder weights tensor of shape (hidden_size, input_size)
        decoder_weights: Decoder weights tensor of shape (input_size, hidden_size)
        seed: Random seed for permutation (for reproducibility)
    
    Returns:
        Tuple of (permuted_encoder_weights, permuted_decoder_weights)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Get dimensions
    hidden_size, input_size = encoder_weights.shape
    assert decoder_weights.shape == (input_size, hidden_size), \
        f"Decoder shape {decoder_weights.shape} doesn't match expected ({input_size}, {hidden_size})"
    
    # Convert to numpy for easier manipulation
    enc_np = encoder_weights.detach().cpu().numpy()
    dec_np = decoder_weights.detach().cpu().numpy()
    
    # Create list of all (i,k) pairs
    pairs = [(i, k) for i in range(hidden_size) for k in range(input_size)]
    
    # Create random permutation
    permuted_pairs = pairs.copy()
    random.shuffle(permuted_pairs)
    
    # Create inverse mapping: destination position -> source position
    # For each new position (i,k), we need to know which source (i',k') to use
    inverse_map = {permuted_pairs[idx]: pairs[idx] for idx in range(len(pairs))}
    
    # Create new weight matrices
    enc_new = np.zeros_like(enc_np)
    dec_new = np.zeros_like(dec_np)
    
    # Apply permutation
    for (i, k) in pairs:
        # Get the source position for this destination
        i_prime, k_prime = inverse_map[(i, k)]
        
        # Assign encoder weight: Enc'[i,k] = Enc[i', k']
        enc_new[i, k] = enc_np[i_prime, k_prime]
        
        # Assign decoder weight: Dec'[k,i] = Dec[k', i']
        dec_new[k, i] = dec_np[k_prime, i_prime]
    
    # Convert back to tensors
    enc_new_tensor = torch.from_numpy(enc_new).to(encoder_weights.dtype)
    dec_new_tensor = torch.from_numpy(dec_new).to(decoder_weights.dtype)
    
    return enc_new_tensor, dec_new_tensor


def create_permuted_model(
    source_model: torch.nn.Module,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Create a new model with permuted weights from a source model.
    
    Args:
        source_model: Source TwoLayerNet model to permute weights from
        seed: Random seed for permutation (for reproducibility)
        device: Device to place the new model on (if None, uses source model's device)
    
    Returns:
        New TwoLayerNet model with permuted weights
    """
    # Extract weights from source model
    encoder_weights = source_model.fc1.weight.data.clone()
    decoder_weights = source_model.fc2.weight.data.clone()
    
    # Get dimensions
    hidden_size, input_size = encoder_weights.shape
    
    # Apply permutation
    enc_permuted, dec_permuted = permute_weight_pairs(
        encoder_weights,
        decoder_weights,
        seed=seed
    )
    
    # Create new model with same architecture
    new_model = TwoLayerNet(
        input_size=input_size,
        hidden_size=hidden_size,
        encoder_initialization_scale=1.0,  # Doesn't matter, we'll set weights directly
        decoder_initialization_scale=1.0,
    )
    
    # Set permuted weights
    with torch.no_grad():
        new_model.fc1.weight.data = enc_permuted
        new_model.fc2.weight.data = dec_permuted
        
        # Copy biases if they exist (no permutation needed for biases)
        if source_model.fc1.bias is not None:
            if new_model.fc1.bias is not None:
                new_model.fc1.bias.data = source_model.fc1.bias.data.clone()
        if source_model.fc2.bias is not None:
            if new_model.fc2.bias is not None:
                new_model.fc2.bias.data = source_model.fc2.bias.data.clone()
    
    # Move to device if specified
    if device is not None:
        new_model = new_model.to(device)
    else:
        # Use source model's device
        if next(source_model.parameters()).is_cuda:
            new_model = new_model.cuda()
    
    return new_model


def load_and_permute_model(
    checkpoint_path: Path,
    config,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a model from checkpoint and create a permuted version.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        config: ExperimentConfig with model architecture info
        seed: Random seed for permutation
        device: Device to place model on
    
    Returns:
        New TwoLayerNet model with permuted weights
    """
    # Create source model and load checkpoint
    source_model = TwoLayerNet(
        input_size=config.data.d,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=1.0,
        decoder_initialization_scale=1.0,
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle compiled model checkpoints (strip _orig_mod. prefix if present)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_key = k[len('_orig_mod.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    source_model.load_state_dict(state_dict)
    
    # Create permuted model
    permuted_model = create_permuted_model(source_model, seed=seed, device=device)
    
    return permuted_model
