# MFT Denoising: Technical Architecture Documentation

## Overview

This project investigates **encoder-decoder weight pair decoupling** in two-layer neural networks trained on sparse denoising tasks. The key phenomenon: weight pairs form distinct "blobs" in 2D scatter plots, indicating clean mean field theory (MFT) behavior.

**Core Question**: Can we observe and leverage this weight decoupling structure for efficient training?

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Experiment Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ main.py      │ │ experiment_  │ │ quick_hyper │
            │              │ │ runner.py    │ │ param_sweep │
            │ Training     │ │              │ │              │
            │ orchestration│ │ High-level   │ │ Automated    │
            │              │ │ automation   │ │ param search │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                   │                │                │
                   └────────┬───────┴────────────────┘
                            ▼
                ┌────────────────────────┐
                │  ExperimentTracker     │
                │  (experiment.py)       │
                │  - Logging             │
                │  - Checkpointing       │
                │  - Results management  │
                └────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ TwoLayerNet │     │ TwoHotStream│     │ diagnostics │
│ (nn.py)     │     │ (data.py)   │     │ .py         │
│             │     │             │     │             │
│ Encoder     │     │ Sparse data │     │ Blob        │
│ ↓ tanh³     │     │ generation  │     │ monitoring  │
│ Decoder     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Core Components

### 1. Configuration System ([mft_denoising/config.py](mft_denoising/config.py))

Hierarchical dataclass structure for experiment configuration:

```python
ExperimentConfig
├── ModelConfig
│   ├── hidden_size: int (neurons in hidden layer)
│   ├── encoder_initialization_scale: float (weight init std)
│   └── decoder_initialization_scale: float
├── TrainingConfig
│   ├── optimizer_type: "adam" | "sgld"
│   ├── learning_rate: float
│   ├── batch_size: int
│   ├── epochs: int
│   ├── l2_regularization: float
│   ├── enable_diagnostics: bool (track blob formation)
│   └── save_epoch_checkpoints: bool
├── LossConfig
│   ├── loss_type: "mse" | "scaled_mse"
│   └── lambda_on: float (weighting for sparse positions)
└── DataConfig
    ├── d: int (input dimension)
    ├── sparsity: int (number of active positions)
    ├── noise_variance: float
    ├── train_dataset_size: int
    └── test_dataset_size: int
```

**Purpose**: Single source of truth for all experiment parameters. All components receive this config.

**Key Function**: `ExperimentConfig.from_preset(name: str)` - Load predefined configurations like "reference_3blob"

### 2. Data Generation ([mft_denoising/data.py](mft_denoising/data.py))

Generates two-hot sparse signals with additive Gaussian noise:

```python
TwoHotStream
    ↓ yield batches
    (batch_size, d) tensor
    - Exactly 2 active positions per sample
    - Values: μ₁, μ₂ at active positions
    - Additive noise: N(0, σ²)
    ↓
TwoHotDataset (PyTorch wrapper)
    ↓
create_dataloaders()
    → train_loader, test_loader
```

**Key Properties**:
- **Sparsity**: Only 2 out of d positions are active (rest are zero + noise)
- **Signal values**: Sampled from uniform distribution U[0.5, 1.5]
- **Noise**: Independent Gaussian on each position
- **Purpose**: Test if network learns to denoise by separating signal from noise

**Functions**:
- `TwoHotStream.__next__()`: Generate batch of noisy sparse signals
- `create_dataloaders(config)`: Factory for train/test data loaders

### 3. Model Architecture ([mft_denoising/nn.py](mft_denoising/nn.py))

Simple two-layer network with **tanh³ activation**:

```
Input (d-dim) → fc1 (Encoder: d×h) → tanh³ → fc2 (Decoder: h×d) → Output (d-dim)
                  ↑ weight pairs        ↑          ↑
              W_enc[i,:] ←───────────→ W_dec[:,i]
                         coupled pairs form "blobs"
```

**Class**: `TwoLayerNet`

**Key Properties**:
- **Encoder weights**: `fc1.weight` shape (hidden_size, d) - each row is an encoder
- **Decoder weights**: `fc2.weight` shape (d, hidden_size) - each column is a decoder
- **Weight pairs**: (W_enc[i,:], W_dec[:,i]) are the i-th encoder-decoder pair
- **Initialization**: Xavier/Kaiming scaled by `initialization_scale` parameter

**Forward Pass**:
```python
def forward(self, x):
    hidden = torch.tanh(self.fc1(x)) ** 3  # Cubic nonlinearity
    return self.fc2(hidden)
```

**Why tanh³?**:
- Smooth nonlinearity
- Theoretical tractability for mean field analysis
- Encourages weight decoupling (hypothesis being tested)

### 4. Loss Functions ([main.py](main.py):18-24)

Two loss types controlled by `config.loss.loss_type`:

**A. Standard MSE**:
```python
loss = F.mse_loss(predictions, targets)
```

**B. Scaled MSE** (position-weighted):
```python
loss = λ_on × MSE(active_positions) + λ_off × MSE(inactive_positions)
```

Where:
- `λ_on = config.loss.lambda_on` (typically 1.0)
- `λ_off = 1.0` (fixed)
- Active positions: where ground truth signal has values μ₁, μ₂
- Inactive positions: where ground truth is zero

**Purpose**: Scaled MSE emphasizes accurate reconstruction of sparse signal positions.

### 5. Training Loop - Stage 1 ([main.py](main.py):26-130)

**Function**: `train(model, train_loader, test_loader, config, tracker)`

**Purpose**: Train both encoder and decoder from scratch

**Flow**:
```python
for epoch in range(config.training.epochs):
    # Training phase
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch)
        loss = compute_loss(predictions, batch, config)
        loss.backward()
        optimizer.step()

    # Validation phase
    test_loss = evaluate(model, test_loader, config)

    # Optional: Real-time diagnostics
    if config.training.enable_diagnostics:
        blob_metrics = compute_lightweight_blob_metrics(model)
        # Track: n_clusters, silhouette_score, weight_correlation

    # Logging
    tracker.log_epoch(epoch, train_metrics, test_metrics, diagnostics)

    # Optional: Save checkpoint
    if config.training.save_epoch_checkpoints:
        tracker.save_checkpoint(epoch, model, optimizer)
```

**Optimizer Options**:
- **ADAM**: Standard deterministic optimizer
- **SGLD** (Stochastic Gradient Langevin Dynamics):
  ```python
  # Gradient descent + noise injection
  θ_new = θ_old - lr × ∇L + √(2 × lr × T) × N(0, I)
  ```
  Temperature parameter `T` controls exploration

**Output**: Trained model with coupled encoder-decoder pairs

### 6. Training Loop - Stage 2 ([main.py](main.py):133-262)

**Function**: `train_frozen_encoder(config, tracker, gmm_means, n_frozen_encoders)`

**Purpose**: Sample frozen encoders from GMM and train new decoders

**Flow**:
```python
# 1. Sample frozen encoders from GMM
frozen_encoders = sample_from_gmm(
    gmm_means,           # Cluster centers from Stage 1
    gmm_covariances,     # Cluster spreads
    n_frozen_encoders    # How many to sample
)

# 2. Create model with frozen encoder weights
model = TwoLayerNet(config)
model.fc1.weight.data = frozen_encoders
model.fc1.requires_grad_(False)  # Freeze encoder

# 3. Train only decoder
optimizer = create_optimizer(
    model.fc2.parameters(),  # Only decoder parameters
    config
)

for epoch in range(config.training.epochs):
    # Same training loop as Stage 1
    # But encoder weights are fixed
    ...
```

**Key Insight**: If Stage 1 forms clean blobs, we can:
1. Fit GMM to encoder weights
2. Sample new encoders from GMM clusters
3. Train decoders to match these frozen encoders
4. Result: Faster convergence, leverages discovered structure

**GMM Fitting** ([main.py](main.py):265-298):
```python
from sklearn.mixture import GaussianMixture

# Extract encoder weights from trained Stage 1 model
encoder_weights = model.fc1.weight.data  # Shape: (hidden_size, d)

# Fit GMM
gmm = GaussianMixture(
    n_components=expected_n_clusters,  # e.g., 3 for reference_3blob
    covariance_type='full'
)
gmm.fit(encoder_weights)

# Extract cluster parameters
gmm_means = gmm.means_          # Shape: (n_clusters, d)
gmm_covariances = gmm.covariances_  # Shape: (n_clusters, d, d)
```

### 7. Real-Time Blob Diagnostics ([mft_denoising/diagnostics.py](mft_denoising/diagnostics.py))

**Function**: `compute_lightweight_blob_metrics(model, n_samples=5000)`

**Purpose**: Monitor weight pair decoupling during training

**Method**:
```python
# 1. Sample encoder-decoder weight pairs
encoder_weights = model.fc1.weight.data  # (h, d)
decoder_weights = model.fc2.weight.data  # (d, h)

pairs = []
for i in range(min(hidden_size, n_samples)):
    enc_i = encoder_weights[i, :]  # i-th encoder (d-dim)
    dec_i = decoder_weights[:, i]  # i-th decoder (d-dim)
    pairs.append([enc_i, dec_i])   # 2D point

pairs = torch.stack(pairs)  # Shape: (n_samples, 2, d)

# 2. Flatten to 2D points
points_2d = pairs.reshape(n_samples, 2 * d)

# 3. Compute correlation
correlation = torch.corrcoef(
    torch.stack([pairs[:, 0].flatten(), pairs[:, 1].flatten()])
)[0, 1]

# 4. Run DBSCAN clustering
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=5).fit(points_2d)
n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

# 5. Compute silhouette score (if multiple clusters)
if n_clusters >= 2:
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(points_2d, clustering.labels_)
else:
    silhouette = None  # Undefined for single cluster

return {
    'n_clusters_dbscan': n_clusters,
    'silhouette_score': silhouette,
    'weight_correlation': correlation.item()
}
```

**Interpretation**:
- **n_clusters ≈ expected**: Weight pairs forming expected blob structure
- **silhouette > 0.7**: Excellent cluster separation
- **silhouette 0.5-0.7**: Good separation
- **silhouette < 0.5**: Weak or overlapping clusters
- **correlation → 0**: Encoder and decoder weights decorrelating

### 8. Experiment Tracking ([mft_denoising/experiment.py](mft_denoising/experiment.py))

**Class**: `ExperimentTracker`

**Purpose**: Centralized logging, checkpointing, and results management

**Lifecycle**:
```python
# 1. Initialize
tracker = ExperimentTracker(
    config=config,
    experiment_name="my_experiment",
    output_dir="experiments/my_experiment_20260118_123456"
)

# 2. Start experiment
tracker.start()
# Creates output directory
# Saves config.json
# Initializes training_history list

# 3. Log epochs
for epoch in range(epochs):
    tracker.log_epoch(
        epoch=epoch,
        train_metrics={'loss': 10.5},
        test_metrics={'scaled_loss': 12.3},
        diagnostics={'n_clusters': 3, 'silhouette_score': 0.85}
    )
    # Optionally saves checkpoint
    # Appends to training_history

# 4. Finalize
tracker.save_results()
# Writes results.json with:
# - experiment_name
# - output_dir
# - config (full config dict)
# - training_history (list of epoch dicts)
```

**Key Methods**:
- `start()`: Initialize experiment, create output directory
- `log_epoch()`: Record metrics, optionally run diagnostics and save checkpoints
- `save_checkpoint()`: Save model and optimizer state_dict
- `save_results()`: Write final results.json

**Output Structure**:
```
experiments/
└── my_experiment_20260118_123456/
    ├── config.json              # Full experiment configuration
    ├── results.json             # Training history and final metrics
    └── checkpoints/
        ├── epoch_0.pt
        ├── epoch_1.pt
        └── ...
```

### 9. High-Level Experiment Automation ([experiments_claude/experiment_runner.py](experiments_claude/experiment_runner.py))

**Purpose**: Streamline experiment execution via subprocess

**Key Functions**:

**A. `create_config(preset_name: str) -> dict`**
```python
# Load preset configuration
config = create_config('reference_3blob')
# Returns config as dictionary for easy modification
```

**B. `run_experiment(config: dict, experiment_name: str, wait: bool = True)`**
```python
# Run experiment in subprocess
run_experiment(
    config=modified_config,
    experiment_name="test_lr01",
    wait=True  # Block until completion
)
# Subprocess executes: python main.py --config <temp_config.json>
```

**C. `load_results(experiment_name: str) -> dict`**
```python
# Load results from most recent matching experiment
result = load_results("test_lr01")
# Returns parsed results.json

# IMPORTANT: Sorts directories by modification time
# Always loads most recent if multiple matches exist
```

**Bug Fix Applied** (lines 188-199):
```python
# Find all matching directories
matching_dirs = list(experiments_dir.glob(f"{experiment_name}*"))

# Sort by modification time (most recent first)
matching_dirs = sorted(matching_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

# Always use most recent
results_path = matching_dirs[0] / "results.json"
```

### 10. Hyperparameter Sweep ([experiments_claude/quick_hyperparam_sweep.py](experiments_claude/quick_hyperparam_sweep.py))

**Purpose**: Automated exploration of hyperparameter space

**Architecture**:
```python
design_sweep(budget_hours=1.5)
    ↓ Generate experiment configurations
    List[{name, learning_rate, batch_size, encoder_init_scale, ...}]
    ↓
run_sweep(experiments, save_results=True)
    ↓ For each config
    run_sweep_experiment(config, name_suffix)
        ↓ Execute experiment
        run_experiment(config, experiment_name, wait=True)
        ↓ Load results
        load_results(experiment_name)
        ↓ Extract metrics
        {success, training_time_seconds, final_test_loss,
         final_n_clusters, final_silhouette, config_params}
    ↓ Aggregate results
    sweep_results_final.json
    ↓
analyze_sweep_results(results)
    → Print analysis (fastest, best loss, best blob quality)
```

**Key Functions**:

**A. `design_sweep(budget_hours: float) -> List[dict]`**

Generates experiment configurations exploring:
- **Learning rate**: [0.005, 0.01, 0.02, 0.04, 0.08]
- **Batch size**: [5120, 10240, 15360, 20480]
- **Init scale**: [0.01, 0.02, 0.03, 0.05, 0.1]
- **Combinations**: Promising parameter interactions

Returns ~21 experiments (estimated 1-1.5 hours total)

**B. `create_sweep_config(...) -> dict`**

Creates modified configuration:
```python
config = create_config('reference_3blob')
config['training']['learning_rate'] = lr
config['training']['batch_size'] = batch_size
config['training']['epochs'] = 12  # Reduced for speed
config['training']['enable_diagnostics'] = True
config['training']['save_epoch_checkpoints'] = False  # Save space
config['model']['encoder_initialization_scale'] = encoder_init_scale
config['model']['decoder_initialization_scale'] = decoder_init_scale
return config
```

**C. `run_sweep_experiment(config: dict, name_suffix: str) -> dict`**

Executes single sweep experiment with robust error handling:

```python
try:
    # Run experiment
    run_experiment(config, f"sweep_{name_suffix}", wait=True)

    # Load results
    result = load_results(f"sweep_{name_suffix}")

    # Validate structure (Bug Fix Applied)
    if 'training_history' not in result:
        raise ValueError(f"Results missing 'training_history' key")
    if not result['training_history']:
        raise ValueError(f"Results has empty 'training_history'")

    # Extract metrics
    final_epoch = result['training_history'][-1]
    return {
        'success': True,
        'training_time_seconds': elapsed_time,
        'final_test_loss': final_epoch['test']['scaled_loss'],
        'final_n_clusters': final_epoch['diagnostics']['n_clusters_dbscan'],
        'final_silhouette': final_epoch['diagnostics']['silhouette_score'],
        'config_params': {...}
    }

except Exception as e:
    # Robust error handling (Bug Fix Applied)
    return {
        'success': False,
        'error': str(e),
        'training_time_seconds': elapsed_time,
        'config_params': {
            'learning_rate': config.get('training', {}).get('learning_rate', 'unknown'),
            # Use .get() with defaults to prevent KeyError in exception handler
            ...
        }
    }
```

**D. `analyze_sweep_results(results: List[dict])`**

Prints analysis:
- Success rate
- Fastest experiments
- Lowest loss experiments
- Best blob quality (highest silhouette)
- Speedups vs reference baseline

**Output Files**:
- `sweep_results_intermediate.json`: Updated after each experiment (for monitoring)
- `sweep_results_final.json`: Complete results after sweep finishes

### 11. Sweep Visualization ([experiments_claude/visualize_sweep.py](experiments_claude/visualize_sweep.py))

**Purpose**: Generate sanity-check plots from sweep results

**Function**: `plot_sweep_summary(results: List[dict])`

**Creates 3-panel figure**:

**Panel 1: Training Efficiency**
- Scatter plot: training time (x) vs final loss (y)
- Color-coded by number of clusters
- Reference experiment marked with star

**Panel 2: Cluster Formation**
- Bar chart: experiment index (x) vs number of clusters (y)
- Green dashed line at target (3 clusters)
- Color matches cluster count

**Panel 3: Blob Quality**
- Bar chart: experiment index (x) vs silhouette score (y)
- Only experiments with multiple clusters
- Thresholds: 0.5 (good), 0.7 (excellent)

**Outputs**:
- `experiments_claude/figures/sweep_summary.png`: Main visualization
- `experiments_claude/figures/sweep_legend.txt`: Experiment index → name mapping

**Usage**:
```bash
python experiments_claude/visualize_sweep.py
# Or monitor intermediate results:
python experiments_claude/visualize_sweep.py experiments_claude/sweep_results_intermediate.json
```

## Complete Data Flow

### Single Experiment Flow

```
1. Load Configuration
   ExperimentConfig.from_preset('reference_3blob')
   ↓
2. Create Data Loaders
   create_dataloaders(config)
   → train_loader, test_loader
   ↓
3. Initialize Model
   TwoLayerNet(config)
   ↓
4. Initialize Tracker
   ExperimentTracker(config, experiment_name, output_dir)
   tracker.start()  # Creates directories, saves config
   ↓
5. Training Loop (Stage 1)
   for epoch in range(epochs):
       - Train on batches
       - Evaluate on test set
       - Optionally compute diagnostics (blob metrics)
       - Log epoch to tracker
       - Optionally save checkpoint
   ↓
6. Finalize
   tracker.save_results()
   → results.json with full training history
```

### Hyperparameter Sweep Flow

```
1. Design Sweep
   experiments = design_sweep(budget_hours=1.5)
   → List of 21 config variants
   ↓
2. For each experiment:
   ├─ Create modified config
   ├─ run_experiment(config, name, wait=True)
   │  └─ Subprocess: python main.py --config temp.json
   │     └─ Full single experiment flow (above)
   ├─ load_results(name)
   │  └─ Read results.json from output directory
   ├─ Extract metrics
   │  └─ training_time, final_loss, n_clusters, silhouette
   └─ Save to sweep_results_intermediate.json
   ↓
3. Analysis
   analyze_sweep_results(results)
   → Print best performers by various metrics
   ↓
4. Visualization
   plot_sweep_summary(results)
   → Generate plots in experiments_claude/figures/
```

## Key Design Decisions

### 1. Why Two Stages?

**Stage 1**: Discover weight pair structure
- Train from scratch
- Observe blob formation
- Extract cluster structure via GMM

**Stage 2**: Leverage discovered structure
- Sample frozen encoders from GMM clusters
- Train decoders to match
- Hypothesis: Should be faster/more efficient if blobs are meaningful

### 2. Why Real-Time Diagnostics?

Track blob formation **during** training:
- Verify expected structure emerges
- Detect training failures early (no blobs forming)
- Correlate hyperparameters with blob quality
- Inform when to stop training (structure stabilizes)

### 3. Why Scaled MSE Loss?

Standard MSE treats all positions equally:
```
Loss = mean((ŷ - y)²)
```

But in sparse data, signal is concentrated:
- 2 active positions: Carry information
- d-2 inactive positions: Just noise

Scaled MSE emphasizes active positions:
```
Loss = λ_on × mean((ŷ_active - y_active)²) + mean((ŷ_inactive - y_inactive)²)
```

Result: Network learns to denoise by focusing on signal.

### 4. Why Subprocess for Experiments?

`run_experiment()` uses subprocess rather than direct function call:

**Benefits**:
- **Isolation**: Each experiment runs in fresh Python process
- **No module caching**: Avoids stale code bugs
- **Clean memory**: Prevents memory leaks across experiments
- **Crash resilience**: One experiment crash doesn't kill sweep

**Implementation**:
```python
cmd = ['python', 'main.py', '--config', temp_config_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if wait:
    stdout, stderr = process.communicate()
```

### 5. Why Sort Directories by Modification Time?

**Problem**: `glob()` returns directories in arbitrary order (filesystem-dependent)

**Bug Scenario**:
```
experiments/
├── sweep_lr01_20260117_100000/  (old run, mtime: Jan 17 10:00)
└── sweep_lr01_20260118_120000/  (new run, mtime: Jan 18 12:00)

# glob returns: [old, new] or [new, old] (unpredictable)
# Using [-1] might load WRONG results
```

**Fix**: Always sort by modification time:
```python
matching_dirs = sorted(
    experiments_dir.glob(f"{experiment_name}*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True  # Most recent first
)
results_path = matching_dirs[0] / "results.json"
```

## Common Usage Patterns

### Run Single Experiment

```python
from experiments_claude.experiment_runner import create_config, run_experiment, load_results

# Load and modify config
config = create_config('reference_3blob')
config['training']['learning_rate'] = 0.01
config['training']['epochs'] = 12

# Run experiment
run_experiment(config, 'test_lr01', wait=True)

# Load results
result = load_results('test_lr01')
print(f"Final loss: {result['training_history'][-1]['test']['scaled_loss']}")
print(f"Clusters: {result['training_history'][-1]['diagnostics']['n_clusters_dbscan']}")
```

### Run Hyperparameter Sweep

```python
from experiments_claude.quick_hyperparam_sweep import design_sweep, run_sweep, analyze_sweep_results

# Design experiments
experiments = design_sweep(budget_hours=1.5)

# Run sweep (takes 1-1.5 hours)
results = run_sweep(experiments, save_results=True)

# Analyze results
analyze_sweep_results(results)
```

### Monitor Sweep Progress

```bash
# Watch intermediate results
watch -n 30 'tail -50 experiments_claude/sweep_results_intermediate.json'

# Check running experiments
ls -lt experiments/sweep_* | head -5
```

### Visualize Sweep Results

```python
from experiments_claude.visualize_sweep import load_sweep_results, plot_sweep_summary

results = load_sweep_results('experiments_claude/sweep_results_final.json')
plot_sweep_summary(results)
```

## Testing Infrastructure

### Unit Tests

**File**: [experiments_claude/test_sweep_infrastructure.py](experiments_claude/test_sweep_infrastructure.py)

**Purpose**: Validate sweep infrastructure bug fixes

**Test Classes**:

1. **TestLoadResults**: Directory selection logic
   - Verifies most recent directory is loaded when multiple exist
   - Tests modification time sorting

2. **TestRunSweepExperiment**: Error handling
   - Success case: Returns valid metrics dict
   - Missing training_history: Catches error, returns error dict
   - Empty training_history: Validates and reports error

3. **TestSweepPrinting**: Output formatting
   - Success message formatting (handles None silhouette)
   - Failure message formatting

**Run Tests**:
```bash
# With pytest
python -m pytest experiments_claude/test_sweep_infrastructure.py -v

# Standalone (no pytest required)
python experiments_claude/test_sweep_simple.py
```

## Bug Fixes Applied

### Bug #1: Non-deterministic Directory Selection

**Location**: [experiments_claude/experiment_runner.py](experiments_claude/experiment_runner.py):188-199

**Fix**: Sort by modification time instead of relying on glob order

### Bug #2: Missing Result Validation

**Location**: [experiments_claude/quick_hyperparam_sweep.py](experiments_claude/quick_hyperparam_sweep.py):98-108

**Fix**: Validate `training_history` exists and is non-empty before accessing

### Bug #3: Fragile Exception Handler

**Location**: [experiments_claude/quick_hyperparam_sweep.py](experiments_claude/quick_hyperparam_sweep.py):132-144

**Fix**: Use `.get()` with defaults instead of direct key access

## File Reference

### Core Implementation
- [main.py](main.py) - Training orchestration, Stage 1 & 2 loops
- [mft_denoising/config.py](mft_denoising/config.py) - Configuration dataclasses
- [mft_denoising/nn.py](mft_denoising/nn.py) - TwoLayerNet model architecture
- [mft_denoising/data.py](mft_denoising/data.py) - Sparse data generation
- [mft_denoising/experiment.py](mft_denoising/experiment.py) - ExperimentTracker
- [mft_denoising/diagnostics.py](mft_denoising/diagnostics.py) - Blob metrics

### Automation & Analysis
- [experiments_claude/experiment_runner.py](experiments_claude/experiment_runner.py) - High-level automation
- [experiments_claude/quick_hyperparam_sweep.py](experiments_claude/quick_hyperparam_sweep.py) - Hyperparameter sweep
- [experiments_claude/visualize_sweep.py](experiments_claude/visualize_sweep.py) - Result visualization

### Testing
- [experiments_claude/test_sweep_infrastructure.py](experiments_claude/test_sweep_infrastructure.py) - Pytest unit tests
- [experiments_claude/test_sweep_simple.py](experiments_claude/test_sweep_simple.py) - Standalone tests

## Future Directions

### Potential Extensions

1. **Adaptive sweep**: Use Bayesian optimization to explore hyperparameter space
2. **Multi-stage GMM**: Fit GMM at multiple training checkpoints, track cluster evolution
3. **Blob stability metrics**: Quantify how stable blob structure is across training runs
4. **Theoretical analysis**: Compare observed blob structure to MFT predictions
5. **Scaling studies**: Test on larger d, different sparsity patterns

### Open Questions

1. **Blob universality**: Do blobs form for other activation functions?
2. **Stage 2 efficiency**: How much speedup can we get from frozen encoder sampling?
3. **Optimal hyperparameters**: What settings maximize blob quality?
4. **Loss function impact**: How does scaled_mse vs mse affect blob formation?
5. **Regularization effects**: How does L2 regularization change weight coupling?

## Troubleshooting

### Sweep Crashes with TypeError

**Symptom**: `TypeError: unsupported format string passed to NoneType.__format__`

**Cause**: Loading stale results from old experiment runs

**Fix**: Applied in current code (sorts directories by modification time)

### Module Caching Issues

**Symptom**: Code changes not taking effect in running experiments

**Fix**: Start fresh Python process (subprocess isolation handles this)

### No Blobs Forming

**Possible Causes**:
- Learning rate too high/low
- Initialization scale inappropriate
- Insufficient training epochs
- Wrong loss function for task

**Debug**: Check diagnostics in training_history, verify n_clusters trends toward expected value

### Tests Failing

**Common Issues**:
- Missing pytest: Use [test_sweep_simple.py](experiments_claude/test_sweep_simple.py) instead
- Path issues: Tests assume codebase structure, check sys.path modifications
- Mocking issues: Verify mock patches match actual import structure

## Summary

This codebase implements a complete pipeline for:
1. **Training** two-layer networks on sparse denoising tasks
2. **Monitoring** encoder-decoder weight pair decoupling in real-time
3. **Automating** hyperparameter exploration
4. **Analyzing** blob formation quality across configurations

The architecture prioritizes **modularity** (clean separation of concerns), **observability** (real-time diagnostics), and **reproducibility** (comprehensive config and logging).
