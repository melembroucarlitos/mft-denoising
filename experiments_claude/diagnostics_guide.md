# Training Diagnostics Guide

Comprehensive guide to using the real-time diagnostic tools for monitoring blob formation during training.

---

## Overview

Three integrated tools help you efficiently explore parameter space and monitor blob formation:

1. **Real-Time Diagnostics** - Per-epoch blob metrics with <5% overhead
2. **Batch Size Profiler** - Find optimal batch size for your hardware
3. **Training Monitor** - Live visualization of metrics during training

---

## Tool 1: Real-Time Diagnostics

### What It Does

Computes blob formation metrics during training by sampling weight pairs and running lightweight clustering analysis. Metrics are saved to `results.json` alongside training history.

### Metrics Computed

- **Weight Correlation**: Pearson correlation between encoder and decoder weights
- **Number of Clusters**: Distinct blobs found via DBSCAN
- **Silhouette Score**: Clustering quality (-1 to 1, higher is better)
- **Cluster Centers**: Centroids of top 3 clusters
- **Weight Statistics**: Mean and standard deviation

### Performance

- **Overhead**: <5% (typically 50-100ms per epoch for d=1024)
- **Sampling**: 5,000 pairs by default (vs 524K total for d=1024, hidden=512)
- **Accuracy**: 95% confidence within ±1.4% of true values

### How to Use

#### Option 1: Enable in Config JSON

```json
{
  "training": {
    ...
    "enable_diagnostics": true,
    "diagnostic_sample_size": 5000
  }
}
```

#### Option 2: Enable Programmatically

```python
from experiments_claude.experiment_runner import create_config, run_experiment

config = create_config('reference_3blob')
config['training']['enable_diagnostics'] = True
config['training']['diagnostic_sample_size'] = 5000

run_experiment(config, 'blob_diagnostic_test')
```

### Output Format

Diagnostics are saved in `results.json` under each epoch's `diagnostics` field:

```json
{
  "training_history": [
    {
      "epoch": 1,
      "train": {"loss": 172.97, "scaled_loss": 118.57, ...},
      "test": {"scaled_loss": 81.03, ...},
      "diagnostics": {
        "weight_correlation": 0.023,
        "encoder_std": 0.145,
        "decoder_std": 0.138,
        "encoder_mean": -0.002,
        "decoder_mean": 0.001,
        "n_clusters_dbscan": 1,
        "silhouette_score": null,
        "cluster_centers": [[0.01, 0.02]],
        "n_noise_points": 8,
        "computation_time_ms": 95.3,
        "n_pairs_analyzed": 5000
      }
    },
    ...
  ]
}
```

### Interpreting Results

| Metric | Good Values | What It Means |
|--------|-------------|---------------|
| **n_clusters_dbscan** | 3 | Clear blob structure formed |
| **silhouette_score** | 0.5-0.8 | Well-separated blobs |
| **weight_correlation** | Near 0 or negative | Encoder-decoder decorrelation |
| **encoder_std** | Moderate (0.2-0.5) | Weights spread across regimes |

**Blob Formation Timeline:**
- **Early training** (epochs 1-5): Usually 1 cluster, no silhouette score
- **Mid training** (epochs 5-15): Clusters begin to separate
- **Late training** (epochs 15-20): Stable 3-blob structure (if successful)

### Advanced: Adjusting Sample Size

```python
# Faster but noisier (good for quick experiments)
config['training']['diagnostic_sample_size'] = 2000

# More accurate but slower (good for final runs)
config['training']['diagnostic_sample_size'] = 10000

# Full computation (no sampling, very slow)
from mft_denoising.diagnostics import full_blob_analysis
metrics = full_blob_analysis(model)
```

---

## Tool 2: Batch Size Profiler

### What It Does

Tests multiple batch sizes with short experiments (5 epochs each) to measure:
- Time per epoch
- GPU memory usage
- Throughput (samples/second)
- Final loss convergence

Generates a markdown report with recommendations.

### When to Use

- Before starting a large parameter sweep
- When switching to different hardware
- When changing model architecture (d, hidden_size)

### How to Use

#### Option 1: Command Line

```bash
cd /home/ubuntu/code/mft-denoising

python experiments_claude/batch_size_profiler.py \
  experiments_claude/configs/reference_3blob.json \
  --batch-sizes 2048 5120 10240 20480 \
  --epochs 5 \
  --output experiments_claude/batch_profile_d1024.md
```

#### Option 2: Python API

```python
from experiments_claude.experiment_runner import create_config
from experiments_claude.batch_size_profiler import profile_batch_sizes

config = create_config('reference_3blob')
results = profile_batch_sizes(
    config,
    batch_sizes=[2048, 5120, 10240, 20480],
    test_epochs=5,
    output_path='experiments_claude/batch_profile.md'
)

# Access results programmatically
for profile in results:
    if not profile.oom_error:
        print(f"Batch {profile.batch_size}: {profile.samples_per_second:.0f} samples/sec")
```

### Example Output

```markdown
# Batch Size Profile: d=1024, hidden=512

| Batch Size | Time/Epoch | GPU Memory | Samples/Sec | Final Train Loss | Final Test Loss | Status |
|------------|------------|------------|-------------|------------------|-----------------|--------|
| 2,048      | 12.3s      | 4.2 GB     | 166         | 45.2341          | 42.1123         | OK     |
| 5,120      | 18.7s      | 7.8 GB     | 273         | 44.8765          | 41.9876         | OK     |
| 10,240     | 28.4s      | 14.2 GB    | 360         | 44.5123          | 41.8543         | OK     |
| 20,480     | OOM        | -          | -           | -                | -               | ERROR  |

## Recommendation

**Optimal batch size: 10,240**

- Best throughput: 360 samples/sec
- Time per epoch: 28.4s
- GPU memory usage: 14.2 GB
- Final test loss: 41.8543
```

### Interpretation

- **Throughput**: Higher is better (more samples processed per second)
- **Memory**: Aim for 70-80% of GPU capacity to leave room for other processes
- **Loss**: Should be similar across batch sizes (if not, adjust learning rate)

---

## Tool 3: Training Monitor

### What It Does

Watches `results.json` during training and provides live-updating plots of:
- Silhouette score over time
- Number of clusters over time
- Weight correlation over time
- Train/test loss curves

### How to Use

#### Live Monitoring

```bash
# Terminal 1: Start experiment with diagnostics
python main.py experiments_claude/configs/reference_3blob.json

# Terminal 2: Start monitor (in separate terminal)
python experiments_claude/monitor_training.py experiments/reference_3blob_*
```

#### With Custom Refresh Interval

```bash
python experiments_claude/monitor_training.py experiments/my_experiment --interval 10
```

#### Static Plot (After Training)

```bash
python experiments_claude/monitor_training.py experiments/my_experiment \
  --static \
  --output my_training_plot.png
```

#### Python API

```python
from experiments_claude.monitor_training import TrainingMonitor

monitor = TrainingMonitor('experiments/my_experiment_20260118_143022')
monitor.start_live_plot(refresh_interval=5)
```

### Interpreting the Plots

**4-Panel Layout:**

1. **Top-Left: Silhouette Score**
   - Tracks blob separation quality
   - Look for: Increasing trend, stabilizing around 0.5-0.8

2. **Top-Right: Number of Clusters**
   - Counts distinct blobs
   - Look for: Transition from 1 → 3 clusters during training

3. **Bottom-Left: Weight Correlation**
   - Encoder-decoder correlation
   - Look for: Values near 0 or negative (decorrelation)

4. **Bottom-Right: Train/Test Loss**
   - Standard loss curves
   - Look for: Convergence without divergence

---

## Complete Workflow Example

### Step 1: Optimize Batch Size

```bash
cd /home/ubuntu/code/mft-denoising

python experiments_claude/batch_size_profiler.py \
  experiments_claude/configs/reference_3blob.json \
  --epochs 5
```

Review the generated markdown report and choose optimal batch size (e.g., 10240).

### Step 2: Update Config

```python
from experiments_claude.experiment_runner import create_config
import json

config = create_config('reference_3blob')
config['training']['batch_size'] = 10240  # From profiler recommendation
config['training']['enable_diagnostics'] = True
config['training']['diagnostic_sample_size'] = 5000

# Save updated config
with open('experiments_claude/configs/optimized_3blob.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### Step 3: Run with Monitoring

```bash
# Terminal 1: Start training
python main.py experiments_claude/configs/optimized_3blob.json

# Terminal 2: Monitor live
python experiments_claude/monitor_training.py experiments/optimized_3blob_*
```

### Step 4: Analyze Results

```python
from experiments_claude.experiment_runner import load_results
import matplotlib.pyplot as plt

results = load_results('optimized_3blob')

# Extract diagnostics timeline
epochs = [h['epoch'] for h in results['training_history']]
silhouettes = [h['diagnostics']['silhouette_score'] for h in results['training_history']]
n_clusters = [h['diagnostics']['n_clusters_dbscan'] for h in results['training_history']]

# Plot blob formation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(epochs, silhouettes, marker='o')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Blob Quality Over Training')
ax1.grid(True)

ax2.plot(epochs, n_clusters, marker='s', drawstyle='steps-post')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Number of Clusters')
ax2.set_title('Cluster Count')
ax2.grid(True)

plt.tight_layout()
plt.savefig('blob_formation_analysis.png')
print("Saved: blob_formation_analysis.png")
```

---

## Troubleshooting

### "No diagnostics available" in monitor

**Cause**: Diagnostics not enabled in config
**Fix**: Set `enable_diagnostics: true` in your config JSON

### OOM errors during batch size profiling

**Cause**: Batch size too large for GPU memory
**Fix**: This is expected! The profiler tests until OOM to find limits. Use the largest batch size that didn't OOM.

### Silhouette score always null

**Cause**: DBSCAN not finding enough clusters (needs ≥2 clusters)
**Fix**: This is normal early in training or in regimes that don't form blobs. Check later epochs.

### Diagnostics slow down training significantly

**Cause**: Sample size too large
**Fix**: Reduce `diagnostic_sample_size` from 5000 to 2000-3000

### Monitor not updating

**Cause**: Results file being written
**Fix**: Wait a few seconds. The monitor handles partial writes automatically.

---

## Theory: What the Metrics Mean

### Silhouette Score

Measures how well-separated clusters are:
- **0.7-1.0**: Excellent separation (strong blob structure)
- **0.5-0.7**: Good separation (moderate blobs)
- **0.25-0.5**: Weak separation (fuzzy boundaries)
- **< 0.25**: No meaningful clustering

### DBSCAN Parameters

- **eps=0.1**: Neighborhood radius for clustering
- **min_samples=50**: Minimum points to form a cluster

These are tuned for weight pair data in the range [-1, 1]. Adjust if your weights have different scales.

### Weight Correlation

- **Positive (~0.5-1.0)**: Encoder and decoder weights move together
- **Near zero (~-0.2 to 0.2)**: Independent evolution (good for blob formation)
- **Negative (~-1.0 to -0.5)**: Encoder and decoder anti-correlated

The 3-blob regime typically shows correlation near 0 or slightly negative.

---

## Performance Tips

### For Quick Iteration

```python
# Minimal sampling, no plots
config['training']['diagnostic_sample_size'] = 2000
config['training']['epochs'] = 10  # Shorter runs
config['save_plots'] = False  # Skip per-epoch plots
```

### For Production Runs

```python
# More accurate sampling
config['training']['diagnostic_sample_size'] = 10000
config['training']['epochs'] = 20
config['save_plots'] = True  # Save all plots
```

### For Final Analysis

```python
# Post-training full analysis
from experiments_claude.analysis.measure_decoupling import analyze_experiment

metrics = analyze_experiment('my_experiment')
# Includes full clustering, GMM fitting, etc.
```

---

## API Reference

### diagnostics.py

```python
from mft_denoising.diagnostics import compute_lightweight_blob_metrics

metrics = compute_lightweight_blob_metrics(
    model,                    # TwoLayerNet model
    n_samples=5000,          # Number of pairs to sample
    compute_full_stats=False, # If True, use all pairs (slow)
    eps=0.1,                 # DBSCAN epsilon
    min_samples=50           # DBSCAN min samples
)
```

### batch_size_profiler.py

```python
from experiments_claude.batch_size_profiler import profile_batch_sizes

results = profile_batch_sizes(
    config,                              # ExperimentConfig
    batch_sizes=[2048, 5120, 10240],    # Batch sizes to test
    test_epochs=5,                       # Epochs per test
    output_path='batch_profile.md'      # Markdown report path
)
```

### monitor_training.py

```python
from experiments_claude.monitor_training import TrainingMonitor

monitor = TrainingMonitor('experiments/my_experiment')
monitor.start_live_plot(refresh_interval=5)  # Live monitoring
monitor.generate_static_plot('plot.png')     # Static plot
```

---

## Changelog

### v1.0 (2026-01-18)
- Initial release
- Real-time diagnostics with <5% overhead
- Batch size profiler for GPU optimization
- Live training monitor with 4-panel plots

---

**Questions or issues?** Check the README or examine the source code for implementation details.
