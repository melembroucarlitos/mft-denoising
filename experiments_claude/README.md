# Interactive Hyperparameter Exploration

## Goal
Find hyperparameter regimes that produce clean encoder-decoder weight pair decoupling (3-blob structure) in sparse denoising task.

## Tools

### experiment_runner.py
Flexible experiment launcher with helper functions:
- `create_config(base, **overrides)` - Quick config generation
- `run_experiment(config, name)` - Run full experiment (~10 min)
- `quick_run(config)` - Fast test (1-2 min with reduced epochs/data)
- `load_results(name)` - Load results from experiment
- `compare_experiments(names)` - Side-by-side comparison

### Training Diagnostics (NEW!)

**Real-Time Diagnostics** - Monitor blob formation during training with <5% overhead:
- Enable with `enable_diagnostics: true` in config
- Per-epoch metrics: silhouette score, n_clusters, weight correlation
- Saves to `results.json` for later analysis

**Batch Size Profiler** (`batch_size_profiler.py`) - Find optimal batch size:
- Tests multiple batch sizes with short experiments
- Measures time/epoch, GPU memory, throughput
- Generates markdown report with recommendations
- Usage: `python experiments_claude/batch_size_profiler.py config.json`

**Training Monitor** (`monitor_training.py`) - Live visualization:
- Real-time plots of blob formation metrics
- 4-panel view: silhouette, clusters, correlation, loss
- Usage: `python experiments_claude/monitor_training.py experiments/my_exp`

**See [diagnostics_guide.md](diagnostics_guide.md) for complete documentation.**

### analysis/measure_decoupling.py
Automated metrics for quantifying decoupling:
- Silhouette score (clustering quality)
- Number of distinct blobs (DBSCAN)
- Gaussian mixture model fitting
- Statistical metrics (correlation, means, stds)

## Reference Config (Produces 3-Blob Plot)

Key parameters from known good config:
- **d**: 1024 (input dimension - critical, lower dims don't work)
- **hidden_size**: 512
- **encoder_init_scale**: 0.03 (very small!)
- **decoder_init_scale**: 0.03 (very small!)
- **decoder_reg**: 0.03 (3x baseline)
- **lambda_on**: 16.0 (half of baseline 32)
- **learning_rate**: 0.02 (2x baseline)
- **epochs**: 20 (much shorter than baseline 300!)
- **noise_variance**: 0.005 (cleaner data)
- **n_train**: 300k

## Quick Start

### Basic Usage

```python
import sys
sys.path.insert(0, '/home/ubuntu/code/mft-denoising/experiments_claude')

from experiment_runner import create_config, run_experiment, quick_run

# Run reference config
config = create_config('reference_3blob')
run_experiment(config, 'validation_run')

# Try modified parameters
config2 = create_config('reference_3blob', encoder_initialization_scale=0.05)
quick_run(config2, experiment_name='test_init_005')
```

### With Real-Time Diagnostics

```python
from experiment_runner import create_config, run_experiment

# Enable diagnostics
config = create_config('reference_3blob')
config['training']['enable_diagnostics'] = True
config['training']['diagnostic_sample_size'] = 5000

# Run experiment
run_experiment(config, 'blob_diagnostic_test')

# Monitor in separate terminal:
# python experiments_claude/monitor_training.py experiments/blob_diagnostic_test_*
```

### Find Optimal Batch Size

```bash
cd /home/ubuntu/code/mft-denoising
python experiments_claude/batch_size_profiler.py \
  experiments_claude/configs/reference_3blob.json \
  --batch-sizes 2048 5120 10240 20480 \
  --epochs 5
```

## Experiments Log

### Validation Run
- Date: 2026-01-17
- Config: reference_3blob (exact from user's plot)
- Status: Running...
- Result: TBD

---

## Key Insights (to be filled in during exploration)

### Critical Parameters
- TBD

### Phase Transitions Observed
- TBD

### Stable Regimes
- TBD
