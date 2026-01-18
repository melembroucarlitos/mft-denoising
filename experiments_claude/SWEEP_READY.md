# Hyperparameter Sweep - Ready to Run

## Test Sweep Results ✓

**Status**: Infrastructure validated with 3 test experiments
**Test time**: ~35 seconds total
**Success rate**: 3/3 (100%)

All experiments completed successfully with proper metrics extraction.

## Full Sweep Plan

**Total experiments**: 22
**Estimated time**: 1-1.5 hours
**Time per experiment**: ~3-4 minutes (12 epochs, d=1024, n_train=300k)

### Sweep Strategy

**1. Learning Rate Sweep** (5 experiments)
- Test: [0.005, 0.01, 0.02, 0.04, 0.08]
- Goal: Find stability boundaries and potential speedups

**2. Batch Size Sweep** (4 experiments)
- Test: [5120, 10240, 15360, 20480]
- Goal: Maximize throughput without OOM

**3. Init Scale Sweep** (4 experiments)
- Test: [0.01, 0.02, 0.05, 0.1]
- Goal: Find instability triggers

**4. Combinations** (6 experiments)
- Fast training (high LR + large batch)
- Stable training (low LR + small batch)
- Aggressive (large init + high LR)
- Conservative (small init + low LR)
- Efficient (large batch + large init)
- Asymmetric init (encoder ≠ decoder)

**5. Overtraining Tests** (2 experiments)
- 20 epochs instead of 12
- Test if blob structure degrades

### Metrics Tracked

For each experiment:
- **Training time** (looking for OOM speedups)
- **Final test loss**
- **Blob quality**: n_clusters, silhouette score
- **Weight correlation**

### Expected Findings

1. **Stability region**: LR and batch size ranges where 3-blob structure emerges reliably
2. **Speedups**: Potential 1.5-2x faster with optimized hyperparameters
3. **Failure modes**: Too high LR or too large init → instability/collapse
4. **Overtraining effects**: Whether extended training degrades blob quality

## How to Run

### Start Full Sweep

```python
cd /home/ubuntu/code/mft-denoising

python3 <<'EOF'
from experiments_claude.quick_hyperparam_sweep import design_sweep, run_sweep, analyze_sweep_results

print("Starting full hyperparameter sweep...")
print("Estimated time: 1-1.5 hours")
print()

experiments = design_sweep(budget_hours=1.5)
results = run_sweep(experiments, save_results=True)

print("\n" + "="*80)
print("SWEEP COMPLETE")
print("="*80)

analyze_sweep_results(results)

print("\nResults saved to:")
print("  experiments_claude/sweep_results_final.json")
EOF
```

### Monitor Progress

Results are saved incrementally to:
- `experiments_claude/sweep_results_intermediate.json` (updated after each experiment)
- `experiments_claude/sweep_results_final.json` (final results)

### Generate Visualizations

After sweep completes, create plots:

```python
from experiments_claude.quick_hyperparam_sweep import design_sweep
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('experiments_claude/sweep_results_final.json', 'r') as f:
    results = json.load(f)

successful = [r for r in results if r['success']]

# Plot 1: Training time vs Loss
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

times = [r['training_time_seconds'] for r in successful]
losses = [r['final_test_loss'] for r in successful]
clusters = [r['final_n_clusters'] for r in successful]

ax1.scatter(times, losses, c=clusters, cmap='viridis', s=100, alpha=0.7)
ax1.set_xlabel('Training Time (s)')
ax1.set_ylabel('Final Test Loss')
ax1.set_title('Efficiency: Time vs Loss')
ax1.grid(True, alpha=0.3)

# Plot 2: LR vs Blob Quality
lr_sweep = [r for r in successful if 'lr' in r['name'] or 'reference' in r['name']]
lrs = [r['config_params']['learning_rate'] for r in lr_sweep]
sils = [r['final_silhouette'] if r['final_silhouette'] else 0 for r in lr_sweep]

ax2.plot(lrs, sils, 'o-', markersize=10)
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('LR Sensitivity')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# Plot 3: Batch Size vs Throughput
batch_sweep = [r for r in successful if 'batch' in r['name'] or 'reference' in r['name']]
batches = [r['config_params']['batch_size'] for r in batch_sweep]
throughputs = [r['config_params']['batch_size'] / r['training_time_seconds'] * r['config_params']['epochs'] for r in batch_sweep]

ax3.plot(batches, throughputs, 's-', markersize=10)
ax3.set_xlabel('Batch Size')
ax3.set_ylabel('Samples/sec')
ax3.set_title('Batch Size Efficiency')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments_claude/figures/sweep_analysis.png', dpi=300)
print("Saved: experiments_claude/figures/sweep_analysis.png")
```

## Files

**Created**:
- `experiments_claude/quick_hyperparam_sweep.py` - Main sweep runner
- `experiments_claude/test_sweep.py` - Test sweep validator
- `experiments_claude/SWEEP_READY.md` - This file

**Output locations**:
- Results: `experiments_claude/sweep_results_final.json`
- Plots: `experiments_claude/figures/sweep_analysis.png`
- Individual experiments: `experiments/sweep_*_TIMESTAMP/`

## Notes

- Sweep uses reduced epochs (12 instead of 20) for speed
- All experiments track blob formation with diagnostics
- Intermediate results saved after each experiment (safe to interrupt)
- GPU will run hot for ~1.5 hours - normal for A100
- Estimated disk usage: ~2-3 GB for all experiments

## Ready to Commit

Before running the full sweep, commit the current state:

```bash
git add experiments_claude/
git commit -m "Add hyperparameter sweep infrastructure and Gaussian health analysis

- Per-epoch checkpoint saving (config.py, experiment.py)
- Gaussian-ness diagnostics (gaussian_diagnostics.py, 400 lines)
- Per-epoch analysis (compare_epochs_gaussian.py)
- Visualization tools (plot_gaussianity_trends.py)
- Hyperparameter sweep (quick_hyperparam_sweep.py, test_sweep.py)
- Test sweep validated: 3/3 experiments passed

Ready for full sweep: 22 experiments, ~1.5 hours"
```

Then run the sweep and let GPU work autonomously!
