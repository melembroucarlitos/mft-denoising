# Gaussian Health Analysis Results

## Summary

We've implemented per-epoch checkpoint saving and Gaussian-ness diagnostics to track how blob quality evolves during training. This allows us to quantitatively measure whether clusters become "more Gaussian" over epochs.

## Implementation

### Added Features

1. **Per-epoch checkpoint saving** (`mft_denoising/config.py`, `mft_denoising/experiment.py`)
   - New config field: `save_epoch_checkpoints: bool`
   - Saves model state after each epoch as `checkpoint_epoch_{N}.pth`

2. **Per-epoch Gaussian health analysis** (`experiments_claude/analysis/compare_epochs_gaussian.py`)
   - `analyze_epochs_gaussian()`: Load checkpoints and compute Gaussian diagnostics for each epoch
   - `compare_with_reference()`: Side-by-side comparison with reference experiment
   - Auto-detects available epochs from checkpoint files

3. **Gaussian-ness trend visualization** (`experiments_claude/analysis/plot_gaussianity_trends.py`)
   - `plot_gaussianity_trends()`: 4-panel plot of all Gaussian metrics over epochs
   - `plot_log_percentiles()`: Log-scale tail ratio plots for specific clusters
   - `plot_cluster_size_evolution()`: Cluster size trends

### Gaussian Health Metrics

For each cluster at each epoch, we compute:

- **Radial KS**: χ²₂ goodness-of-fit (< 0.05 is excellent)
- **Tail ratios**: Empirical/expected at 90%, 99%, 99.9% percentiles (~1.0 is Gaussian)
- **Angle R**: Angular uniformity (< 0.013 for m=50k is uniform)
- **Projection AD**: Median Anderson-Darling over 20 random 1D projections (< 0.5 is excellent)
- **Mardia kurtosis dev**: |E[||y||⁴] - 8| (< 0.5 is normal kurtosis)

## Experiment Results

### Diagnostic Demo (5 epochs, d=1024, n_train=50k)

**Status**: ✓ COMPLETED

**Key Findings**:
- Very poor Gaussian-ness throughout all epochs
- Extremely high tail ratios (10-66x expected at 99.9th percentile)
- Very high projection AD (hundreds to thousands)
- Strong directional bias (angle R ~ 0.3)
- **Interpretation**: With only 5 epochs and limited data (50k samples), clusters don't have time to settle into clean Gaussian structure

**Cluster Evolution**:
- Epochs 1-2: 2 clusters detected
- Epochs 3-4: 6 clusters (fragmented, poor silhouette)
- Epoch 5: 5 clusters (partial consolidation)

**Detailed Metrics** (Epoch 2 as example):
```
Cluster 0 (262,740 points):
  Radial KS: 0.337 (poor)
  Tail ratios: {6: 1.56, 10: 10.37, 14: 66.50} (extreme heavy tails)
  Angle R: 0.3258 (strong directional bias)
  Projection AD: 1129.87 (very non-Gaussian)
```

### Reference 3-Blob (20 epochs, d=1024, n_train=300k)

**Status**: 🔄 RUNNING (experiment `reference_3blob_20260118_035836`)

**Expected timeline**:
- Started: Jan 18, 03:58
- Progress: Epoch 2/20 completed
- Estimated completion: ~10-12 minutes from start

**Purpose**:
- Track emergence of clean 3-blob structure
- Measure when clusters become "more Gaussian"
- Hypothesis: Clean separation by epochs 14-15, with improving Gaussian-ness in later epochs

**Next Steps** (once training completes):
1. Run `analyze_epochs_gaussian('reference_3blob_20260118_035836', epochs=[14, 15, 16, 17, 18, 19, 20])`
2. Plot log-scale tail percentiles for outer clusters (cluster IDs 0 and 2)
3. Compare with diagnostic_demo and reference_3blob_full epoch 20

## Expected Results for Reference 3-Blob

Based on the reference_3blob_full epoch 20 analysis:

**Outer clusters (clusters 0 and 2)**:
- Radial KS: 0.076-0.079 (moderate, not excellent)
- Tail ratios: 14-15x at 99.9th percentile (heavy tails)
- Angle R: ~0.10 (moderate directional bias)
- Projection AD: 4.5-22 (elevated, non-Gaussian)

**Middle cluster (cluster 1)**:
- Radial KS: 0.017 (excellent)
- Tail ratios: ~1.0 (close to Gaussian)
- Angle R: 0.0038 (excellent uniformity)
- Projection AD: 5.40 (elevated but best of three)

**Key Insight**: Even the "clean" reference 3-blob is NOT perfectly Gaussian, particularly in outer clusters. This suggests:
- Complex encoder-decoder coupling even in equilibrium
- Possible multi-modal substructure within clusters
- Training artifacts or transition states

## Hypothesis: Gaussian-ness Improves Over Epochs

User's hypothesis: Outer clusters become "more Gaussian" in later epochs (14-20) as clean separation emerges.

**Testing approach**:
1. Analyze epochs 14-20 of reference_3blob
2. Plot tail ratios on log scale (as requested)
3. Look for trends:
   - Decreasing tail ratios → less heavy tails
   - Decreasing angle R → more uniform angles
   - Decreasing projection AD → more Gaussian projections
   - Decreasing radial KS → better χ²₂ fit

**Expected finding**: Outer clusters show improving (but not perfect) Gaussian-ness as training progresses and blobs stabilize.

## Usage Examples

### Analyze Per-Epoch Gaussian Health

```python
from experiments_claude.analysis.compare_epochs_gaussian import analyze_epochs_gaussian

# Auto-detect all epochs
results = analyze_epochs_gaussian('reference_3blob_20260118_035836')

# Or specify epochs of interest
results = analyze_epochs_gaussian('reference_3blob_20260118_035836',
                                  epochs=[14, 15, 16, 17, 18, 19, 20])
```

### Visualize Trends

```python
from experiments_claude.analysis.plot_gaussianity_trends import (
    plot_gaussianity_trends,
    plot_log_percentiles,
    plot_cluster_size_evolution
)

# Comprehensive 4-panel plot
plot_gaussianity_trends(results, 'Reference 3-Blob',
                       output_path='experiments_claude/figures/gaussianity_trends.png')

# Focus on tail behavior for outer cluster (log scale)
plot_log_percentiles(results, 'Reference 3-Blob', cluster_id=0,
                    output_path='experiments_claude/figures/tail_percentiles_cluster0.png')

# Track cluster sizes
plot_cluster_size_evolution(results, 'Reference 3-Blob',
                           output_path='experiments_claude/figures/cluster_evolution.png')
```

### Compare with Reference

```python
from experiments_claude.analysis.compare_epochs_gaussian import compare_with_reference

# Side-by-side comparison table
compare_with_reference(results,
                      reference_experiment='reference_3blob_full_20260118_011103',
                      reference_epoch=20)
```

## Files Created

1. `mft_denoising/config.py` - Added `save_epoch_checkpoints` field
2. `mft_denoising/experiment.py` - Extended `log_epoch()` with checkpoint saving
3. `experiments_claude/analysis/compare_epochs_gaussian.py` - Per-epoch analysis
4. `experiments_claude/analysis/plot_gaussianity_trends.py` - Visualization tools
5. `experiments_claude/configs/diagnostic_demo.json` - Updated with checkpoint saving
6. `experiments_claude/configs/reference_3blob.json` - Updated with checkpoint saving and diagnostics

## Next Steps

1. **Wait for reference_3blob to complete** (~8 more minutes)
2. **Analyze epochs 14-20** focusing on outer clusters
3. **Generate log-scale tail plots** to visualize "becoming more Gaussian" hypothesis
4. **Compare with diagnostic_demo** as a "non-Gaussian reference"
5. **Document findings** about Gaussian-ness evolution and phase transitions

## Technical Notes

- Checkpoint files are ~4.1 MB each (for d=1024, hidden=512)
- Gaussian health analysis takes ~30-60 seconds per epoch (depends on cluster count)
- Results saved to `gaussian_health_per_epoch.json` in experiment directory
- All analysis uses reservoir sampling (50k points per cluster) for efficiency
- Histogram clustering used for initial cluster assignment (~500 bins instead of 500k points)
