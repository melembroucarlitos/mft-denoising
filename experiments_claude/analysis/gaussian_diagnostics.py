"""
Per-cluster Gaussian-ness diagnostics for weight pair analysis.

Tests whether points within each cluster follow multivariate Gaussian
distribution with arbitrary 2×2 covariance structure.

Based on multi-faceted statistical tests:
- Radial distribution (KS test vs χ²₂)
- Tail behavior (empirical vs theoretical)
- Angular uniformity (Rayleigh R statistic)
- 1D projections (Anderson-Darling)
- Mardia multivariate kurtosis
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class RunningCov2D:
    """
    Online mean and covariance computation using Welford's algorithm.
    
    Numerically stable computation of mean and covariance matrix for
    2D data without storing all points.
    """
    n: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    S: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    
    def update(self, x: np.ndarray):
        """Update with new 2D point."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.S += np.outer(delta, delta2)
    
    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return final mean and covariance."""
        if self.n <= 1:
            return self.mean, np.full((2, 2), np.nan)
        Sigma = self.S / (self.n - 1)
        return self.mean, Sigma


def reservoir_sample(
    cluster_points: List[np.ndarray],
    max_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Uniform reservoir sampling without replacement.
    
    Maintains uniform sample of up to max_size points from a stream.
    """
    if len(cluster_points) <= max_size:
        return np.array(cluster_points)
    
    # Reservoir sampling algorithm
    reservoir = cluster_points[:max_size].copy()
    for i in range(max_size, len(cluster_points)):
        j = rng.integers(0, i + 1)
        if j < max_size:
            reservoir[j] = cluster_points[i]
    
    return np.array(reservoir)


def whitening_matrix_2x2(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute whitening matrix W such that W·Σ·W^T = I.
    
    Uses eigendecomposition with regularization for numerical stability.
    
    Args:
        Sigma: (2, 2) covariance matrix
        eps: Regularization added to eigenvalues
    
    Returns:
        W: (2, 2) whitening matrix
    """
    # Regularize
    Sigma_reg = Sigma + eps * np.eye(2)
    
    # Eigendecomposition (symmetric matrix)
    lambdas, Q = np.linalg.eigh(Sigma_reg)
    
    # Whitening: W = diag(1/sqrt(λ)) · Q^T
    W = np.diag(1.0 / np.sqrt(lambdas)) @ Q.T
    
    return W


def chi2_df2_cdf(t: float) -> float:
    """
    CDF of chi-square distribution with df=2.
    
    For df=2, this is: F(t) = 1 - exp(-t/2)
    """
    if t < 0:
        return 0.0
    return 1.0 - np.exp(-t / 2.0)


def ks_distance(values: np.ndarray, ref_cdf_func) -> float:
    """
    Kolmogorov-Smirnov distance between empirical CDF and reference CDF.
    
    Returns supremum of |F_empirical(x) - F_reference(x)|.
    """
    values_sorted = np.sort(values)
    n = len(values_sorted)
    
    D = 0.0
    for i in range(n):
        x = values_sorted[i]
        F_emp = (i + 1) / n
        F_ref = ref_cdf_func(x)
        D = max(D, abs(F_emp - F_ref))
    
    return D


def tail_excess(
    d2_values: np.ndarray,
    thresholds: List[float] = [6.0, 10.0, 14.0]
) -> Dict[str, float]:
    """
    Compare empirical tail probabilities to χ²₂ reference.
    
    tail_ratio(t) = P_empirical(d² > t) / P_chi2(d² > t)
    
    Values >> 1 suggest heavy tails.
    """
    n = len(d2_values)
    ratios = {}
    
    for t in thresholds:
        emp_tail = np.sum(d2_values > t) / n
        ref_tail = np.exp(-t / 2.0)  # For χ²₂
        ratios[str(int(t))] = float(emp_tail / max(ref_tail, 1e-12))
    
    return ratios


def rayleigh_R(thetas: np.ndarray) -> float:
    """
    Rayleigh R statistic for circular uniformity.
    
    R = |Σ exp(iθ)| / n
    
    For uniform angles, R ~ O(1/√n). Large R indicates directional bias.
    """
    sx = np.sum(np.cos(thetas))
    sy = np.sum(np.sin(thetas))
    n = len(thetas)
    return np.sqrt(sx**2 + sy**2) / n


def anderson_darling_normal(z_values: np.ndarray) -> float:
    """
    Anderson-Darling test statistic for normality (N(0,1)).
    
    A² = -n - (1/n) Σ (2i-1)[log Φ(z_i) + log(1-Φ(z_{n+1-i}))]
    
    where Φ is the standard normal CDF.
    
    Returns A² as effect size (not p-value).
    """
    z_sorted = np.sort(z_values)
    n = len(z_sorted)
    
    # Use scipy for reliable normal CDF
    Phi = stats.norm.cdf(z_sorted)
    
    # Clamp to avoid log(0)
    Phi = np.clip(Phi, 1e-15, 1 - 1e-15)
    
    s = 0.0
    for i in range(n):
        Fi = Phi[i]
        Fj = Phi[n - 1 - i]
        s += (2 * (i + 1) - 1) * (np.log(Fi) + np.log(1 - Fj))
    
    A2 = -n - s / n
    return A2


def gaussian_health_checks(
    points: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    sample_size: int = 50000,
    n_projections: int = 20,
    eps: float = 1e-6,
    random_seed: int = 42
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-cluster Gaussian-ness diagnostics.
    
    Args:
        points: (N, 2) array of weight pairs
        labels: (N,) cluster assignments (-1 = noise)
        n_clusters: Number of clusters (excluding noise)
        sample_size: Max points per cluster to analyze (reservoir sampling)
        n_projections: Number of random 1D projections to test
        eps: Regularization for whitening
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping cluster_id → diagnostics:
        {
            cluster_id: {
                "n": int,
                "mu": [float, float],
                "Sigma": [[float, float], [float, float]],
                "radial_KS": float,
                "tail_ratios": {6: float, 10: float, 14: float},
                "angle_R": float,
                "proj_AD_median": float,
                "proj_AD_max": float,
                "mardia_kurtosis_dev": float
            }
        }
    """
    rng = np.random.default_rng(random_seed)
    results = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_points = points[cluster_mask]
        n_points = len(cluster_points)
        
        if n_points < 10:
            results[cluster_id] = {
                "n": n_points,
                "note": "too few points for reliable diagnostics"
            }
            continue
        
        # 1. Compute mean and covariance (streaming for efficiency)
        rc = RunningCov2D()
        for point in cluster_points:
            rc.update(point)
        mu, Sigma = rc.finalize()
        
        # 2. Whitening matrix
        try:
            W = whitening_matrix_2x2(Sigma, eps)
        except np.linalg.LinAlgError:
            results[cluster_id] = {
                "n": n_points,
                "mu": mu.tolist(),
                "Sigma": Sigma.tolist(),
                "note": "singular covariance matrix"
            }
            continue
        
        # 3. Reservoir sample for diagnostics
        if n_points > sample_size:
            # Reservoir sampling
            indices = rng.choice(n_points, size=sample_size, replace=False)
            sampled_points = cluster_points[indices]
            m = sample_size
        else:
            sampled_points = cluster_points
            m = n_points
        
        # 4. Whiten sampled points
        Y = np.array([W @ (x - mu) for x in sampled_points])
        
        # 5. Compute radial distances squared
        d2 = np.sum(Y**2, axis=1)
        
        # 6. Compute angles
        theta = np.arctan2(Y[:, 1], Y[:, 0])
        
        # 7. Radial KS test (d² should follow χ²₂)
        radial_KS = ks_distance(d2, chi2_df2_cdf)
        
        # 8. Tail excess ratios
        tail_ratios = tail_excess(d2)
        
        # 9. Angular uniformity (Rayleigh R)
        angle_R = rayleigh_R(theta)
        
        # 10. Random projection Anderson-Darling tests
        ADs = []
        for _ in range(n_projections):
            # Random unit vector in 2D
            angle = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(angle), np.sin(angle)])
            
            # Project whitened points
            z = Y @ u  # Should be ~ N(0,1)
            
            AD = anderson_darling_normal(z)
            ADs.append(AD)
        
        proj_AD_median = float(np.median(ADs))
        proj_AD_max = float(np.max(ADs))
        
        # 11. Mardia kurtosis deviation
        # b_{2,2} = E[(||y||²)²] should be 8 for 2D Gaussian
        b22 = np.mean(d2**2)
        mardia_kurtosis_dev = float(abs(b22 - 8.0))
        
        # Store results
        results[cluster_id] = {
            "n": int(n_points),
            "mu": mu.tolist(),
            "Sigma": Sigma.tolist(),
            "radial_KS": float(radial_KS),
            "tail_ratios": tail_ratios,
            "angle_R": float(angle_R),
            "proj_AD_median": proj_AD_median,
            "proj_AD_max": proj_AD_max,
            "mardia_kurtosis_dev": mardia_kurtosis_dev
        }
    
    return results


def assess_gaussian_quality(health: Dict[str, Any]) -> str:
    """
    Provide qualitative assessment of Gaussian-ness.
    
    Returns: "EXCELLENT", "GOOD", "MODERATE", "POOR", or "INSUFFICIENT DATA"
    """
    if "note" in health:
        return "INSUFFICIENT DATA"
    
    radial_KS = health["radial_KS"]
    angle_R = health["angle_R"]
    proj_AD = health["proj_AD_median"]
    
    # Scoring thresholds
    excellent = (radial_KS < 0.03 and angle_R < 0.015 and proj_AD < 0.5)
    good = (radial_KS < 0.05 and angle_R < 0.025 and proj_AD < 1.0)
    moderate = (radial_KS < 0.10 and angle_R < 0.05 and proj_AD < 2.0)
    
    if excellent:
        return "EXCELLENT"
    elif good:
        return "GOOD"
    elif moderate:
        return "MODERATE"
    else:
        return "POOR"
