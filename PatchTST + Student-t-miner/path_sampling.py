"""
Path sampling using Student-t distribution with optional latent volatility clustering
"""
import logging
import numpy as np
import torch
from scipy.stats import t as student_t

from .config import LATENT_VOL_PARAMS

logger = logging.getLogger(__name__)


def sample_student_t(nu: float, size: tuple) -> np.ndarray:
    """
    Sample from Student-t distribution
    
    Args:
        nu: Degrees of freedom
        size: Output shape
    Returns:
        Samples from Student-t(0, 1, nu)
    """
    # Use scipy for stable sampling
    return student_t.rvs(df=nu, size=size)


def apply_latent_volatility_clustering(
    log_sigma: np.ndarray, mode: str, num_simulations: int
) -> np.ndarray:
    """
    Apply AR(1) latent volatility clustering
    
    Args:
        log_sigma: [H] base log sigma
        mode: "HF", "LF-crypto", or "LF-equity"
        num_simulations: Number of paths
    Returns:
        log_sigma_prime: [N, H] adjusted log sigma per simulation
    """
    params = LATENT_VOL_PARAMS.get(mode, {"rho": 0.99, "s_vol": 0.10})
    rho = params["rho"]
    s_vol = params["s_vol"]
    
    H = len(log_sigma)
    
    # Generate latent AR(1) process z_t for each simulation
    z = np.zeros((num_simulations, H))
    for t in range(1, H):
        z[:, t] = rho * z[:, t - 1] + np.random.normal(0, s_vol, num_simulations)
    
    # Add to log_sigma
    log_sigma_expanded = np.tile(log_sigma, (num_simulations, 1))  # [N, H]
    log_sigma_prime = log_sigma_expanded + z
    
    return log_sigma_prime


def sample_paths(
    mu: np.ndarray,  # [H] mean log returns
    log_sigma: np.ndarray,  # [H] log standard deviation
    nu: float,  # Degrees of freedom
    num_simulations: int,
    anchor_price: float,
    mode: str = "LF-crypto",
    use_latent_vol: bool = True,
) -> np.ndarray:
    """
    Sample price paths from Student-t distribution
    
    Args:
        mu: [H] mean log returns per horizon step
        log_sigma: [H] log standard deviation per horizon step
        nu: Degrees of freedom
        num_simulations: Number of paths to generate
        anchor_price: Starting price
        mode: "HF", "LF-crypto", or "LF-equity"
        use_latent_vol: Whether to apply latent volatility clustering
    Returns:
        paths: [N, H+1] price paths (includes anchor at index 0)
    """
    H = len(mu)
    
    # Apply latent volatility clustering if enabled
    if use_latent_vol:
        log_sigma_prime = apply_latent_volatility_clustering(
            log_sigma, mode, num_simulations
        )  # [N, H]
    else:
        log_sigma_prime = np.tile(log_sigma, (num_simulations, 1))  # [N, H]
    
    sigma_prime = np.exp(log_sigma_prime)  # [N, H]
    
    # Expand mu to [N, H]
    mu_expanded = np.tile(mu, (num_simulations, 1))
    
    # Sample noise from Student-t
    eps = sample_student_t(nu, size=(num_simulations, H))  # [N, H]
    
    # Compute log returns
    log_returns = mu_expanded + sigma_prime * eps  # [N, H]
    
    # Convert to prices
    # P_t = P_{t-1} * exp(r_t)
    paths = np.zeros((num_simulations, H + 1))
    paths[:, 0] = anchor_price
    
    for t in range(1, H + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(log_returns[:, t - 1])
    
    # Ensure all prices are positive and finite
    paths = np.maximum(paths, 1e-8)  # Avoid zero/negative
    paths = np.where(np.isfinite(paths), paths, anchor_price)  # Replace inf/NaN
    
    return paths


def apply_sigma_calibration(
    log_sigma: np.ndarray, sigma_scale: float
) -> np.ndarray:
    """
    Apply calibration scaling to sigma
    
    Args:
        log_sigma: [H] log standard deviation
        sigma_scale: Calibration scale factor
    Returns:
        Adjusted log_sigma
    """
    return log_sigma + np.log(sigma_scale)
