"""Quality metrics for denoising evaluation: SNR, RMSE, PSNR, correlation."""

import numpy as np


def snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB. Higher is better."""
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    if noise_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Root Mean Square Error. Lower is better."""
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def psnr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB for bounded signals. Higher is better."""
    peak = float(np.max(np.abs(clean)))
    mse_val = float(np.mean((clean - noisy) ** 2))
    if mse_val == 0.0:
        return float("inf")
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse_val))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient in [-1, 1]."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> dict:
    """
    Compute full denoising quality metrics.

    Returns a dict matching the Metrics TypedDict from data-model.md.
    """
    snr_noisy = snr(clean, noisy)
    snr_denoised = snr(clean, denoised)
    rmse_noisy = rmse(clean, noisy)
    rmse_denoised = rmse(clean, denoised)
    rmse_improvement = (
        float((rmse_noisy - rmse_denoised) / rmse_noisy * 100.0) if rmse_noisy > 0 else 0.0
    )
    return {
        "snr_noisy": snr_noisy,
        "snr_denoised": snr_denoised,
        "snr_improvement": snr_denoised - snr_noisy,
        "rmse_noisy": rmse_noisy,
        "rmse_denoised": rmse_denoised,
        "rmse_improvement_percent": rmse_improvement,
        "psnr": psnr(clean, denoised),
        "correlation": pearson_correlation(clean, denoised),
    }
