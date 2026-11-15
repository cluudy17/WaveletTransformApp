"""Wavelet denoising core: DWT decomposition, VisuShrink, SureShrink, reconstruction."""

from typing import Literal
import numpy as np
import pywt


ThresholdingMethod = Literal["hard", "soft"]
ThresholdAlgorithm = Literal["visushrink", "sureshrink"]


def _estimate_noise_std(detail_coeffs: np.ndarray) -> float:
    """Estimate noise std from finest detail coefficients using MAD."""
    return float(np.median(np.abs(detail_coeffs)) / 0.6745)


def _visushrink_threshold(sigma: float, n: int) -> float:
    """Universal threshold: σ√(2 log N)."""
    return float(sigma * np.sqrt(2.0 * np.log(n)))


def _sure_loss(coeffs: np.ndarray, threshold: float) -> float:
    """
    SURE risk estimate for soft thresholding.

    Formula (Donoho & Johnstone 1995, Stein 1981, unit variance):
        SURE(λ) = n - 2·#{|xi| ≤ λ} + Σ min(xi², λ²)
    Derived via: SURE = ||g(y)-y||² + 2·div(g) - n
    where ||g-y||² = Σ min(xi², λ²) and div = #{|xi| > λ} = n - count_below.
    """
    n = len(coeffs)
    count_below = int(np.sum(np.abs(coeffs) <= threshold))
    return float(n - 2 * count_below + np.sum(np.minimum(coeffs ** 2, threshold ** 2)))


def _sureshrink_threshold(coeffs: np.ndarray, sigma: float = 1.0, n_candidates: int = 150) -> float:
    """SURE-minimising threshold for one set of detail coefficients.

    Normalises coefficients by sigma before SURE minimisation (unit-variance
    formula) and scales the result back, so the returned threshold is in the
    original coefficient units.
    """
    if sigma <= 0:
        sigma = 1.0
    normalized = coeffs / sigma
    abs_norm = np.abs(normalized)
    candidates = np.linspace(0.0, float(abs_norm.max()), n_candidates)
    losses = [_sure_loss(normalized, t) for t in candidates]
    return float(candidates[int(np.argmin(losses))]) * sigma


def apply_threshold(coeffs: np.ndarray, threshold: float, method: ThresholdingMethod) -> np.ndarray:
    """Apply hard or soft thresholding to a coefficient array."""
    return pywt.threshold(coeffs, threshold, mode=method)


def decompose(signal: np.ndarray, wavelet: str, level: int) -> list:
    """Return DWT coefficient list [cA_n, cD_n, ..., cD_1]."""
    return pywt.wavedec(signal, wavelet, level=level)


def reconstruct(coeffs: list, wavelet: str, original_length: int) -> np.ndarray:
    """Reconstruct signal from DWT coefficient list and trim to original length."""
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed[:original_length]


def denoise(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    method: ThresholdingMethod,
    algorithm: ThresholdAlgorithm,
) -> tuple[np.ndarray, dict]:
    """
    Denoise signal via DWT thresholding.

    Returns (denoised_signal, dwt_coefficients).
    dwt_coefficients keys: 'approx' and 'detail_1' … 'detail_<level>'.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be a 1-D array")
    if level < 1:
        raise ValueError("decomposition level must be >= 1")
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    if level > max_level:
        raise ValueError(f"decomposition level {level} exceeds maximum {max_level} for this signal/wavelet")

    coeffs = decompose(signal, wavelet, level)
    # coeffs[0] = approximation, coeffs[1..] = detail levels (finest last→index level)
    # pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # finest detail is coeffs[-1] (cD_1)
    finest_detail = coeffs[-1]
    sigma = _estimate_noise_std(finest_detail)

    thresholded = [coeffs[0]]  # keep approximation unchanged
    for i, detail in enumerate(coeffs[1:], start=1):
        if algorithm == "visushrink":
            threshold = _visushrink_threshold(sigma, len(signal))
        else:
            threshold = _sureshrink_threshold(detail, sigma)
        thresholded.append(apply_threshold(detail, threshold, method))

    denoised = reconstruct(thresholded, wavelet, len(signal))

    coeff_dict: dict = {"approx": coeffs[0]}
    for i, detail in enumerate(coeffs[1:], start=1):
        coeff_dict[f"detail_{i}"] = detail

    return denoised, coeff_dict


def fourier_denoise(signal: np.ndarray, window: str = "rectangular") -> np.ndarray:
    """Denoise via FFT amplitude thresholding with a VisuShrink-analogue threshold.

    For white noise with std σ, each rfft bin has amplitude ≈ σ√N, so the
    universal threshold σ√(2N logN) zeros most noise bins while preserving
    high-amplitude signal peaks.

    window:
      'rectangular' — hard threshold: bins below threshold are zeroed (ideal
                       brick-wall filter).
      'hanning'     — soft threshold with a raised-cosine transition zone
                       [0.5·T, T] in magnitude space; analogous to Hanning-
                       windowed FIR filter design (smoothed transition band,
                       reduced spectral ringing), no time-domain edge artefacts.
    """
    n = len(signal)
    # MAD of first differences estimates σ without requiring DWT
    sigma = float(np.median(np.abs(np.diff(signal))) / (0.6745 * np.sqrt(2)))

    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)
    threshold = sigma * np.sqrt(2.0 * n * np.log(n))

    if window == "rectangular":
        spectrum[magnitudes < threshold] = 0.0
    else:  # "hanning" — raised-cosine transition zone around the threshold
        t_low = 0.5 * threshold
        weights = np.where(
            magnitudes >= threshold, 1.0,
            np.where(
                magnitudes <= t_low, 0.0,
                0.5 * (1.0 - np.cos(np.pi * (magnitudes - t_low) / t_low)),
            ),
        )
        spectrum = spectrum * weights

    return np.fft.irfft(spectrum, n)[:n]


def denoise_wpd(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    method: ThresholdingMethod,
    algorithm: ThresholdAlgorithm,
) -> tuple[np.ndarray, dict]:
    """
    Denoise signal via Wavelet Packet Decomposition (WPD).

    Thresholds all 2^level leaf nodes in the full binary packet tree,
    vs DWT which thresholds only the detail branch at each level.
    Returns (denoised_signal, wpd_coeff_dict).
    wpd_coeff_dict maps node path → original (pre-threshold) coefficient array.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be a 1-D array")
    if level < 1:
        raise ValueError("decomposition level must be >= 1")
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    if level > max_level:
        raise ValueError(
            f"decomposition level {level} exceeds maximum {max_level} for this signal/wavelet"
        )

    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="periodization", maxlevel=level)
    sigma = _estimate_noise_std(wp["d"].data)  # level-1 detail node for noise std estimate

    leaves = wp.get_level(level, "freq")  # 2^level nodes, frequency-ordered low→high
    coeff_dict: dict = {}
    for node in leaves:
        coeff_dict[node.path] = node.data.copy()  # original coefficients for visualisation
        if algorithm == "visushrink":
            threshold = _visushrink_threshold(sigma, len(signal))
        else:
            threshold = _sureshrink_threshold(node.data, sigma)
        node.data = apply_threshold(node.data, threshold, method)

    denoised = wp.reconstruct(update=False)[: len(signal)]
    return denoised, coeff_dict
