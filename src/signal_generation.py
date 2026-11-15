# -*- coding: utf-8 -*-
"""Synthetic signal generation with controllable noise."""

import numpy as np


def _add_noise(clean: np.ndarray, noise_level: float, noise_type: str = "gaussian") -> np.ndarray:
    rng = np.random.default_rng()
    if noise_type == "gaussian":
        noise = rng.normal(0.0, noise_level, len(clean))
    else:
        noise = rng.uniform(-noise_level, noise_level, len(clean))
    return clean + noise


def generate_simple_sine_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    frequency: float = 50.0,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single-frequency sine signal.

    Returns (time, clean_signal, noisy_signal).
    """
    t = np.linspace(0.0, duration, int(duration * sampling_rate), endpoint=False)
    clean = np.sin(2.0 * np.pi * frequency * t).astype(np.float64)
    noisy = _add_noise(clean, noise_level)
    return t, clean, noisy


def generate_sine_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    frequencies: list[float] | None = None,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a multi-frequency sine signal.

    Returns (time, clean_signal, noisy_signal).
    """
    if frequencies is None:
        frequencies = [50.0, 120.0, 300.0]
    t = np.linspace(0.0, duration, int(duration * sampling_rate), endpoint=False)
    clean = sum(np.sin(2.0 * np.pi * f * t) for f in frequencies) / len(frequencies)
    clean = clean.astype(np.float64)
    noisy = _add_noise(clean, noise_level)
    return t, clean, noisy


def generate_chirp_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    f_start: float = 10.0,
    f_end: float = 400.0,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a linear chirp (frequency sweep) signal.

    Returns (time, clean_signal, noisy_signal).
    """
    from scipy.signal import chirp

    t = np.linspace(0.0, duration, int(duration * sampling_rate), endpoint=False)
    clean = chirp(t, f0=f_start, f1=f_end, t1=duration, method="linear").astype(np.float64)
    noisy = _add_noise(clean, noise_level)
    return t, clean, noisy


def generate_composite_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a composite signal: sinusoids + a step-like transient.

    Returns (time, clean_signal, noisy_signal).
    """
    t = np.linspace(0.0, duration, int(duration * sampling_rate), endpoint=False)
    clean = (
        0.5 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 150.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 300.0 * t)
    )
    # Add a smooth transient bump in the middle
    mid = len(t) // 2
    width = len(t) // 10
    bump = np.zeros_like(t)
    bump[mid - width : mid + width] = 0.5 * np.hanning(2 * width)
    clean = (clean + bump).astype(np.float64)
    noisy = _add_noise(clean, noise_level)
    return t, clean, noisy


def generate_rectangle_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    frequency: float = 5.0,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a square wave signal with sharp discontinuities.

    Ideal for testing Haar and biorthogonal wavelets.
    Returns (time, clean_signal, noisy_signal).
    """
    t = np.linspace(0.0, duration, int(duration * sampling_rate), endpoint=False)
    clean = np.sign(np.sin(2.0 * np.pi * frequency * t)).astype(np.float64)
    noisy = _add_noise(clean, noise_level)
    return t, clean, noisy


def generate_sine_with_transients(
    duration: float = 2.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    n_transients: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a multi-frequency sine with sharp transient spikes.

    Used to test thresholding stability (Scenario 3).
    Returns (time, clean_signal, noisy_signal).
    """
    t = np.linspace(0.0, duration, int(duration * sampling_rate), endpoint=False)
    clean = (
        0.6 * np.sin(2.0 * np.pi * 30.0 * t)
        + 0.4 * np.sin(2.0 * np.pi * 80.0 * t)
    ).astype(np.float64)
    rng = np.random.default_rng(42)
    positions = rng.integers(len(t) // 8, 7 * len(t) // 8, size=n_transients)
    spike_width = max(2, len(t) // 200)
    for pos in positions:
        start = max(0, pos - spike_width)
        end = min(len(t), pos + spike_width + 1)
        width = end - start
        clean[start:end] += 1.8 * np.hanning(width) * float(rng.choice([-1, 1]))
    noisy = _add_noise(clean, noise_level)
    return t, clean, noisy


SIGNAL_GENERATORS = {
    "Sinus (pojedyncza częstotliwość)": generate_simple_sine_signal,
    "Sinus (wieloczęstotliwościowy)": generate_sine_signal,
    "Chirp (sweep częstotliwości)": generate_chirp_signal,
    "Złożony (sinusy + transjenta)": generate_composite_signal,
    "Prostokąt (sygnał skokowy)": generate_rectangle_signal,
    "Sinus z transjentami": generate_sine_with_transients,
}
