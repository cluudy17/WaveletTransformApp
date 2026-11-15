"""Plotting utilities: time-domain signal comparison and scalograms."""

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pywt


def plot_signals(
    time: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot clean (if available), noisy, and denoised signals on separate subplots.

    Caller is responsible for calling plt.close(fig) to avoid memory leaks.
    """
    rows = 3 if clean is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.8 * rows), sharex=True)

    idx = 0
    if clean is not None:
        axes[idx].plot(time, clean, color="#2196F3", linewidth=1.2)
        axes[idx].set_title("Clean (reference)")
        axes[idx].set_ylabel("Amplitude")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    axes[idx].plot(time, noisy, color="#F44336", linewidth=0.8, alpha=0.85)
    axes[idx].set_title("Noisy")
    axes[idx].set_ylabel("Amplitude")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    axes[idx].plot(time, denoised, color="#4CAF50", linewidth=1.2)
    axes[idx].set_title("Denoised")
    axes[idx].set_ylabel("Amplitude")
    axes[idx].set_xlabel("Time (s)")
    axes[idx].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_scalogram(
    signal: np.ndarray,
    sampling_rate: float,
    wavelet: str = "morl",
    max_scales: int = 128,
    title: str = "Scalogram",
) -> matplotlib.figure.Figure:
    """
    Compute CWT and render a scalogram (time-frequency map).

    Caller is responsible for calling plt.close(fig) to avoid memory leaks.
    """
    scales = np.arange(1, max_scales + 1)
    try:
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1.0 / sampling_rate)
    except Exception:
        # Fallback: use 'morl' if the selected discrete wavelet is not CWT-compatible
        coefficients, frequencies = pywt.cwt(signal, scales, "morl", 1.0 / sampling_rate)

    energy = np.abs(coefficients) ** 2
    t = np.arange(len(signal)) / sampling_rate

    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(t, frequencies, energy, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_dwt_coefficients(
    coeff_dict: dict,
    sampling_rate: float,
) -> matplotlib.figure.Figure:
    """
    Plot DWT approximation and detail coefficients per level.

    Caller is responsible for calling plt.close(fig) to avoid memory leaks.
    """
    keys = list(coeff_dict.keys())
    n_plots = len(keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        coeffs = coeff_dict[key]
        t = np.arange(len(coeffs)) / sampling_rate
        ax.plot(t, coeffs, linewidth=0.8, color="#555555")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("DWT Coefficients")
    fig.tight_layout()
    return fig


def plot_wpd_coefficients(
    coeff_dict: dict,
    sampling_rate: float,
    level: int,
) -> matplotlib.figure.Figure:
    """
    Display WPD leaf-node coefficients as a frequency-ordered heatmap.

    Rows (top→bottom) correspond to nodes returned by get_level(level, 'freq'),
    i.e. low-frequency to high-frequency sub-bands.
    Caller is responsible for calling plt.close(fig) to avoid memory leaks.
    """
    paths = list(coeff_dict.keys())
    if not paths:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No WPD coefficients", ha="center", va="center")
        return fig

    n_nodes = len(paths)
    node_len = len(coeff_dict[paths[0]])
    matrix = np.array([coeff_dict[p] for p in paths])  # shape (2^level, node_len)

    vmax = float(np.max(np.abs(matrix))) or 1.0
    t_end = node_len * n_nodes / sampling_rate  # approximate

    fig, ax = plt.subplots(figsize=(10, max(3, n_nodes * 0.35 + 1.5)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        extent=[0.0, t_end, n_nodes - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Coefficient value")
    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels(paths, fontsize=max(5, 9 - n_nodes // 4))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("WP node path (freq order, low→high)")
    ax.set_title(f"WPD Coefficients — level {level}  ({n_nodes} leaf nodes)")
    fig.tight_layout()
    return fig
