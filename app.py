# -*- coding: utf-8 -*-
"""Wavelet Denoising Application – main Streamlit entry point."""

import io
import sys
from pathlib import Path

# Force UTF-8 for stdout/stderr so Polish characters display correctly on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt
import streamlit as st

# Use a font with full Latin Extended-A support (covers ą ę ć ś ł ó ź ż ń)
matplotlib.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, str(Path(__file__).parent))

from src.file_io import csv_bytes_to_array, wav_bytes_to_array
from src.metrics import compute_metrics
from src.signal_generation import SIGNAL_GENERATORS
from src.signal_processing import denoise, denoise_wpd, fourier_denoise
from src.visualization import plot_dwt_coefficients, plot_scalogram, plot_signals, plot_wpd_coefficients

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wavelet Denoising",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Wavelet Denoising Research Station")
st.caption("Interactive analysis of DWT-based signal denoising algorithms")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # ── Data source ──────────────────────────────────────────────────────────
    st.subheader("Data Source")
    data_source = st.radio("Source", ["Signal generator", "Upload file"], horizontal=True)

    clean_signal: np.ndarray | None = None
    noisy_signal: np.ndarray | None = None
    sampling_rate: float = 1000.0

    if data_source == "Signal generator":
        signal_type = st.selectbox("Signal type", list(SIGNAL_GENERATORS.keys()))
        noise_level = st.slider("Noise level", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        duration = st.slider("Duration (s)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        sampling_rate = float(
            st.selectbox("Sampling rate (Hz)", [500, 1000, 2000, 4000], index=1)
        )

        simple_sine_key = "Sinus (pojedyncza częstotliwość)"
        sine_frequency = None
        if signal_type == simple_sine_key:
            sine_frequency = float(
                st.slider("Częstotliwość (Hz)", min_value=1, max_value=int(sampling_rate // 2), value=50)
            )

        @st.cache_data
        def _generate(sig_type: str, dur: float, sr: float, nl: float, freq: float | None):
            gen_fn = SIGNAL_GENERATORS[sig_type]
            if freq is not None:
                return gen_fn(duration=dur, sampling_rate=int(sr), noise_level=nl, frequency=freq)
            return gen_fn(duration=dur, sampling_rate=int(sr), noise_level=nl)

        time_axis, clean_signal, noisy_signal = _generate(
            signal_type, duration, sampling_rate, noise_level, sine_frequency
        )

    else:
        uploaded = st.file_uploader("Upload WAV or CSV", type=["wav", "csv"])
        if uploaded is not None:
            raw = uploaded.read()
            try:
                if uploaded.name.lower().endswith(".wav"):
                    loaded_signal, sampling_rate = wav_bytes_to_array(raw)
                else:
                    loaded_signal, sampling_rate, _t = csv_bytes_to_array(raw)
                time_axis = np.arange(len(loaded_signal)) / sampling_rate
                st.success(
                    f"Loaded {len(loaded_signal)} samples @ {sampling_rate:.0f} Hz"
                    f" ({len(loaded_signal)/sampling_rate:.2f} s)"
                )
                add_noise = st.checkbox("Add AWGN noise (semi-synthetic method)", value=True)
                if add_noise:
                    rms = float(np.sqrt(np.mean(loaded_signal ** 2)))
                    snr_target = st.slider("Input SNR [dB]", min_value=0, max_value=30, value=10, step=1)
                    sigma = rms / (10 ** (snr_target / 20))
                    st.caption(f"σ = {sigma:.5f}  (signal RMS = {rms:.5f})")
                    rng = np.random.default_rng(42)
                    clean_signal = loaded_signal.copy()
                    noisy_signal = loaded_signal + rng.normal(0.0, sigma, len(loaded_signal))
                else:
                    clean_signal = None
                    noisy_signal = loaded_signal
            except Exception as exc:
                st.error(f"Failed to load file: {exc}")
                noisy_signal = None
        else:
            st.info("Upload a WAV or CSV file to begin.")

    st.divider()

    # ── Wavelet parameters ────────────────────────────────────────────────────
    st.subheader("Wavelet Parameters")
    decomp_mode = st.radio(
        "Decomposition mode",
        ["DWT", "WPD"],
        horizontal=True,
        help="DWT: asymmetric tree (detail branches only). WPD: full binary tree (2ˡ leaf nodes).",
    )

    wavelet_families = ["haar", "db", "sym", "coif", "bior"]
    all_wavelets = sorted(
        [w for fam in wavelet_families for w in pywt.wavelist(fam)],
        key=lambda w: (w[:4], int("".join(filter(str.isdigit, w)) or "0")),
    )
    default_idx = all_wavelets.index("db4") if "db4" in all_wavelets else 0
    wavelet = st.selectbox("Mother wavelet", all_wavelets, index=default_idx)

    max_level_allowed = 10
    if noisy_signal is not None:
        max_level_allowed = min(10, pywt.dwt_max_level(len(noisy_signal), wavelet))
    decomp_level = st.slider(
        "Decomposition level", min_value=1, max_value=max_level_allowed, value=min(4, max_level_allowed)
    )

    st.divider()

    # ── Denoising parameters ──────────────────────────────────────────────────
    st.subheader("Denoising Parameters")
    threshold_method = st.radio(
        "Thresholding method", ["soft", "hard"], horizontal=True
    )
    threshold_algo = st.selectbox(
        "Threshold algorithm",
        ["visushrink", "sureshrink"],
        format_func=lambda x: "VisuShrink (universal)" if x == "visushrink" else "SureShrink (SURE-adaptive)",
    )

    st.divider()

    # ── Fourier comparison ────────────────────────────────────────────────────
    st.subheader("Fourier Comparison")
    fourier_enabled = st.checkbox("Enable Fourier filter", value=False)
    fourier_window = "rectangular"
    if fourier_enabled:
        fourier_window = st.radio(
            "Filter variant",
            ["rectangular", "hanning"],
            format_func=lambda x: "Rectangular (hard thresholding)" if x == "rectangular" else "Hanning (soft transition)",
            horizontal=True,
        )

    st.divider()

    # ── Visualization options ────────────────────────────────────────────────
    st.subheader("Visualization")
    show_scalogram = st.checkbox("Show scalograms", value=False)
    show_dwt_coeffs = st.checkbox("Show decomposition coefficients", value=False)

# ── Main canvas ───────────────────────────────────────────────────────────────
if noisy_signal is None:
    st.info("Configure a data source in the sidebar to begin analysis.")
    st.stop()

# Compute denoising
@st.cache_data
def _denoise(sig: bytes, wav: str, lvl: int, method: str, algo: str):
    arr = np.frombuffer(sig, dtype=np.float64).copy()
    return denoise(arr, wav, lvl, method, algo)  # type: ignore[arg-type]


@st.cache_data
def _denoise_wpd(sig: bytes, wav: str, lvl: int, method: str, algo: str):
    arr = np.frombuffer(sig, dtype=np.float64).copy()
    return denoise_wpd(arr, wav, lvl, method, algo)  # type: ignore[arg-type]


try:
    if decomp_mode == "DWT":
        denoised_signal, coeff_dict = _denoise(
            noisy_signal.tobytes(), wavelet, decomp_level, threshold_method, threshold_algo
        )
    else:
        denoised_signal, coeff_dict = _denoise_wpd(
            noisy_signal.tobytes(), wavelet, decomp_level, threshold_method, threshold_algo
        )
except ValueError as exc:
    st.error(f"Denoising error: {exc}")
    st.stop()

# Fourier denoising (optional)
fourier_denoised: np.ndarray | None = None
if fourier_enabled:
    @st.cache_data
    def _fourier_denoise(sig: bytes, window: str) -> np.ndarray:
        arr = np.frombuffer(sig, dtype=np.float64).copy()
        return fourier_denoise(arr, window)

    fourier_denoised = _fourier_denoise(noisy_signal.tobytes(), fourier_window)

# ── Metrics ───────────────────────────────────────────────────────────────────
if clean_signal is not None:
    m = compute_metrics(clean_signal, noisy_signal, denoised_signal)
    st.subheader("Quality Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SNR (noisy)", f"{m['snr_noisy']:.2f} dB")
    c2.metric(
        "SNR (denoised)",
        f"{m['snr_denoised']:.2f} dB",
        delta=f"{m['snr_improvement']:+.2f} dB",
    )
    c3.metric(
        "RMSE (denoised)",
        f"{m['rmse_denoised']:.4f}",
        delta=f"{m['rmse_improvement_percent']:+.1f}%",
        delta_color="inverse",
    )
    c4.metric("Correlation", f"{m['correlation']:.4f}")
else:
    st.info(
        "Clean reference not available (uploaded file). "
        "SNR/RMSE metrics require a reference signal."
    )

st.divider()

# ── Time-domain plot ──────────────────────────────────────────────────────────
st.subheader("Time-Domain Signals")
fig_signals = plot_signals(time_axis, noisy_signal, denoised_signal, clean_signal)
st.pyplot(fig_signals)
plt.close(fig_signals)

st.divider()

# ── Fourier comparison section ────────────────────────────────────────────────
if fourier_enabled and fourier_denoised is not None:
    st.subheader("Fourier Filter Comparison")
    variant_label = "rectangular" if fourier_window == "rectangular" else "Hanning"

    if clean_signal is not None:
        mf = compute_metrics(clean_signal, noisy_signal, fourier_denoised)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("SNR (noisy)", f"{mf['snr_noisy']:.2f} dB")
        col_b.metric(
            f"SNR Fourier/{variant_label}",
            f"{mf['snr_denoised']:.2f} dB",
            delta=f"{mf['snr_improvement']:+.2f} dB",
        )
        col_c.metric(
            "RMSE Fourier",
            f"{mf['rmse_denoised']:.4f}",
            delta=f"{mf['rmse_improvement_percent']:+.1f}%",
            delta_color="inverse",
        )
        col_d.metric("Correlation", f"{mf['correlation']:.4f}")

    # Overlay comparison plot: DWT vs Fourier
    fig_cmp, axes_cmp = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes_cmp[0].plot(time_axis, denoised_signal, color="#4CAF50", linewidth=1.2)
    axes_cmp[0].set_title(f"DWT ({decomp_mode}) / {threshold_algo} / {threshold_method}")
    axes_cmp[0].set_ylabel("Amplitude")
    axes_cmp[0].grid(True, alpha=0.3)
    axes_cmp[1].plot(time_axis, fourier_denoised, color="#FF9800", linewidth=1.2)
    axes_cmp[1].set_title(f"Fourier / {variant_label}")
    axes_cmp[1].set_ylabel("Amplitude")
    axes_cmp[1].set_xlabel("Time (s)")
    axes_cmp[1].grid(True, alpha=0.3)
    if clean_signal is not None:
        axes_cmp[0].plot(time_axis, clean_signal, color="#2196F3", linewidth=0.8, alpha=0.5, label="clean")
        axes_cmp[0].legend(fontsize=8)
        axes_cmp[1].plot(time_axis, clean_signal, color="#2196F3", linewidth=0.8, alpha=0.5, label="clean")
        axes_cmp[1].legend(fontsize=8)
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    st.divider()

# ── Scalograms ────────────────────────────────────────────────────────────────
if show_scalogram:
    st.subheader("Time-Frequency Analysis (Scalograms)")
    col_a, col_b = st.columns(2)

    # Limit signal length for CWT performance
    max_cwt_samples = 4000
    sig_for_cwt = noisy_signal[:max_cwt_samples]
    den_for_cwt = denoised_signal[:max_cwt_samples]
    sr_cwt = sampling_rate

    with col_a:
        st.caption("Noisy signal")
        with st.spinner("Computing scalogram…"):
            fig_sc_noisy = plot_scalogram(sig_for_cwt, sr_cwt, title="Noisy – Scalogram")
        st.pyplot(fig_sc_noisy)
        plt.close(fig_sc_noisy)

    with col_b:
        st.caption("Denoised signal")
        with st.spinner("Computing scalogram…"):
            fig_sc_den = plot_scalogram(den_for_cwt, sr_cwt, title="Denoised – Scalogram")
        st.pyplot(fig_sc_den)
        plt.close(fig_sc_den)

    st.divider()

# ── Decomposition coefficients ────────────────────────────────────────────────
if show_dwt_coeffs:
    if decomp_mode == "DWT":
        st.subheader("DWT Coefficients")
        fig_coeffs = plot_dwt_coefficients(coeff_dict, sampling_rate)
    else:
        st.subheader(f"WPD Coefficients ({2 ** decomp_level} leaf nodes)")
        fig_coeffs = plot_wpd_coefficients(coeff_dict, sampling_rate, decomp_level)
    st.pyplot(fig_coeffs)
    plt.close(fig_coeffs)
