"""Unit tests for src/visualization.py."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from src.visualization import plot_dwt_coefficients, plot_scalogram, plot_signals


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    t = np.linspace(0, 1, 512, endpoint=False)
    clean = np.sin(2 * np.pi * 50 * t)
    noisy = clean + np.random.default_rng(0).normal(0, 0.2, len(t))
    denoised = clean + np.random.default_rng(1).normal(0, 0.05, len(t))
    return t, clean, noisy, denoised


class TestPlotSignals:
    def test_returns_figure(self, sample_data):
        t, clean, noisy, denoised = sample_data
        fig = plot_signals(t, noisy, denoised, clean)
        assert isinstance(fig, plt.Figure)

    def test_without_clean(self, sample_data):
        t, _, noisy, denoised = sample_data
        fig = plot_signals(t, noisy, denoised)
        assert isinstance(fig, plt.Figure)

    def test_three_subplots_with_clean(self, sample_data):
        t, clean, noisy, denoised = sample_data
        fig = plot_signals(t, noisy, denoised, clean)
        assert len(fig.axes) == 3

    def test_two_subplots_without_clean(self, sample_data):
        t, _, noisy, denoised = sample_data
        fig = plot_signals(t, noisy, denoised)
        assert len(fig.axes) == 2

    def test_axes_labels(self, sample_data):
        t, _, noisy, denoised = sample_data
        fig = plot_signals(t, noisy, denoised)
        # xlabel on last subplot, ylabel on all
        assert "Time" in fig.axes[-1].get_xlabel()
        assert "Amplitude" in fig.axes[0].get_ylabel()


class TestPlotScalogram:
    def test_returns_figure(self, sample_data):
        t, _, noisy, _ = sample_data
        fig = plot_scalogram(noisy, 512.0, wavelet="morl")
        assert isinstance(fig, plt.Figure)

    def test_fallback_wavelet(self, sample_data):
        t, _, noisy, _ = sample_data
        # db4 is a discrete wavelet; should fall back gracefully
        fig = plot_scalogram(noisy, 512.0, wavelet="db4")
        assert isinstance(fig, plt.Figure)

    def test_axes_labels(self, sample_data):
        t, _, noisy, _ = sample_data
        fig = plot_scalogram(noisy, 512.0)
        ax = fig.axes[0]
        assert "Time" in ax.get_xlabel()
        assert "Frequency" in ax.get_ylabel()


class TestPlotDwtCoefficients:
    def test_returns_figure(self):
        coeff_dict = {
            "approx": np.random.rand(64),
            "detail_1": np.random.rand(64),
            "detail_2": np.random.rand(128),
        }
        fig = plot_dwt_coefficients(coeff_dict, 1000.0)
        assert isinstance(fig, plt.Figure)

    def test_correct_subplot_count(self):
        coeff_dict = {"approx": np.ones(32), "detail_1": np.ones(32)}
        fig = plot_dwt_coefficients(coeff_dict, 1000.0)
        assert len(fig.axes) == 2
