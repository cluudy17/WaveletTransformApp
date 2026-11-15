"""Unit tests for src/signal_processing.py."""

import numpy as np
import pytest

from src.signal_processing import (
    _estimate_noise_std,
    _sure_loss,
    _sureshrink_threshold,
    _visushrink_threshold,
    apply_threshold,
    decompose,
    denoise,
    reconstruct,
)


@pytest.fixture
def clean_sine():
    t = np.linspace(0, 1, 1024, endpoint=False)
    return np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def noisy_sine(clean_sine):
    rng = np.random.default_rng(42)
    return clean_sine + rng.normal(0, 0.3, len(clean_sine))


class TestEstimateNoiseStd:
    def test_gaussian_noise(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.5, 2048)
        est = _estimate_noise_std(noise)
        assert abs(est - 0.5) < 0.1

    def test_zero_array(self):
        assert _estimate_noise_std(np.zeros(100)) == pytest.approx(0.0)


class TestVisuShrinkThreshold:
    def test_positive_result(self):
        assert _visushrink_threshold(1.0, 1024) > 0

    def test_scales_with_sigma(self):
        t1 = _visushrink_threshold(1.0, 1024)
        t2 = _visushrink_threshold(2.0, 1024)
        assert pytest.approx(t2) == 2 * t1

    def test_scales_with_n(self):
        t_small = _visushrink_threshold(1.0, 256)
        t_large = _visushrink_threshold(1.0, 1024)
        assert t_large > t_small


class TestSureLoss:
    def test_zero_threshold(self):
        coeffs = np.array([1.0, 2.0, -1.5, 0.5])
        loss = _sure_loss(coeffs, 0.0)
        assert np.isfinite(loss)

    def test_large_threshold_zeroes_all(self):
        coeffs = np.array([0.1, 0.2, 0.3])
        loss = _sure_loss(coeffs, 10.0)
        # all zeroed → loss = n - 2n + 0 = -n  (can be negative)
        assert np.isfinite(loss)


class TestSureShrinkThreshold:
    def test_returns_float(self):
        rng = np.random.default_rng(7)
        coeffs = rng.normal(0, 1, 256)
        t = _sureshrink_threshold(coeffs)
        assert isinstance(t, float)
        assert t >= 0


class TestApplyThreshold:
    def test_soft_small_values_zeroed(self):
        coeffs = np.array([0.5, -0.5, 2.0, -2.0])
        result = apply_threshold(coeffs, 1.0, "soft")
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(-1.0)

    def test_hard_small_values_zeroed(self):
        coeffs = np.array([0.5, -0.5, 2.0, -2.0])
        result = apply_threshold(coeffs, 1.0, "hard")
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(-2.0)


class TestDecomposeReconstruct:
    def test_perfect_reconstruction(self, clean_sine):
        coeffs = decompose(clean_sine, "db4", 4)
        reconstructed = reconstruct(coeffs, "db4", len(clean_sine))
        assert np.allclose(reconstructed, clean_sine, atol=1e-10)

    def test_coeffs_length(self, clean_sine):
        coeffs = decompose(clean_sine, "db4", 3)
        assert len(coeffs) == 4  # approx + 3 detail levels


class TestDenoise:
    def test_visushrink_soft(self, noisy_sine, clean_sine):
        denoised, coeff_dict = denoise(noisy_sine, "db4", 4, "soft", "visushrink")
        assert denoised.shape == noisy_sine.shape
        assert "approx" in coeff_dict
        assert "detail_1" in coeff_dict

    def test_sureshrink_hard(self, noisy_sine):
        denoised, _ = denoise(noisy_sine, "db4", 3, "hard", "sureshrink")
        assert denoised.shape == noisy_sine.shape

    def test_denoised_closer_to_clean(self):
        # Multi-freq signal so DWT detail levels carry real signal energy.
        # High noise (σ=0.8) ensures some coefficients are below threshold → thresholding helps.
        t = np.linspace(0, 1, 1024, endpoint=False)
        clean = (
            np.sin(2 * np.pi * 50 * t)
            + 0.5 * np.sin(2 * np.pi * 200 * t)
            + 0.3 * np.sin(2 * np.pi * 350 * t)
        )
        rng = np.random.default_rng(7)
        noisy = clean + rng.normal(0, 0.8, len(t))
        denoised, _ = denoise(noisy, "db4", 4, "soft", "sureshrink")
        err_noisy = np.mean((clean - noisy) ** 2)
        err_denoised = np.mean((clean - denoised) ** 2)
        assert err_denoised < err_noisy

    def test_invalid_level_raises(self, noisy_sine):
        with pytest.raises(ValueError):
            denoise(noisy_sine, "db4", 0, "soft", "visushrink")

    def test_2d_input_raises(self):
        with pytest.raises(ValueError):
            denoise(np.ones((10, 10)), "db4", 1, "soft", "visushrink")

    def test_level_too_high_raises(self, noisy_sine):
        with pytest.raises(ValueError):
            denoise(noisy_sine, "db4", 100, "soft", "visushrink")
