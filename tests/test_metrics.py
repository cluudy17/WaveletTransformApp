"""Unit tests for src/metrics.py."""

import numpy as np
import pytest

from src.metrics import compute_metrics, pearson_correlation, psnr, rmse, snr


@pytest.fixture
def clean():
    t = np.linspace(0, 1, 1024, endpoint=False)
    return np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def noisy(clean):
    rng = np.random.default_rng(0)
    return clean + rng.normal(0, 0.3, len(clean))


class TestSNR:
    def test_identical_signals_infinite(self, clean):
        assert snr(clean, clean) == float("inf")

    def test_positive_for_low_noise(self, clean, noisy):
        assert snr(clean, noisy) > 0

    def test_higher_snr_for_less_noise(self, clean):
        rng = np.random.default_rng(1)
        mild = clean + rng.normal(0, 0.1, len(clean))
        heavy = clean + rng.normal(0, 1.0, len(clean))
        assert snr(clean, mild) > snr(clean, heavy)


class TestRMSE:
    def test_identical_signals_zero(self, clean):
        assert rmse(clean, clean) == pytest.approx(0.0)

    def test_positive(self, clean, noisy):
        assert rmse(clean, noisy) > 0

    def test_scale_proportional(self, clean):
        rng = np.random.default_rng(2)
        n1 = clean + rng.normal(0, 0.1, len(clean))
        n2 = clean + rng.normal(0, 0.5, len(clean))
        assert rmse(clean, n2) > rmse(clean, n1)


class TestPSNR:
    def test_identical_signals_infinite(self, clean):
        assert psnr(clean, clean) == float("inf")

    def test_finite_for_noisy(self, clean, noisy):
        assert np.isfinite(psnr(clean, noisy))

    def test_higher_for_less_noise(self, clean):
        rng = np.random.default_rng(3)
        mild = clean + rng.normal(0, 0.05, len(clean))
        heavy = clean + rng.normal(0, 0.8, len(clean))
        assert psnr(clean, mild) > psnr(clean, heavy)


class TestCorrelation:
    def test_identical_is_one(self, clean):
        assert pearson_correlation(clean, clean) == pytest.approx(1.0)

    def test_opposite_is_minus_one(self, clean):
        assert pearson_correlation(clean, -clean) == pytest.approx(-1.0)

    def test_range(self, clean, noisy):
        c = pearson_correlation(clean, noisy)
        assert -1.0 <= c <= 1.0

    def test_constant_array(self):
        a = np.ones(100)
        b = np.ones(100)
        assert pearson_correlation(a, b) == pytest.approx(1.0)


class TestComputeMetrics:
    def test_returns_all_keys(self, clean, noisy):
        denoised = clean + np.random.default_rng(4).normal(0, 0.05, len(clean))
        m = compute_metrics(clean, noisy, denoised)
        expected_keys = {
            "snr_noisy", "snr_denoised", "snr_improvement",
            "rmse_noisy", "rmse_denoised", "rmse_improvement_percent",
            "psnr", "correlation",
        }
        assert expected_keys == set(m.keys())

    def test_improvement_positive_for_denoising(self, clean, noisy):
        rng = np.random.default_rng(5)
        denoised = clean + rng.normal(0, 0.05, len(clean))
        m = compute_metrics(clean, noisy, denoised)
        assert m["snr_improvement"] > 0
        assert m["rmse_improvement_percent"] > 0
