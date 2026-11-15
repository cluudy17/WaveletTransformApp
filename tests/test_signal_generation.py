"""Unit tests for src/signal_generation.py."""

import numpy as np
import pytest

from src.signal_generation import (
    SIGNAL_GENERATORS,
    _add_noise,
    generate_chirp_signal,
    generate_composite_signal,
    generate_rectangle_signal,
    generate_simple_sine_signal,
    generate_sine_signal,
    generate_sine_with_transients,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_n(duration: float, sampling_rate: int) -> int:
    return int(duration * sampling_rate)


# ---------------------------------------------------------------------------
# _add_noise
# ---------------------------------------------------------------------------

class TestAddNoise:
    def test_gaussian_output_shape(self):
        clean = np.zeros(256)
        noisy = _add_noise(clean, noise_level=0.5, noise_type="gaussian")
        assert noisy.shape == clean.shape

    def test_uniform_output_shape(self):
        clean = np.zeros(256)
        noisy = _add_noise(clean, noise_level=0.5, noise_type="uniform")
        assert noisy.shape == clean.shape

    def test_noise_is_added(self):
        clean = np.zeros(512)
        noisy = _add_noise(clean, noise_level=1.0, noise_type="gaussian")
        assert not np.allclose(noisy, clean)

    def test_zero_noise_level(self):
        clean = np.ones(128)
        noisy = _add_noise(clean, noise_level=0.0, noise_type="gaussian")
        assert np.allclose(noisy, clean)


# ---------------------------------------------------------------------------
# generate_simple_sine_signal
# ---------------------------------------------------------------------------

class TestGenerateSimpleSineSignal:
    def test_return_shape(self):
        t, clean, noisy = generate_simple_sine_signal(duration=1.0, sampling_rate=1000)
        n = _expected_n(1.0, 1000)
        assert len(t) == len(clean) == len(noisy) == n

    def test_time_starts_at_zero(self):
        t, _, _ = generate_simple_sine_signal()
        assert t[0] == pytest.approx(0.0)

    def test_time_step(self):
        sr = 500
        t, _, _ = generate_simple_sine_signal(sampling_rate=sr)
        assert np.diff(t).mean() == pytest.approx(1.0 / sr, rel=1e-6)

    def test_clean_amplitude(self):
        t, clean, _ = generate_simple_sine_signal(frequency=10.0, noise_level=0.0)
        assert np.max(np.abs(clean)) == pytest.approx(1.0, abs=0.01)

    def test_noisy_differs_from_clean(self):
        _, clean, noisy = generate_simple_sine_signal(noise_level=0.5)
        assert not np.allclose(clean, noisy)

    def test_dtype_float64(self):
        _, clean, noisy = generate_simple_sine_signal()
        assert clean.dtype == np.float64
        assert noisy.dtype == np.float64


# ---------------------------------------------------------------------------
# generate_sine_signal
# ---------------------------------------------------------------------------

class TestGenerateSineSignal:
    def test_return_shape(self):
        t, clean, noisy = generate_sine_signal(duration=2.0, sampling_rate=500)
        n = _expected_n(2.0, 500)
        assert len(t) == len(clean) == len(noisy) == n

    def test_default_frequencies(self):
        _, clean, _ = generate_sine_signal(noise_level=0.0)
        # Default: 3 frequencies normalised by len → amplitude <= 1
        assert np.max(np.abs(clean)) <= 1.0 + 1e-9

    def test_single_frequency_matches_simple_sine(self):
        sr, dur, f = 1000, 1.0, 100.0
        _, c_simple, _ = generate_simple_sine_signal(dur, sr, f, noise_level=0.0)
        _, c_multi, _ = generate_sine_signal(dur, sr, [f], noise_level=0.0)
        assert np.allclose(c_multi, c_simple, atol=1e-10)

    def test_custom_frequencies(self):
        t, clean, _ = generate_sine_signal(frequencies=[10.0, 20.0], noise_level=0.0)
        assert clean is not None


# ---------------------------------------------------------------------------
# generate_chirp_signal
# ---------------------------------------------------------------------------

class TestGenerateChirpSignal:
    def test_return_shape(self):
        t, clean, noisy = generate_chirp_signal(duration=1.0, sampling_rate=2000)
        n = _expected_n(1.0, 2000)
        assert len(t) == len(clean) == len(noisy) == n

    def test_amplitude_bounded(self):
        _, clean, _ = generate_chirp_signal(noise_level=0.0)
        assert np.max(np.abs(clean)) <= 1.0 + 1e-9

    def test_dtype_float64(self):
        _, clean, _ = generate_chirp_signal()
        assert clean.dtype == np.float64


# ---------------------------------------------------------------------------
# generate_composite_signal
# ---------------------------------------------------------------------------

class TestGenerateCompositeSignal:
    def test_return_shape(self):
        t, clean, noisy = generate_composite_signal(duration=1.0, sampling_rate=1000)
        n = _expected_n(1.0, 1000)
        assert len(t) == len(clean) == len(noisy) == n

    def test_transient_bump_present(self):
        _, clean, _ = generate_composite_signal(noise_level=0.0)
        mid = len(clean) // 2
        width = len(clean) // 10
        # Peak near middle must exceed sinusoidal baseline
        bump_region = np.abs(clean[mid - width : mid + width])
        outside = np.abs(clean[:mid - width])
        assert bump_region.mean() > outside.mean()

    def test_dtype_float64(self):
        _, clean, _ = generate_composite_signal()
        assert clean.dtype == np.float64


# ---------------------------------------------------------------------------
# generate_rectangle_signal
# ---------------------------------------------------------------------------

class TestGenerateRectangleSignal:
    def test_return_shape(self):
        t, clean, noisy = generate_rectangle_signal(duration=1.0, sampling_rate=1000)
        n = _expected_n(1.0, 1000)
        assert len(t) == len(clean) == len(noisy) == n

    def test_values_are_plus_minus_one(self):
        _, clean, _ = generate_rectangle_signal(noise_level=0.0)
        unique = np.unique(clean)
        assert set(unique).issubset({-1.0, 0.0, 1.0})

    def test_dtype_float64(self):
        _, clean, _ = generate_rectangle_signal()
        assert clean.dtype == np.float64


# ---------------------------------------------------------------------------
# generate_sine_with_transients
# ---------------------------------------------------------------------------

class TestGenerateSineWithTransients:
    def test_return_shape(self):
        t, clean, noisy = generate_sine_with_transients(duration=2.0, sampling_rate=1000)
        n = _expected_n(2.0, 1000)
        assert len(t) == len(clean) == len(noisy) == n

    def test_transients_visible(self):
        _, clean, _ = generate_sine_with_transients(
            duration=2.0, sampling_rate=1000, noise_level=0.0, n_transients=4
        )
        # Peak amplitude must clearly exceed sinusoidal envelope (0.6 + 0.4 = 1.0)
        assert np.max(np.abs(clean)) > 1.5

    def test_deterministic_with_fixed_seed(self):
        _, c1, _ = generate_sine_with_transients(noise_level=0.0, n_transients=3)
        _, c2, _ = generate_sine_with_transients(noise_level=0.0, n_transients=3)
        assert np.allclose(c1, c2)


# ---------------------------------------------------------------------------
# SIGNAL_GENERATORS dict
# ---------------------------------------------------------------------------

class TestSignalGenerators:
    def test_all_keys_present(self):
        expected = {
            "Sinus (pojedyncza częstotliwość)",
            "Sinus (wieloczęstotliwościowy)",
            "Chirp (sweep częstotliwości)",
            "Złożony (sinusy + transjenta)",
            "Prostokąt (sygnał skokowy)",
            "Sinus z transjentami",
        }
        assert expected == set(SIGNAL_GENERATORS.keys())

    def test_all_values_callable(self):
        for name, fn in SIGNAL_GENERATORS.items():
            assert callable(fn), f"{name!r} is not callable"

    def test_each_generator_returns_three_arrays(self):
        for name, fn in SIGNAL_GENERATORS.items():
            result = fn()
            assert len(result) == 3, f"{name!r} did not return 3 arrays"
            t, clean, noisy = result
            assert len(t) == len(clean) == len(noisy), f"{name!r}: length mismatch"
