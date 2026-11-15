"""Unit tests for src/file_io.py."""

import io
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.file_io import csv_bytes_to_array, load_csv, load_wav, save_csv, save_wav, wav_bytes_to_array


@pytest.fixture
def sample_signal():
    t = np.linspace(0, 1, 4410, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def wav_bytes(sample_signal):
    buf = io.BytesIO()
    sf.write(buf, sample_signal, 44100, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def csv_bytes(sample_signal):
    n = len(sample_signal)
    time = np.arange(n) / 44100.0
    df = pd.DataFrame({"time": time, "signal": sample_signal})
    return df.to_csv(index=False).encode()


class TestLoadWav:
    def test_loads_samples_and_sr(self, wav_bytes):
        buf = io.BytesIO(wav_bytes)
        samples, sr = load_wav(buf)
        assert sr == pytest.approx(44100.0)
        assert len(samples) == 4410
        assert samples.dtype == np.float64

    def test_multichannel_averaged(self):
        stereo = np.column_stack([np.ones(100), -np.ones(100)]).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, stereo, 22050, format="WAV", subtype="FLOAT")
        buf.seek(0)
        samples, sr = load_wav(buf)
        assert samples.ndim == 1
        assert np.allclose(samples, 0.0, atol=1e-6)


class TestLoadCsv:
    def test_with_time_column(self, csv_bytes):
        buf = io.BytesIO(csv_bytes)
        samples, sr, time = load_csv(buf)
        assert len(samples) == 4410
        assert sr == pytest.approx(44100.0, rel=0.01)
        assert time is not None

    def test_without_time_column(self):
        df = pd.DataFrame({"signal": np.random.rand(100)})
        raw = df.to_csv(index=False).encode()
        samples, sr, time = load_csv(io.BytesIO(raw))
        assert len(samples) == 100
        assert sr == pytest.approx(1000.0)
        assert time is None

    def test_no_numeric_column_raises(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        raw = df.to_csv(index=False).encode()
        with pytest.raises(ValueError):
            load_csv(io.BytesIO(raw))


class TestSaveLoadRoundtrip:
    def test_wav_roundtrip(self, sample_signal, tmp_path):
        path = str(tmp_path / "test.wav")
        save_wav(path, sample_signal.astype(np.float64), 44100.0)
        loaded, sr = load_wav(path)
        assert sr == pytest.approx(44100.0)
        # normalised on save, so shape should match
        assert len(loaded) == len(sample_signal)

    def test_csv_roundtrip(self, sample_signal, tmp_path):
        path = str(tmp_path / "test.csv")
        save_csv(path, sample_signal.astype(np.float64), 44100.0)
        loaded, sr, _ = load_csv(path)
        assert sr == pytest.approx(44100.0, rel=0.01)
        assert np.allclose(loaded, sample_signal, atol=1e-5)


class TestBytesHelpers:
    def test_wav_bytes_helper(self, wav_bytes):
        samples, sr = wav_bytes_to_array(wav_bytes)
        assert sr == pytest.approx(44100.0)
        assert samples.ndim == 1

    def test_csv_bytes_helper(self, csv_bytes):
        samples, sr, _ = csv_bytes_to_array(csv_bytes)
        assert len(samples) == 4410
