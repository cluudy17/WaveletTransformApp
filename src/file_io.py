"""File I/O for WAV and CSV signal files."""

import io
from typing import BinaryIO

import numpy as np
import pandas as pd
import soundfile as sf


def load_wav(file: str | BinaryIO) -> tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Returns (samples, sampling_rate). Multi-channel files are averaged to mono.
    """
    data, sr = sf.read(file, dtype="float64", always_2d=True)
    samples = data.mean(axis=1)
    return samples, float(sr)


def load_csv(file: str | BinaryIO) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Load a CSV signal file.

    Expected columns: 'signal' (required), 'time' (optional).
    Returns (samples, sampling_rate, time_axis_or_None).
    """
    df = pd.read_csv(file, encoding="utf-8")

    signal_col = None
    for candidate in ("signal", "amplitude", "value", "data"):
        if candidate in df.columns:
            signal_col = candidate
            break
    if signal_col is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("CSV contains no numeric columns")
        signal_col = numeric_cols[-1]  # last numeric column as signal

    samples = df[signal_col].to_numpy(dtype=np.float64)

    if "time" in df.columns:
        time = df["time"].to_numpy(dtype=np.float64)
        diffs = np.diff(time)
        sr = float(1.0 / np.mean(diffs)) if len(diffs) > 0 else 1000.0
    else:
        time = None
        sr = 1000.0

    return samples, sr, time


def save_wav(file: str, samples: np.ndarray, sampling_rate: float) -> None:
    """Write a mono WAV file normalised to [-1, 1]."""
    peak = np.max(np.abs(samples))
    normalised = samples / peak if peak > 0 else samples
    sf.write(file, normalised.astype(np.float32), int(sampling_rate))


def save_csv(file: str, samples: np.ndarray, sampling_rate: float) -> None:
    """Write signal samples as a two-column CSV (time, signal)."""
    n = len(samples)
    time = np.arange(n) / sampling_rate
    df = pd.DataFrame({"time": time, "signal": samples})
    df.to_csv(file, index=False)


def wav_bytes_to_array(uploaded_bytes: bytes) -> tuple[np.ndarray, float]:
    """Parse raw WAV bytes (from st.file_uploader) into (samples, sr)."""
    return load_wav(io.BytesIO(uploaded_bytes))


def csv_bytes_to_array(uploaded_bytes: bytes) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Parse raw CSV bytes (from st.file_uploader) into (samples, sr, time)."""
    return load_csv(io.BytesIO(uploaded_bytes))
