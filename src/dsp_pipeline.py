"""
Digital Signal Processing (DSP) Preprocessing Pipeline for Audio Classification.

This module provides end-to-end DSP utilities:
- Audio loading with automatic resampling and mono-mixing
- Deterministic waveform length normalization (pad or trim)
- Short-Time Fourier Transform (STFT)
- Mel-filterbank projection (Mel Spectrogram)
- Decibel dynamic range compression (Log-Mel Spectrogram)
- Statistical and min-max normalization
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioConfig:
    """Configuration parameters for audio loading and DSP spectrogram extraction."""

    sample_rate: int = 22050
    duration: float = 2.0  # seconds
    n_fft: int = 1024
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = "hann"
    n_mels: int = 64
    f_min: float = 20.0
    f_max: Optional[float] = None  # None sets to Nyquist: sample_rate // 2
    power: float = 2.0  # 1.0 for energy, 2.0 for power
    top_db: float = 80.0
    normalize: bool = True
    norm_type: str = "standard"  # "standard" (zero-mean, unit-variance) or "minmax"

    def __post_init__(self):
        if self.win_length is None or self.win_length > self.n_fft:
            self.win_length = self.n_fft

    @property
    def target_samples(self) -> int:
        """Total number of audio samples corresponding to duration * sample_rate."""
        return int(self.sample_rate * self.duration)

    @property
    def expected_time_frames(self) -> int:
        """Calculates expected STFT/Mel time frames with librosa center=True padding."""
        # librosa default STFT pads with n_fft // 2 on both ends when center=True:
        # total padded length = target_samples + n_fft
        # frames = 1 + target_samples // hop_length
        return 1 + (self.target_samples // self.hop_length)


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 22050,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Loads an audio file, converts to mono if needed, and resamples to target_sr.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the target audio file (.wav, .flac, etc.).
    target_sr : Optional[int], default=22050
        Desired sampling frequency. If None, preserves original file sample rate.
    mono : bool, default=True
        Whether to average multi-channel signals to single-channel mono.

    Returns
    -------
    Tuple[np.ndarray, int]
        Tuple of (audio_waveform, sample_rate).
    """
    file_path = str(file_path)
    # librosa.load natively handles resampling and mono mixing
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return waveform.astype(np.float32), sr


def pad_or_trim(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Ensures the 1D waveform has exactly target_length samples by center-padding or trimming.

    Parameters
    ----------
    waveform : np.ndarray
        1D floating point audio signal.
    target_length : int
        Desired number of samples.

    Returns
    -------
    np.ndarray
        Padded or trimmed 1D array of length target_length.
    """
    current_length = len(waveform)
    if current_length == target_length:
        return waveform

    if current_length > target_length:
        # Center trim
        start = (current_length - target_length) // 2
        return waveform[start : start + target_length]

    # Center zero-padding
    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(waveform, (pad_left, pad_right), mode="constant")


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """Computes the Short-Time Fourier Transform (STFT) of an audio signal.

    Parameters
    ----------
    waveform : np.ndarray
        1D audio time series.
    n_fft : int
        FFT window size.
    hop_length : int
        Number of audio samples between adjacent STFT columns.
    win_length : Optional[int]
        Window length. If None, defaults to n_fft.
    window : str
        Window function specification (e.g. 'hann', 'hamming').
    center : bool
        If True, pads signal so frame t is centered at waveform[t * hop_length].

    Returns
    -------
    np.ndarray
        Complex-valued STFT matrix of shape (1 + n_fft // 2, time_steps).
    """
    if win_length is None:
        win_length = n_fft

    return librosa.stft(
        y=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )


def compute_mel_spectrogram(
    waveform: np.ndarray,
    config: AudioConfig,
) -> np.ndarray:
    """Extracts a linear power/energy Mel-scale spectrogram from a raw waveform.

    The continuous frequency axis of the STFT magnitude is mapped to the
    perceptually motivated Mel scale using triangular overlapping filterbanks:
    m = 2595 * log10(1 + f / 700)

    Parameters
    ----------
    waveform : np.ndarray
        1D audio waveform.
    config : AudioConfig
        DSP extraction parameters.

    Returns
    -------
    np.ndarray
        Mel spectrogram of shape (n_mels, time_steps).
    """
    win_len = (
        config.win_length
        if (config.win_length is not None and config.win_length <= config.n_fft)
        else config.n_fft
    )
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=win_len,
        window=config.window,
        n_mels=config.n_mels,
        fmin=config.f_min,
        fmax=config.f_max,
        power=config.power,
        center=True,
    )
    return mel_spec.astype(np.float32)


def power_to_db(
    spec: np.ndarray,
    top_db: float = 80.0,
    ref: Union[float, callable] = np.max,
    amin: float = 1e-10,
) -> np.ndarray:
    """Converts a power or energy spectrogram to decibel (dB) units.

    Logarithmic compression models the human perception of loudness (Weber-Fechner Law):
    dB = 10 * log10(S / ref)

    Parameters
    ----------
    spec : np.ndarray
        Input power spectrogram.
    top_db : float
        Threshold dynamic range below peak (clamps values < ref_db - top_db).
    ref : Union[float, callable]
        Reference value or reduction function (np.max).
    amin : float
        Minimum threshold to avoid division by zero or log of zero.

    Returns
    -------
    np.ndarray
        Decibel-scaled spectrogram in dB units.
    """
    return librosa.power_to_db(spec, ref=ref, amin=amin, top_db=top_db).astype(
        np.float32
    )


def normalize_spectrogram(
    spec: np.ndarray,
    norm_type: str = "standard",
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalizes spectrogram values to facilitate neural network convergence.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram array (typically in dB).
    norm_type : str
        'standard' for zero-mean, unit-variance.
        'minmax' for scaling into [-1, 1].
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    np.ndarray
        Normalized spectrogram array.
    """
    if norm_type == "standard":
        mean = np.mean(spec)
        std = np.std(spec)
        return ((spec - mean) / (std + eps)).astype(np.float32)
    elif norm_type == "minmax":
        s_min = np.min(spec)
        s_max = np.max(spec)
        if s_max - s_min < eps:
            return np.zeros_like(spec, dtype=np.float32)
        # Scale to [-1, 1]
        norm = 2.0 * (spec - s_min) / (s_max - s_min + eps) - 1.0
        return norm.astype(np.float32)
    else:
        raise ValueError(f"Unsupported norm_type '{norm_type}'. Choose 'standard' or 'minmax'.")


class AudioDSPPipeline:
    """End-to-End DSP Preprocessor for audio classification.

    Transforms raw audio (.wav, array) into normalized Log-Mel spectrograms:
    Raw Waveform -> Fixed Length Pad/Trim -> STFT -> Mel Filterbank -> Log (dB) -> Normalization.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def transform_waveform(self, waveform: np.ndarray, sr: Optional[int] = None) -> torch.Tensor:
        """Transforms a 1D numpy array into a 2D Log-Mel spectrogram PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Tensor of shape (1, n_mels, time_frames), ready for 2D CNN inference/training.
        """
        # Resample if waveform sample rate differs
        if sr is not None and sr != self.config.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.config.sample_rate)

        # Pad or trim to deterministic target duration
        waveform = pad_or_trim(waveform, self.config.target_samples)

        # Mel Spectrogram
        mel_spec = compute_mel_spectrogram(waveform, self.config)

        # Power to Decibels (Log compression)
        log_mel = power_to_db(mel_spec, top_db=self.config.top_db)

        # Normalization
        if self.config.normalize:
            log_mel = normalize_spectrogram(log_mel, norm_type=self.config.norm_type)

        # Add channel dimension: (1, n_mels, time_steps)
        tensor = torch.from_numpy(log_mel).unsqueeze(0)
        return tensor

    def process_file(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Loads an audio file from disk and transforms it into a Log-Mel tensor.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the target audio file.

        Returns
        -------
        torch.Tensor
            Tensor of shape (1, n_mels, time_frames).
        """
        waveform, sr = load_audio(file_path, target_sr=self.config.sample_rate, mono=True)
        return self.transform_waveform(waveform, sr=sr)


# Alias for flexible integration
MelSpectrogramExtractor = AudioDSPPipeline

