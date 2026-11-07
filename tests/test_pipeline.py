"""
Pytest Test Suite for Audio Event Classification DSP Pipeline & Deep Learning Components.

Verifies:
- Synthetic audio generation integrity (sample rates, finite values, envelope)
- Audio loading and resampling
- Waveform pad/trim edge cases
- STFT and Mel filterbank projection shapes
- Power-to-dB conversion & dynamic range clamping
- Spectrogram normalization (Standard vs MinMax)
- SpecAugment frequency & time masking
- PyTorch AudioEventDataset and DataLoader batch collation
- 2D CNN forward pass shapes, variable length adaptability, and gradient flow
- Single end-to-end optimization step
"""

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.dataset import AudioEventDataset, SpecAugment, create_dataloaders
from src.dsp_pipeline import (
    AudioConfig,
    AudioDSPPipeline,
    compute_mel_spectrogram,
    compute_stft,
    load_audio,
    normalize_spectrogram,
    pad_or_trim,
    power_to_db,
)
from src.model import AudioSpectrogramCNN
from tests.generate_mock import (
    generate_chirp_sweep,
    generate_chord,
    generate_high_whistle,
    generate_low_drone,
    generate_synthetic_dataset,
)


@pytest.fixture(scope="session")
def audio_config() -> AudioConfig:
    return AudioConfig(
        sample_rate=16000,
        duration=1.5,
        n_fft=512,
        hop_length=256,
        n_mels=48,
        top_db=80.0,
    )


@pytest.fixture(scope="session")
def mock_dataset_dir(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("mock_audio_data")
    generate_synthetic_dataset(
        output_dir=temp_dir,
        num_samples_per_class=4,
        sample_rate=16000,
        duration=1.5,
    )
    return temp_dir


# ============================================================================
# 1. Synthetic Generator Tests
# ============================================================================

def test_mock_audio_generation_shapes():
    sr = 22050
    dur = 1.0
    expected_samples = int(sr * dur)

    generators = [
        generate_low_drone,
        generate_high_whistle,
        generate_chirp_sweep,
        generate_chord,
    ]
    for gen in generators:
        signal = gen(duration=dur, sr=sr)
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float32
        assert len(signal) == expected_samples
        assert np.all(np.isfinite(signal)), "Signal must contain only finite float values"
        assert -1.5 <= np.max(signal) <= 1.5, "Signal should stay within reasonable amplitude"


def test_mock_wav_file_roundtrip(tmp_path):
    wav_path = tmp_path / "test_tone.wav"
    sr = 16000
    signal = generate_low_drone(duration=1.0, sr=sr)
    sf.write(str(wav_path), signal, sr, subtype="PCM_16")

    # Read back with soundfile
    data, read_sr = sf.read(str(wav_path))
    assert read_sr == sr
    assert len(data) == len(signal)
    assert np.allclose(data, signal, atol=1e-3)


# ============================================================================
# 2. DSP Pipeline Component Tests
# ============================================================================

def test_pad_or_trim():
    target_len = 1000

    # Shorter signal -> padding
    short_sig = np.ones(400, dtype=np.float32)
    padded = pad_or_trim(short_sig, target_len)
    assert len(padded) == target_len
    # Padding should be zero around center
    assert padded[0] == 0.0
    assert padded[-1] == 0.0
    assert padded[500] == 1.0

    # Longer signal -> trimming
    long_sig = np.ones(1600, dtype=np.float32)
    trimmed = pad_or_trim(long_sig, target_len)
    assert len(trimmed) == target_len

    # Exact length -> identity
    exact_sig = np.ones(target_len, dtype=np.float32)
    assert np.array_equal(pad_or_trim(exact_sig, target_len), exact_sig)


def test_stft_computation_shape(audio_config):
    signal = np.random.randn(audio_config.target_samples).astype(np.float32)
    stft = compute_stft(
        signal,
        n_fft=audio_config.n_fft,
        hop_length=audio_config.hop_length,
    )
    expected_freq_bins = 1 + audio_config.n_fft // 2
    expected_time_frames = audio_config.expected_time_frames

    assert np.iscomplexobj(stft), "STFT output must be complex numbers"
    assert stft.shape == (expected_freq_bins, expected_time_frames)


def test_mel_spectrogram_and_log_db(audio_config):
    signal = generate_chirp_sweep(duration=audio_config.duration, sr=audio_config.sample_rate)
    mel = compute_mel_spectrogram(signal, audio_config)

    assert mel.shape == (audio_config.n_mels, audio_config.expected_time_frames)
    assert np.all(mel >= 0.0), "Power spectrogram values must be non-negative"

    # Convert to dB
    log_mel = power_to_db(mel, top_db=audio_config.top_db)
    assert log_mel.shape == mel.shape
    assert np.all(np.isfinite(log_mel)), "Log-Mel spectrogram values must be finite"

    # Maximum is referenced to peak (0 dB), min is bounded by -top_db
    assert np.isclose(np.max(log_mel), 0.0, atol=1e-4)
    assert np.min(log_mel) >= -audio_config.top_db - 1e-4


def test_spectrogram_normalization():
    data = np.random.uniform(-80, 0, size=(64, 100)).astype(np.float32)

    # Standard normalization: zero mean, unit variance
    norm_std = normalize_spectrogram(data, norm_type="standard")
    assert math.isclose(float(np.mean(norm_std)), 0.0, abs_tol=1e-5)
    assert math.isclose(float(np.std(norm_std)), 1.0, abs_tol=1e-4)

    # Min-max normalization: scaled into [-1, 1]
    norm_minmax = normalize_spectrogram(data, norm_type="minmax")
    assert np.min(norm_minmax) >= -1.0 - 1e-5
    assert np.max(norm_minmax) <= 1.0 + 1e-5


def test_dsp_pipeline_end_to_end(audio_config):
    pipeline = AudioDSPPipeline(audio_config)
    waveform = generate_chord(duration=audio_config.duration, sr=audio_config.sample_rate)

    tensor = pipeline.transform_waveform(waveform, sr=audio_config.sample_rate)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 3  # (Channel, Freq, Time)
    assert tensor.shape == (1, audio_config.n_mels, audio_config.expected_time_frames)
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


# ============================================================================
# 3. Augmentation & Dataset Tests
# ============================================================================

def test_spec_augment():
    augmentor = SpecAugment(freq_mask_max=6, time_mask_max=10, mask_value=-999.0)
    spec = torch.ones((1, 32, 64), dtype=torch.float32)
    aug = augmentor(spec)

    assert aug.shape == spec.shape
    # Check that the mask value was injected
    assert (aug == -999.0).any()


def test_audio_event_dataset_and_loader(mock_dataset_dir, audio_config):
    dataset = AudioEventDataset(data_dir=mock_dataset_dir, config=audio_config, augment=True)
    assert len(dataset) == 16  # 4 classes * 4 samples
    assert len(dataset.class_to_idx) == 4

    # Test single item indexing
    spec, label = dataset[0]
    assert isinstance(spec, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert spec.shape == (1, audio_config.n_mels, audio_config.expected_time_frames)
    assert 0 <= label.item() < 4

    # Test create_dataloaders helper
    train_loader, val_loader, c2i, i2c = create_dataloaders(
        data_dir=mock_dataset_dir,
        batch_size=4,
        train_split=0.75,
        config=audio_config,
        seed=42,
    )
    assert len(c2i) == 4
    batch_specs, batch_labels = next(iter(train_loader))
    assert batch_specs.shape == (4, 1, audio_config.n_mels, audio_config.expected_time_frames)
    assert batch_labels.shape == (4,)


# ============================================================================
# 4. Neural Network Architecture Tests
# ============================================================================

def test_audio_cnn_forward_and_shapes(audio_config):
    num_classes = 4
    batch_size = 4
    model = AudioSpectrogramCNN(num_classes=num_classes, in_channels=1, base_channels=16)

    dummy_input = torch.randn(
        batch_size, 1, audio_config.n_mels, audio_config.expected_time_frames
    )
    logits = model(dummy_input)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()

    # Feature extraction check
    features = model.extract_features(dummy_input)
    assert features.shape[0] == batch_size
    assert features.ndim == 2


def test_audio_cnn_variable_time_steps():
    """Verifies that Adaptive Pooling allows processing varying audio lengths."""
    model = AudioSpectrogramCNN(num_classes=3, base_channels=16)
    model.eval()

    with torch.no_grad():
        out_short = model(torch.randn(2, 1, 64, 40))
        out_long = model(torch.randn(2, 1, 64, 150))

    assert out_short.shape == (2, 3)
    assert out_long.shape == (2, 3)


def test_gradient_flow_and_optimization_step():
    model = AudioSpectrogramCNN(num_classes=4, base_channels=16)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = torch.randn(2, 1, 48, 60, requires_grad=False)
    targets = torch.tensor([1, 3], dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Verify gradients are calculated and not NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN in {name}"

    initial_stem_weight = model.stem[0].weight.clone()
    optimizer.step()
    # Check weights actually changed
    assert not torch.allclose(initial_stem_weight, model.stem[0].weight)
