"""
Synthetic Multi-Frequency Audio Event Generator.

Generates mathematically defined, multi-frequency acoustic signals:
- sine_low: Low-frequency harmonic drone (fundamental + odd/even harmonics)
- sine_high: High-frequency whistling with subtle sinusoidal frequency vibrato
- chirp_sweep: Time-varying frequency chirp sweep (exponential/linear frequency modulation)
- multi_tone_chord: Harmonic multi-tone chord (triad musical frequencies)

Includes:
- Smooth Hann / Tukey / ADSR amplitude envelopes to avoid boundary clicking
- Additive Gaussian noise modeling realistic acoustic channel conditions
- Standard 16-bit PCM WAV export via soundfile
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def apply_envelope(signal: np.ndarray, attack_ratio: float = 0.05, release_ratio: float = 0.05) -> np.ndarray:
    """Applies a smooth linear/cosine rise and fall envelope to eliminate onset/offset transient clicks."""
    n = len(signal)
    attack_len = int(n * attack_ratio)
    release_len = int(n * release_ratio)

    envelope = np.ones(n, dtype=np.float32)
    if attack_len > 0:
        envelope[:attack_len] = 0.5 * (1.0 - np.cos(np.pi * np.arange(attack_len) / attack_len))
    if release_len > 0:
        envelope[-release_len:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(release_len) / release_len))

    return signal * envelope


def generate_low_drone(
    duration: float = 2.0,
    sr: int = 22050,
    base_freq: float = 140.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generates a low-frequency multi-harmonic drone (base + 2nd, 3rd, 4th harmonics)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + harmonics with 1/f spectral decay
    signal = (
        1.0 * np.sin(2 * np.pi * base_freq * t)
        + 0.5 * np.sin(2 * np.pi * 2 * base_freq * t)
        + 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)
        + 0.12 * np.sin(2 * np.pi * 4 * base_freq * t)
    )
    # Normalize peak amplitude and add subtle background noise
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    signal += np.random.normal(0, noise_level, size=len(signal))
    # Final peak headroom scaling to prevent 16-bit PCM integer clipping
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.95 * (signal / peak)
    return apply_envelope(signal).astype(np.float32)


def generate_high_whistle(
    duration: float = 2.0,
    sr: int = 22050,
    base_freq: float = 2800.0,
    vibrato_rate: float = 5.0,
    vibrato_depth: float = 40.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generates a high-frequency acoustic event with gentle frequency vibrato."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    instantaneous_freq = base_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    signal = np.sin(phase) + 0.3 * np.sin(2 * phase)
    signal = signal / np.max(np.abs(signal))
    signal += np.random.normal(0, noise_level, size=len(signal))
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.95 * (signal / peak)
    return apply_envelope(signal).astype(np.float32)


def generate_chirp_sweep(
    duration: float = 2.0,
    sr: int = 22050,
    f_start: float = 300.0,
    f_end: float = 4500.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generates a continuous wideband frequency sweep (chirp)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Instantaneous frequency: linear progression f(t) = f_start + (f_end - f_start) * (t / duration)
    # Phase integral: 2*pi * (f_start * t + 0.5 * beta * t^2)
    beta = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * beta * (t**2))
    signal = np.sin(phase)
    signal = signal / np.max(np.abs(signal))
    signal += np.random.normal(0, noise_level, size=len(signal))
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.95 * (signal / peak)
    return apply_envelope(signal).astype(np.float32)


def generate_chord(
    duration: float = 2.0,
    sr: int = 22050,
    frequencies: Tuple[float, ...] = (440.0, 554.37, 659.25),  # A major triad: A4, C#5, E5
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generates a multi-tone harmonic chord composed of distinct simultaneous frequencies."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    signal = signal / np.max(np.abs(signal))
    signal += np.random.normal(0, noise_level, size=len(signal))
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.95 * (signal / peak)
    return apply_envelope(signal).astype(np.float32)


GENERATOR_REGISTRY = {
    "sine_low": lambda dur, sr: generate_low_drone(
        duration=dur, sr=sr, base_freq=float(np.random.uniform(120, 180))
    ),
    "sine_high": lambda dur, sr: generate_high_whistle(
        duration=dur, sr=sr, base_freq=float(np.random.uniform(2500, 3200))
    ),
    "chirp_sweep": lambda dur, sr: generate_chirp_sweep(
        duration=dur,
        sr=sr,
        f_start=float(np.random.uniform(250, 400)),
        f_end=float(np.random.uniform(3500, 5000)),
    ),
    "multi_tone_chord": lambda dur, sr: generate_chord(
        duration=dur,
        sr=sr,
        frequencies=(
            float(np.random.choice([440.0, 523.25, 392.0])),
            float(np.random.choice([554.37, 659.25, 493.88])),
            float(np.random.choice([659.25, 783.99, 587.33])),
        ),
    ),
}


def generate_synthetic_dataset(
    output_dir: Path,
    num_samples_per_class: int = 15,
    sample_rate: int = 22050,
    duration: float = 2.0,
) -> Dict[str, List[Path]]:
    """Generates a complete dataset with class subdirectories containing WAV files.

    Returns
    -------
    Dict[str, List[Path]]
        Mapping of class_name -> list of generated WAV file paths.
    """
    output_dir = Path(output_dir)
    created_files: Dict[str, List[Path]] = {}

    for class_name, generator in GENERATOR_REGISTRY.items():
        class_folder = output_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        created_files[class_name] = []

        for i in range(num_samples_per_class):
            wav_data = generator(duration, sample_rate)
            file_path = class_folder / f"{class_name}_{i:03d}.wav"
            # Write 16-bit standard PCM WAV
            sf.write(str(file_path), wav_data, sample_rate, subtype="PCM_16")
            created_files[class_name].append(file_path)

    print(
        f"Generated {num_samples_per_class * len(GENERATOR_REGISTRY)} synthetic audio files "
        f"across {len(GENERATOR_REGISTRY)} classes in {output_dir}"
    )
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Audio Dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mock_dataset",
        help="Target folder to store synthetic WAV files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to generate per event class.",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sampling frequency.")
    parser.add_argument("--duration", type=float, default=2.0, help="Clip duration in seconds.")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    generate_synthetic_dataset(
        output_dir=out_path,
        num_samples_per_class=args.num_samples,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
