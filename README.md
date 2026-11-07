# DSP Audio Event Classifier

An end-to-end Audio Event Classification system combining **Digital Signal Processing (DSP)** and **Deep Learning** built with Python, PyTorch, and Librosa.

---

## 📌 Architecture Overview

```text
dsp-audio-classifier/
├── src/
│   ├── __init__.py
│   ├── dsp_pipeline.py    # Audio loading, STFT, Log-Mel spectrogram extraction
│   ├── dataset.py         # PyTorch Dataset supporting spectrogram generation & SpecAugment
│   ├── model.py           # 2D CNN architecture tailored for spectrograms (SE + Residuals)
│   └── train.py           # Complete training and evaluation loop with MPS/CUDA support
├── tests/
│   ├── __init__.py
│   ├── generate_mock.py   # Generates synthetic multi-frequency sine wave .wav files
│   └── test_pipeline.py   # Pytest suite verifying shapes, sample rates, and tensor values
├── requirements.txt
└── README.md
```

---

## 🔬 DSP Theory & Mathematical Formulation

### 1. Sampling & Discrete Signals
Continuous acoustic pressure waves $x(t)$ are digitized through uniform sampling at rate $f_s$ (samples/sec):
$$x[n] = x\left(\frac{n}{f_s}\right), \quad n \in \mathbb{Z}$$
To prevent aliasing according to the **Nyquist-Shannon Sampling Theorem**, the sampling rate must exceed twice the highest frequency component:
$$f_s > 2 f_{\max}$$

### 2. Short-Time Fourier Transform (STFT)
Non-stationary acoustic signals change frequency characteristics over time. The STFT decomposes signals into localized time-frequency frames using a sliding window $w[n]$ (default: Hann window):
$$X[m, k] = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - m \cdot H] \cdot e^{-j \frac{2\pi}{N} k n}$$
where:
- $m$: Time frame index
- $k$: Discrete frequency bin index ($k \in [0, N/2]$)
- $N$: FFT window size (`n_fft`)
- $H$: Hop length (`hop_length`, number of samples between successive windows)
- $w[n]$: Hann window attenuating spectral leakage:
  $$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N - 1}\right)\right)$$

The **linear power spectrogram** represents energy distribution:
$$P[m, k] = |X[m, k]|^2$$

### 3. Mel-Scale Filterbank Projection
Human auditory perception of pitch is non-linear; the ear is significantly more sensitive to fine pitch differences at low frequencies than at high frequencies. The Mel scale models this psychophysical response:
$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

We project the linear frequency bins $k$ onto $M$ triangular overlapping bandpass filters $H_m[k]$:
$$S[m, b] = \sum_{k=0}^{N/2} P[m, k] \cdot H_b[k], \quad b \in [0, M-1]$$
where $M$ is the number of Mel frequency channels (`n_mels`).

### 4. Logarithmic Decibel Compression
Perceived loudness follows the Weber-Fechner Law (logarithmic response):
$$S_{\text{dB}}[m, b] = 10 \cdot \log_{10}\left(\frac{S[m, b]}{\max(S) + \epsilon}\right)$$
Dynamic range is clamped to `top_db` (default: 80 dB) to suppress numerical floor noise.

### 5. Spectrogram Standardization
To ensure stable gradient descent, spectrograms are standardized:
$$z[m, b] = \frac{S_{\text{dB}}[m, b] - \mu}{\sigma + \epsilon}$$

---

## 🧠 Neural Network Architecture (`AudioSpectrogramCNN`)

Unlike standard computer vision images, audio spectrograms are **anisotropic**:
- **Vertical Axis (Frequency)**: Governed by physical harmonics ($f_0, 2f_0, 3f_0$). Shifting vertically transposes musical pitch.
- **Horizontal Axis (Time)**: Governed by temporal envelope, onset, duration, and decay. Shifting horizontally represents time latency.

### Key Architectural Elements
1. **Stem Convolution**: Initial $3 \times 3$ receptive field mapping single-channel spectrograms to base feature channels.
2. **Residual ConvBlocks with Squeeze-and-Excitation (SE)**:
   - Residual shortcuts prevent gradient attenuation.
   - Squeeze-and-Excitation dynamically re-weights channels based on global spectral dependencies, capturing complex harmonic patterns.
3. **Dual Global Pooling (Adaptive Avg + Max)**:
   - Projects variable-length spectrograms `(Batch, C, F, T)` down to fixed-size vectors `(Batch, 2C)`.
   - Makes the model fully invariant to input audio duration.
4. **Regularization**:
   - **SpecAugment**: Random frequency and time band masking.
   - Spatial Dropout2d and Linear Dropout.
   - Label smoothing in Cross-Entropy loss.

---

## 🚀 Quickstart

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Synthesizes 4 distinct multi-frequency acoustic event categories:
- `sine_low`: Low-frequency harmonic drone ($120\text{--}180\text{ Hz}$ base + harmonics)
- `sine_high`: High-frequency whistle with frequency vibrato ($2500\text{--}3200\text{ Hz}$)
- `chirp_sweep`: Continuous wideband frequency sweep ($300\text{ Hz} \to 4500\text{ Hz}$)
- `multi_tone_chord`: Harmonic triad chord (e.g., A major $440, 554.37, 659.25\text{ Hz}$)

```bash
python -m tests.generate_mock --output-dir data/mock_dataset --num-samples 25 --sample-rate 22050 --duration 2.0
```

### 3. Run Pytest Suite
Executes unit tests verifying sample rates, STFT/Mel tensor shapes, dB dynamic ranges, normalization, SpecAugment, model forward pass, and backpropagation:
```bash
python -m pytest -v tests/test_pipeline.py
```

### 4. Train Audio Classifier
```bash
python -m src.train --data-dir data/mock_dataset --output-dir checkpoints --epochs 10 --batch-size 16 --lr 0.001
```

*Note: Automatically leverages Apple Silicon MPS or NVIDIA CUDA if available, falling back to CPU.*

### 5. Run Interactive Demo Application
```bash
# Run CLI demonstration across mock classes & generate diagnostic plots:
.venv/bin/python demo_app.py

# Test any custom audio file:
.venv/bin/python demo_app.py --audio data/mock_dataset/sample_0.wav

# Optional: Launch interactive local web UI (defaults to port 8555)
.venv/bin/python demo_app.py --server --port 8555
```

---

## 🛠️ Programmatic Usage

### DSP Pipeline
```python
from src.dsp_pipeline import AudioConfig, AudioDSPPipeline

config = AudioConfig(
    sample_rate=22050,
    duration=2.0,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
)

pipeline = AudioDSPPipeline(config)
spectrogram_tensor = pipeline.process_file("path/to/audio.wav")
# Shape: (1, 64, 87) -> (Channel, Mel Bins, Time Steps)
```

### PyTorch Model Forward Pass & Embeddings
```python
import torch
from src.model import AudioSpectrogramCNN

model = AudioSpectrogramCNN(num_classes=4, in_channels=1, base_channels=32)

x = torch.randn(8, 1, 64, 87)  # (Batch, Channel, Mels, Time)
logits = model(x)              # Shape: (8, 4)

# Extract 512-dim audio embedding representation
embeddings = model.extract_features(x)  # Shape: (8, 512)
```
