"""
Interactive Audio Event Classifier Demo Application & Web Dashboard.

Provides:
1. CLI Walkthrough mode:
   python demo_app.py
2. Single-file CLI classification:
   python demo_app.py --audio path/to/sample.wav
3. Modern Web Dashboard:
   python demo_app.py --server --port 8555
   - Drag-and-drop file upload (.wav, .mp3)
   - Real-time browser microphone recording (Web Audio PCM WAV encoder)
   - Full DSP Log-Mel spectrogram visualization
   - DSP feature extraction & Neural Network inference latency profiling
   - Preset test buttons for synthetic acoustic classes
"""

import argparse
import base64
import http.server
import io
import json
import socketserver
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless web server
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dsp_pipeline import AudioConfig, AudioDSPPipeline, MelSpectrogramExtractor, load_audio
from src.model import AudioCNN, AudioSpectrogramCNN


def get_default_device() -> torch.device:
    """Auto-detects best available compute hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, AudioConfig, Dict[int, str]]:
    """Loads trained checkpoint, restores audio config and class mappings."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found. Please train model first using src/train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct audio configuration
    if "audio_config" in checkpoint:
        cfg_dict = checkpoint["audio_config"]
        config = AudioConfig(
            sample_rate=cfg_dict.get("sample_rate", 22050),
            duration=cfg_dict.get("duration", 2.0),
            n_fft=cfg_dict.get("n_fft", 1024),
            hop_length=cfg_dict.get("hop_length", 512),
            win_length=cfg_dict.get("win_length", None),
            window=cfg_dict.get("window", "hann"),
            n_mels=cfg_dict.get("n_mels", 64),
            f_min=cfg_dict.get("f_min", 20.0),
            f_max=cfg_dict.get("f_max", None),
            top_db=cfg_dict.get("top_db", 80.0),
        )
    else:
        config = AudioConfig()

    idx_to_class: Dict[int, str] = {}
    if "idx_to_class" in checkpoint:
        idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    elif "class_to_idx" in checkpoint:
        idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    else:
        num_classes = checkpoint["model_state_dict"]["classifier.4.weight"].shape[0]
        idx_to_class = {i: f"Class_{i}" for i in range(num_classes)}

    num_classes = len(idx_to_class)
    model = AudioCNN(num_classes=num_classes)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, config, idx_to_class


def render_spectrogram_base64(
    spec: np.ndarray,
    title: str = "Log-Mel Spectrogram",
) -> str:
    """Renders a Log-Mel spectrogram as a dark-mode styled base64 PNG image."""
    fig, ax = plt.subplots(figsize=(7.5, 2.8), dpi=120)
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=10, fontweight="bold", color="#f8fafc", pad=8)
    ax.set_xlabel("Time Frames", fontsize=8, color="#94a3b8")
    ax.set_ylabel("Mel Frequency Bins", fontsize=8, color="#94a3b8")
    ax.tick_params(colors="#94a3b8", labelsize=8)

    # Dark mode theme styling
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    cbar.set_label("Magnitude (dB / Norm)", color="#94a3b8", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def render_ascii_bar(prob: float, length: int = 24) -> str:
    """Renders a visual horizontal probability progress bar for CLI."""
    filled = int(round(prob * length))
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {prob * 100:5.1f}%"


def predict_audio_waveform(
    waveform: np.ndarray,
    sr: int,
    file_name: str,
    model: torch.nn.Module,
    pipeline: AudioDSPPipeline,
    idx_to_class: Dict[int, str],
    device: torch.device,
) -> Dict:
    """Executes DSP extraction, Log-Mel spectrogram generation, and model inference."""
    # 1. DSP Preprocessing Pipeline
    t0 = time.perf_counter()
    dsp_tensor = pipeline.transform_waveform(waveform, sr=sr)  # (1, n_mels, time_steps)
    t_dsp = (time.perf_counter() - t0) * 1000.0

    # 2. Neural Network Forward Pass
    t1 = time.perf_counter()
    input_tensor = dsp_tensor.unsqueeze(0).to(device)  # (1, 1, n_mels, time_steps)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        embeddings = model.extract_features(input_tensor).squeeze().cpu().numpy()
    t_nn = (time.perf_counter() - t1) * 1000.0

    top_idx = int(np.argmax(probs))
    top_class = idx_to_class.get(top_idx, f"Class {top_idx}")
    top_prob = float(probs[top_idx])

    prob_dict = {
        idx_to_class.get(i, f"Class {i}"): float(probs[i])
        for i in range(len(probs))
    }

    spec_numpy = dsp_tensor.squeeze(0).cpu().numpy()
    spec_b64 = render_spectrogram_base64(
        spec_numpy,
        title=f"Log-Mel Spectrogram ({spec_numpy.shape[0]} Mels × {spec_numpy.shape[1]} Frames)",
    )

    return {
        "file_name": file_name,
        "duration_sec": len(waveform) / sr,
        "sample_rate": sr,
        "dsp_latency_ms": t_dsp,
        "nn_latency_ms": t_nn,
        "total_latency_ms": t_dsp + t_nn,
        "predicted_idx": top_idx,
        "predicted_class": top_class,
        "confidence": top_prob,
        "probabilities": prob_dict,
        "raw_waveform": waveform,
        "spectrogram": spec_numpy,
        "spectrogram_image": spec_b64,
        "embedding_norm": float(np.linalg.norm(embeddings)),
    }


def predict_audio_bytes(
    audio_bytes: bytes,
    file_name: str,
    model: torch.nn.Module,
    pipeline: AudioDSPPipeline,
    idx_to_class: Dict[int, str],
    device: torch.device,
) -> Dict:
    """Decodes in-memory audio bytes (WAV, MP3, etc.) and runs inference."""
    import librosa
    try:
        # Load from in-memory byte buffer
        waveform, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=pipeline.config.sample_rate,
            mono=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to decode audio bytes: {e}")

    return predict_audio_waveform(
        waveform=waveform.astype(np.float32),
        sr=sr,
        file_name=file_name,
        model=model,
        pipeline=pipeline,
        idx_to_class=idx_to_class,
        device=device,
    )


def predict_audio_file(
    audio_path: Path,
    model: torch.nn.Module,
    pipeline: AudioDSPPipeline,
    idx_to_class: Dict[int, str],
    device: torch.device,
) -> Dict:
    """Loads audio from disk and runs inference."""
    waveform, sr = load_audio(audio_path, target_sr=pipeline.config.sample_rate, mono=True)
    return predict_audio_waveform(
        waveform=waveform,
        sr=sr,
        file_name=audio_path.name,
        model=model,
        pipeline=pipeline,
        idx_to_class=idx_to_class,
        device=device,
    )


def generate_diagnostic_plot(
    results: List[Dict],
    output_path: Path,
):
    """Creates a multi-panel visual figure showing Waveforms, Spectrograms, and Class Probabilities."""
    num_samples = len(results)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, res in enumerate(results):
        waveform = res["raw_waveform"]
        spec = res["spectrogram"]
        probs = res["probabilities"]

        # 1. Waveform
        ax_wave = axes[i, 0]
        time_axis = np.linspace(0, res["duration_sec"], len(waveform))
        ax_wave.plot(time_axis, waveform, color="#1f77b4", lw=0.8)
        ax_wave.set_title(f"Audio Waveform: {res['file_name']}", fontsize=10, fontweight="bold")
        ax_wave.set_xlabel("Time (s)", fontsize=8)
        ax_wave.set_ylabel("Amplitude", fontsize=8)
        ax_wave.grid(True, alpha=0.3)
        ax_wave.set_ylim(-1.1, 1.1)

        # 2. Spectrogram
        ax_spec = axes[i, 1]
        im = ax_spec.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
        ax_spec.set_title(
            f"Log-Mel Spectrogram ({spec.shape[0]} mels × {spec.shape[1]} frames)",
            fontsize=10,
            fontweight="bold",
        )
        ax_spec.set_xlabel("Time Frames", fontsize=8)
        ax_spec.set_ylabel("Mel Frequency Bin", fontsize=8)
        fig.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)

        # 3. Probabilities
        ax_prob = axes[i, 2]
        classes = list(probs.keys())
        values = [probs[c] for c in classes]
        colors = ["#2ca02c" if c == res["predicted_class"] else "#7f7f7f" for c in classes]

        y_pos = np.arange(len(classes))
        ax_prob.barh(y_pos, values, color=colors, alpha=0.85)
        ax_prob.set_yticks(y_pos)
        ax_prob.set_yticklabels(classes, fontsize=9)
        ax_prob.set_xlim(0, 1.05)
        ax_prob.set_xlabel("Probability", fontsize=8)
        ax_prob.set_title(
            f"Prediction: {res['predicted_class']} ({res['confidence'] * 100:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color="#1b5e20" if res["confidence"] > 0.7 else "#d84315",
        )
        for y, val in zip(y_pos, values):
            ax_prob.text(val + 0.02, y, f"{val*100:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"📊 Diagnostic spectrogram visualization saved to: {output_path}")


def run_cli_demo(
    checkpoint_path: Path,
    target_audio: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    plot_path: Path = Path("checkpoints/demo_spectrograms.png"),
):
    """Executes terminal walkthrough demonstration."""
    device = get_default_device()
    print("=" * 70)
    print(" 🎵 DSP AUDIO EVENT CLASSIFICATION — INTERACTIVE DEMO 🎵")
    print(f" Compute Device:  {device}")
    print(f" Model Weights:   {checkpoint_path}")
    print("=" * 70)

    model, config, idx_to_class = load_classifier(checkpoint_path, device)
    pipeline = AudioDSPPipeline(config)

    print(f"\nConfiguration:")
    print(f"  • Sample Rate:  {config.sample_rate} Hz")
    print(f"  • Duration:     {config.duration} s ({config.target_samples} samples)")
    print(f"  • FFT Size:     {config.n_fft} (Hop: {config.hop_length})")
    print(f"  • Mel Channels: {config.n_mels}")
    print(f"  • Classes:      {', '.join(idx_to_class.values())}\n")

    audio_files: List[Path] = []
    if target_audio is not None and target_audio.exists():
        audio_files = [target_audio]
    else:
        dataset_path = dataset_dir or Path("data/mock_dataset")
        if dataset_path.exists():
            for cls_folder in sorted(dataset_path.iterdir()):
                if cls_folder.is_dir() and not cls_folder.name.startswith("."):
                    wavs = sorted(cls_folder.glob("*.wav"))
                    if wavs:
                        audio_files.append(wavs[0])
            root_sample = dataset_path / "sample_0.wav"
            if root_sample.exists() and root_sample not in audio_files:
                audio_files.insert(0, root_sample)

    if not audio_files:
        print("No audio files found. Generating sample tone...")
        from tests.generate_mock import generate_synthetic_dataset
        dataset_path = Path("data/mock_dataset")
        generate_synthetic_dataset(dataset_path, num_samples_per_class=2)
        audio_files = sorted(dataset_path.glob("*/*.wav"))[:4]

    results = []
    for file_path in audio_files:
        print("-" * 70)
        print(f"Analyzing: {file_path}")
        res = predict_audio_file(file_path, model, pipeline, idx_to_class, device)
        results.append(res)

        print(f"  • Waveform:   {res['duration_sec']:.2f}s @ {res['sample_rate']} Hz")
        print(f"  • Latency:    DSP Extraction: {res['dsp_latency_ms']:.2f}ms | Neural Network: {res['nn_latency_ms']:.2f}ms")
        print(f"  • Predicted:  \033[1;32m{res['predicted_class']}\033[0m (Confidence: {res['confidence']*100:.2f}%)")
        print("  • Class Probability Distribution:")
        for cls_name, prob in res["probabilities"].items():
            mark = "  ★ " if cls_name == res["predicted_class"] else "    "
            print(f"{mark}{cls_name:<18} {render_ascii_bar(prob)}")

    print("-" * 70)
    generate_diagnostic_plot(results, plot_path)
    print("\n✅ Demo analysis complete.")
    print("To test any custom audio file:")
    print("  .venv/bin/python demo_app.py --audio path/to/your_audio.wav\n")


def build_dashboard_html(config: AudioConfig, idx_to_class: Dict[int, str], device: torch.device) -> str:
    """Generates modern web application HTML with drag-drop upload and audio recording."""
    classes_json = json.dumps(list(idx_to_class.values()))
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DSP Audio Event Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #151d30;
            --bg-accent: #1e293b;
            --border: #2a374f;
            --primary: #38bdf8;
            --primary-hover: #0284c7;
            --success: #10b981;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            min-height: 100vh;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 24px;
            text-align: center;
        }}
        h1 {{
            font-size: 28px;
            font-weight: 700;
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        .subtitle {{
            color: var(--text-sub);
            font-size: 14px;
            margin-top: 6px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }}
        .badge-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .badge {{
            background: var(--bg-accent);
            border: 1px solid var(--border);
            color: #bae6fd;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }}
        /* Drag and Drop Zone */
        .drop-zone {{
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 32px 20px;
            text-align: center;
            background: rgba(30, 41, 59, 0.4);
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }}
        .drop-zone:hover, .drop-zone.dragover {{
            border-color: var(--primary);
            background: rgba(56, 189, 248, 0.08);
            transform: translateY(-1px);
        }}
        .drop-icon {{
            font-size: 40px;
            margin-bottom: 8px;
            display: block;
        }}
        .drop-title {{
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
        }}
        .drop-hint {{
            font-size: 13px;
            color: var(--text-sub);
            margin-top: 4px;
        }}
        input[type="file"] {{
            display: none;
        }}
        /* Action Bar */
        .action-bar {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        .btn {{
            background: var(--primary);
            color: #0b0f19;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: 0.15s;
        }}
        .btn:hover {{
            background: #7dd3fc;
        }}
        .btn-record {{
            background: #ef4444;
            color: white;
        }}
        .btn-record:hover {{
            background: #dc2626;
        }}
        .btn-record.recording {{
            background: #dc2626;
            animation: pulse 1.2s infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
            70% {{ box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        /* Presets Area */
        .preset-title {{
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-sub);
            margin-bottom: 10px;
        }}
        .preset-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .btn-preset {{
            background: var(--bg-accent);
            color: #e2e8f0;
            border: 1px solid var(--border);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            transition: 0.15s;
        }}
        .btn-preset:hover {{
            background: #334155;
            border-color: var(--primary);
            color: var(--primary);
        }}
        /* Results Section */
        #results-container {{
            display: none;
        }}
        .res-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .res-tag {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
            font-size: 20px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 8px;
        }}
        .latency-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }}
        .latency-card {{
            background: var(--bg-accent);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .latency-val {{
            font-size: 18px;
            font-weight: 700;
            color: var(--primary);
        }}
        .latency-lbl {{
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 2px;
            text-transform: uppercase;
        }}
        /* Probabilities */
        .prob-item {{
            margin-bottom: 10px;
        }}
        .prob-info {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 4px;
        }}
        .prob-track {{
            background: #0f172a;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        .prob-fill {{
            background: linear-gradient(90deg, #0284c7, #38bdf8);
            height: 100%;
            border-radius: 6px;
            transition: width 0.4s ease;
        }}
        .prob-fill.winner {{
            background: linear-gradient(90deg, #059669, #10b981);
        }}
        /* Spectrogram Image */
        .spec-container {{
            margin-top: 20px;
            text-align: center;
        }}
        .spec-img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            background: #0f172a;
        }}
        audio {{
            width: 100%;
            margin-top: 14px;
            border-radius: 8px;
        }}
        .spinner {{
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎵 Audio Event Classification Dashboard</h1>
            <p class="subtitle">Real-time DSP Preprocessing & Deep 2D Spectrogram CNN Inference</p>
            <div class="badge-list" style="justify-content: center;">
                <span class="badge">Hardware: {device}</span>
                <span class="badge">Sampling Rate: {config.sample_rate} Hz</span>
                <span class="badge">Window/FFT: {config.n_fft} / {config.hop_length}</span>
                <span class="badge">Mel Channels: {config.n_mels}</span>
                <span class="badge">Classes: {len(idx_to_class)}</span>
            </div>
        </header>

        <!-- Upload & Record Card -->
        <div class="card">
            <div id="drop-zone" class="drop-zone">
                <span class="drop-icon">📁</span>
                <div class="drop-title">Drag & Drop Audio File Here</div>
                <div class="drop-hint">or click to browse (.wav, .mp3, .flac, .ogg)</div>
                <input type="file" id="file-input" accept=".wav,.mp3,.flac,.ogg,.m4a">
            </div>

            <div class="action-bar">
                <button id="browse-btn" class="btn" style="background:#334155; color:white;">📂 Browse File</button>
                <button id="record-btn" class="btn btn-record">
                    <span id="record-icon">🎙️</span>
                    <span id="record-text">Record Microphone</span>
                </button>
            </div>
        </div>

        <!-- Presets Card -->
        <div class="card">
            <div class="preset-title">Or Choose a Presynthesized Sound Event</div>
            <div class="preset-buttons" id="presets-container"></div>
        </div>

        <!-- Results Card -->
        <div class="card" id="results-container">
            <div class="res-header">
                <div>
                    <span style="font-size: 13px; color: var(--text-sub); display: block;">CLASSIFICATION RESULT</span>
                    <span id="file-analyzed" style="font-weight: 600; font-size: 16px;">sample.wav</span>
                </div>
                <div class="res-tag" id="predicted-badge">Predicted Class</div>
            </div>

            <div class="latency-grid">
                <div class="latency-card">
                    <div class="latency-val" id="lat-dsp">0.0 ms</div>
                    <div class="latency-lbl">DSP Extraction</div>
                </div>
                <div class="latency-card">
                    <div class="latency-val" id="lat-nn">0.0 ms</div>
                    <div class="latency-lbl">Neural Inference</div>
                </div>
                <div class="latency-card">
                    <div class="latency-val" id="lat-total">0.0 ms</div>
                    <div class="latency-lbl">Total Latency</div>
                </div>
                <div class="latency-card">
                    <div class="latency-val" id="confidence-val">0.0%</div>
                    <div class="latency-lbl">Confidence</div>
                </div>
            </div>

            <!-- Audio Player -->
            <audio id="audio-player" controls></audio>

            <!-- Probabilities -->
            <div style="margin-top: 20px;">
                <div style="font-size: 13px; color: var(--text-sub); margin-bottom: 10px; font-weight: 600;">
                    PROBABILITY DISTRIBUTION
                </div>
                <div id="prob-list"></div>
            </div>

            <!-- Log-Mel Spectrogram Image -->
            <div class="spec-container">
                <div style="font-size: 13px; color: var(--text-sub); margin-bottom: 8px; font-weight: 600; text-align: left;">
                    EXTRACTED LOG-MEL SPECTROGRAM
                </div>
                <img id="spec-image" class="spec-img" alt="Log-Mel Spectrogram">
            </div>
        </div>
    </div>

    <script>
        const classes = {classes_json};
        const presetsContainer = document.getElementById('presets-container');
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const browseBtn = document.getElementById('browse-btn');
        const recordBtn = document.getElementById('record-btn');
        const recordText = document.getElementById('record-text');
        const recordIcon = document.getElementById('record-icon');
        const resultsContainer = document.getElementById('results-container');
        const audioPlayer = document.getElementById('audio-player');

        // Populate Presets
        classes.forEach(c => {{
            const btn = document.createElement('button');
            btn.className = 'btn-preset';
            btn.innerHTML = '⚡ ' + c;
            btn.onclick = () => runPreset(c);
            presetsContainer.appendChild(btn);
        }});

        // File Browse Handlers
        browseBtn.onclick = () => fileInput.click();
        dropZone.onclick = (e) => {{
            if (e.target !== fileInput) fileInput.click();
        }};
        fileInput.onchange = (e) => {{
            if (e.target.files.length > 0) {{
                processFile(e.target.files[0]);
            }}
        }};

        // Drag & Drop Handlers
        ['dragenter', 'dragover'].forEach(name => {{
            dropZone.addEventListener(name, (e) => {{
                e.preventDefault();
                dropZone.classList.add('dragover');
            }});
        }});
        ['dragleave', 'drop'].forEach(name => {{
            dropZone.addEventListener(name, (e) => {{
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }});
        }});
        dropZone.addEventListener('drop', (e) => {{
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {{
                processFile(dt.files[0]);
            }}
        }});

        // Microphone Recorder Setup
        let isRecording = false;
        let mediaStream = null;
        let audioContext = null;
        let scriptProcessor = null;
        let recordedChunks = [];
        let recordTimer = null;
        let recordSeconds = 0;

        recordBtn.onclick = async () => {{
            if (!isRecording) {{
                startRecording();
            }} else {{
                stopRecording();
            }}
        }};

        async function startRecording() {{
            try {{
                recordedChunks = [];
                recordSeconds = 0;
                mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true, video: false }});
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(mediaStream);
                
                // Buffer audio PCM
                scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                scriptProcessor.onaudioprocess = (e) => {{
                    if (!isRecording) return;
                    const channelData = e.inputBuffer.getChannelData(0);
                    recordedChunks.push(new Float32Array(channelData));
                }};

                source.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);

                isRecording = true;
                recordBtn.classList.add('recording');
                recordIcon.innerText = '⏹️';
                recordText.innerText = 'Stop & Classify (0s)';

                recordTimer = setInterval(() => {{
                    recordSeconds++;
                    recordText.innerText = `Stop & Classify (${{recordSeconds}}s)`;
                    if (recordSeconds >= 5) stopRecording(); // auto-stop at 5s
                }}, 1000);

            }} catch (err) {{
                alert('Microphone access denied or unsupported: ' + err.message);
            }}
        }}

        async function stopRecording() {{
            if (!isRecording) return;
            isRecording = false;
            clearInterval(recordTimer);
            recordBtn.classList.remove('recording');
            recordIcon.innerText = '🎙️';
            recordText.innerText = 'Record Microphone';

            if (mediaStream) {{
                mediaStream.getTracks().forEach(t => t.stop());
            }}
            if (scriptProcessor) scriptProcessor.disconnect();

            // Flatten PCM samples
            let totalLength = 0;
            for (let c of recordedChunks) totalLength += c.length;
            const pcmData = new Float32Array(totalLength);
            let offset = 0;
            for (let c of recordedChunks) {{
                pcmData.set(c, offset);
                offset += c.length;
            }}

            // Encode to 16-bit PCM WAV
            const wavBlob = encodeWAV(pcmData, audioContext.sampleRate);
            audioContext.close();

            processAudioBlob(wavBlob, 'microphone_capture.wav');
        }}

        function encodeWAV(samples, sampleRate) {{
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            const writeString = (view, offset, str) => {{
                for (let i = 0; i < str.length; i++) {{
                    view.setUint8(offset + i, str.charCodeAt(i));
                }}
            }};

            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM
            view.setUint16(22, 1, true); // Mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true); // 16-bit
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * 2, true);

            let index = 44;
            for (let i = 0; i < samples.length; i++, index += 2) {{
                let s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }}
            return new Blob([view], {{ type: 'audio/wav' }});
        }}

        // Process File / Preset
        function processFile(file) {{
            processAudioBlob(file, file.name);
        }}

        async function processAudioBlob(blob, fileName) {{
            resultsContainer.style.display = 'block';
            document.getElementById('file-analyzed').innerText = fileName + ' (' + (blob.size / 1024).toFixed(1) + ' KB)';
            document.getElementById('predicted-badge').innerHTML = '<span class="spinner"></span> Processing...';
            
            // Set audio player source
            audioPlayer.src = URL.createObjectURL(blob);

            try {{
                const res = await fetch('/predict_upload?name=' + encodeURIComponent(fileName), {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/octet-stream' }},
                    body: blob
                }});
                const data = await res.json();
                renderResults(data);
            }} catch (err) {{
                alert('Inference error: ' + err.message);
            }}
        }}

        async function runPreset(className) {{
            resultsContainer.style.display = 'block';
            document.getElementById('file-analyzed').innerText = 'Preset: ' + className;
            document.getElementById('predicted-badge').innerHTML = '<span class="spinner"></span> Processing...';

            try {{
                const res = await fetch('/predict?class=' + encodeURIComponent(className));
                const data = await res.json();
                audioPlayer.src = '/sample_audio?class=' + encodeURIComponent(className);
                renderResults(data);
            }} catch (err) {{
                alert('Preset error: ' + err.message);
            }}
        }}

        function renderResults(data) {{
            document.getElementById('predicted-badge').innerText = data.predicted_class;
            document.getElementById('confidence-val').innerText = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('lat-dsp').innerText = data.dsp_latency_ms.toFixed(1) + ' ms';
            document.getElementById('lat-nn').innerText = data.nn_latency_ms.toFixed(1) + ' ms';
            document.getElementById('lat-total').innerText = data.total_latency_ms.toFixed(1) + ' ms';

            // Render Probabilities
            const probList = document.getElementById('prob-list');
            probList.innerHTML = '';
            for (let c in data.probabilities) {{
                const p = data.probabilities[c];
                const isWin = (c === data.predicted_class);
                const percent = (p * 100).toFixed(1);

                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <div class="prob-info">
                        <span style="${{isWin ? 'color:#38bdf8; font-weight:700;' : ''}}">${{c}}</span>
                        <span style="font-weight:600;">${{percent}}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill ${{isWin ? 'winner' : ''}}" style="width: ${{percent}}%;"></div>
                    </div>
                `;
                probList.appendChild(item);
            }}

            // Render Spectrogram Image
            if (data.spectrogram_image) {{
                document.getElementById('spec-image').src = data.spectrogram_image;
            }}
        }}
    </script>
</body>
</html>"""


def run_web_server(checkpoint_path: Path, port: int = 8555):
    """Launches interactive HTTP dashboard with file upload and live microphone classification."""
    device = get_default_device()
    model, config, idx_to_class = load_classifier(checkpoint_path, device)
    pipeline = AudioDSPPipeline(config)

    HTML_PAGE = build_dashboard_html(config, idx_to_class, device)

    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))

            elif path == "/predict":
                params = urllib.parse.parse_qs(parsed.query)
                target_cls = params.get("class", ["sine_high"])[0]

                # Locate sample
                sample_file = Path(f"data/mock_dataset/{target_cls}/{target_cls}_000.wav")
                if not sample_file.exists():
                    sample_file = Path("data/mock_dataset/sample_0.wav")

                res = predict_audio_file(sample_file, model, pipeline, idx_to_class, device)
                res.pop("raw_waveform", None)
                res.pop("spectrogram", None)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))

            elif path == "/sample_audio":
                params = urllib.parse.parse_qs(parsed.query)
                target_cls = params.get("class", ["sine_high"])[0]
                sample_file = Path(f"data/mock_dataset/{target_cls}/{target_cls}_000.wav")
                if not sample_file.exists():
                    sample_file = Path("data/mock_dataset/sample_0.wav")

                if sample_file.exists():
                    with open(sample_file, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/predict_upload":
                content_len = int(self.headers.get("Content-Length", 0))
                audio_bytes = self.rfile.read(content_len)

                params = urllib.parse.parse_qs(parsed.query)
                file_name = params.get("name", ["uploaded_audio.wav"])[0]

                try:
                    res = predict_audio_bytes(
                        audio_bytes=audio_bytes,
                        file_name=file_name,
                        model=model,
                        pipeline=pipeline,
                        idx_to_class=idx_to_class,
                        device=device,
                    )
                    res.pop("raw_waveform", None)
                    res.pop("spectrogram", None)

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode("utf-8"))

                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return  # Suppress request logging spam

    # Automatic port fallback
    socketserver.TCPServer.allow_reuse_address = True
    actual_port = port
    httpd = None
    for p in range(port, port + 50):
        try:
            httpd = socketserver.TCPServer(("", p), DashboardHandler)
            actual_port = p
            break
        except OSError:
            continue

    if httpd is None:
        raise RuntimeError(f"Could not bind to any port in range {port} to {port + 50}")

    print(f"🚀 Audio Classifier Web Server running at: http://localhost:{actual_port}")
    print("Press Ctrl+C to terminate server.")
    with httpd:
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="DSP Audio Event Classifier Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model weights checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to single audio .wav file to classify.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch browser-accessible web application.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8555,
        help="Port for local web server (defaults to 8555).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None or not ckpt_path.exists():
        candidates = [
            Path("checkpoints/best_model.pt"),
            Path("checkpoints/best_model.pth"),
        ]
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break

    if ckpt_path is None or not ckpt_path.exists():
        print("Checkpoint not found. Running training on mock data first...")
        from src.train import main as train_main
        train_main()
        ckpt_path = Path("checkpoints/best_model.pt")

    target_audio = Path(args.audio) if args.audio else None

    if args.server:
        run_web_server(ckpt_path, port=args.port)
    else:
        run_cli_demo(checkpoint_path=ckpt_path, target_audio=target_audio)


if __name__ == "__main__":
    main()
