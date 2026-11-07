"""
PyTorch Dataset and SpecAugment Augmentations for Audio Event Classification.

Provides:
- Frequency and Time Masking (SpecAugment)
- AudioEventDataset: robust on-the-fly spectrogram generation from audio files
- DataLoader creation utilities with automated train/validation split
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .dsp_pipeline import AudioConfig, AudioDSPPipeline


class SpecAugment:
    """SpecAugment data augmentation for audio spectrograms.

    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition", Interspeech 2019.

    Applies:
    - Frequency Masking: consecutive channels [f0, f0 + f) are masked.
    - Time Masking: consecutive time frames [t0, t0 + t) are masked.
    """

    def __init__(
        self,
        freq_mask_max: int = 8,
        time_mask_max: int = 16,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        mask_value: float = 0.0,
    ):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Applies frequency and time masking in-place or on a clone.

        Parameters
        ----------
        spec : torch.Tensor
            Spectrogram tensor of shape (C, F, T) or (F, T).

        Returns
        -------
        torch.Tensor
            Augmented spectrogram tensor.
        """
        augmented = spec.clone()
        if augmented.ndim == 2:
            augmented = augmented.unsqueeze(0)

        _, num_mels, num_time = augmented.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f_len = np.random.randint(0, min(self.freq_mask_max, num_mels) + 1)
            f0 = np.random.randint(0, max(1, num_mels - f_len + 1))
            augmented[:, f0 : f0 + f_len, :] = self.mask_value

        # Time masking
        for _ in range(self.num_time_masks):
            t_len = np.random.randint(0, min(self.time_mask_max, num_time) + 1)
            t0 = np.random.randint(0, max(1, num_time - t_len + 1))
            augmented[:, :, t0 : t0 + t_len] = self.mask_value

        if spec.ndim == 2:
            return augmented.squeeze(0)
        return augmented


class AudioEventDataset(Dataset):
    """PyTorch Dataset for Audio Event Classification.

    Reads raw audio files and extracts Log-Mel spectrograms on the fly.
    """

    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

    def __init__(
        self,
        audio_paths: Optional[List[Union[str, Path]]] = None,
        labels: Optional[List[int]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        config: Optional[AudioConfig] = None,
        augment: bool = False,
        augmentor: Optional[SpecAugment] = None,
    ):
        """Initializes the dataset either from explicit lists of paths & labels or a data directory.

        Directory structure expected if data_dir is provided:
            data_dir/
                class_a/
                    sample1.wav
                    sample2.wav
                class_b/
                    sample3.wav
        """
        self.config = config or AudioConfig()
        self.pipeline = AudioDSPPipeline(self.config)
        self.augment = augment
        self.augmentor = augmentor or SpecAugment(freq_mask_max=8, time_mask_max=16)

        self.audio_paths: List[Path] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = class_to_idx or {}

        if data_dir is not None:
            self._load_from_directory(Path(data_dir))
        elif audio_paths is not None and labels is not None:
            if len(audio_paths) != len(labels):
                raise ValueError("audio_paths and labels must have identical length.")
            self.audio_paths = [Path(p) for p in audio_paths]
            self.labels = [int(l) for l in labels]
        else:
            raise ValueError("Either data_dir or (audio_paths, labels) must be provided.")

        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def _load_from_directory(self, data_dir: Path):
        """Scans directory hierarchy and collects audio files and class indices."""
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

        classes = sorted(
            [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
        if not classes:
            raise ValueError(f"No class subdirectories found in '{data_dir}'.")

        if not self.class_to_idx:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            if cls_name not in self.class_to_idx:
                continue
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = data_dir / cls_name
            for file_path in sorted(cls_dir.iterdir()):
                if file_path.suffix.lower() in self.AUDIO_EXTENSIONS:
                    self.audio_paths.append(file_path)
                    self.labels.append(cls_idx)

        if not self.audio_paths:
            raise ValueError(f"No valid audio files found in '{data_dir}'.")

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads audio file, computes Log-Mel spectrogram, and applies augmentation if enabled.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            spectrogram: Tensor of shape (1, n_mels, time_steps)
            label: Tensor of shape () (scalar long)
        """
        file_path = self.audio_paths[idx]
        label = self.labels[idx]

        spec_tensor = self.pipeline.process_file(file_path)

        if self.augment and self.augmentor is not None:
            spec_tensor = self.augmentor(spec_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    train_split: float = 0.8,
    config: Optional[AudioConfig] = None,
    num_workers: int = 0,
    seed: int = 42,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Convenience helper to build training and validation DataLoaders from a directory.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Root dataset directory containing subdirectories for each audio class.
    batch_size : int
        Batch size.
    train_split : float
        Fraction of data assigned to training set.
    config : Optional[AudioConfig]
        DSP parameters.
    num_workers : int
        Number of multiprocessing workers for DataLoader.
    seed : int
        Random generator seed for reproducible dataset splitting.
    augment_train : bool
        Whether to apply SpecAugment to the training split.

    Returns
    -------
    Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]
        (train_loader, val_loader, class_to_idx, idx_to_class)
    """
    full_dataset = AudioEventDataset(
        data_dir=data_dir,
        config=config,
        augment=False,
    )

    total_samples = len(full_dataset)
    train_len = int(train_split * total_samples)
    val_len = total_samples - train_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=generator)

    # Wrap training subset to enable data augmentation
    train_paths = [full_dataset.audio_paths[i] for i in train_subset.indices]
    train_labels = [full_dataset.labels[i] for i in train_subset.indices]
    val_paths = [full_dataset.audio_paths[i] for i in val_subset.indices]
    val_labels = [full_dataset.labels[i] for i in val_subset.indices]

    train_ds = AudioEventDataset(
        audio_paths=train_paths,
        labels=train_labels,
        class_to_idx=full_dataset.class_to_idx,
        config=config,
        augment=augment_train,
    )

    val_ds = AudioEventDataset(
        audio_paths=val_paths,
        labels=val_labels,
        class_to_idx=full_dataset.class_to_idx,
        config=config,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, full_dataset.class_to_idx, full_dataset.idx_to_class
