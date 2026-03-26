from __future__ import annotations

from torch.utils.data import DataLoader

from data.augmentations import AudioAugmentationPipeline
from data.dataset import AudioPhysioDataset, DatasetConfig


def build_dataloader(config: DatasetConfig, split_name: str, batch_size: int = 8, num_workers: int = 0, augment: bool = False) -> DataLoader:
    transform = AudioAugmentationPipeline(sample_rate=config.sample_rate) if augment else None
    dataset = AudioPhysioDataset(config=config, split_name=split_name, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split_name == "train", num_workers=num_workers)
