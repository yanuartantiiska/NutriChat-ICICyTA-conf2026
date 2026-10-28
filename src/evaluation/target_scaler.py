import json
from pathlib import Path

import torch


class TargetScaler:

    def __init__(
        self,
        stats_path="config/target_stats.json",
    ):
        stats_path = Path(stats_path)

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Target stats not found: {stats_path}"
            )

        with open(
            stats_path,
            "r",
            encoding="utf-8",
        ) as f:
            stats = json.load(f)

        self.targets = stats["targets"]

        self.mean = torch.tensor(
            [
                stats["mean"][name]
                for name in self.targets
            ],
            dtype=torch.float32,
        )

        self.std = torch.tensor(
            [
                stats["std"][name]
                for name in self.targets
            ],
            dtype=torch.float32,
        )

        if torch.any(self.std <= 0):
            raise ValueError(
                "All target standard deviations must be > 0."
            )

    def normalize(
        self,
        values,
    ):
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)

        return (
            values - mean
        ) / std

    def denormalize(
        self,
        values,
    ):
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)

        return (
            values * std
        ) + mean