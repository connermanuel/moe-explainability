"""Model configurations for MoE routing analysis."""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Configuration for MoE model routing extraction."""

    name: str
    layers: List[int]
    num_experts: int
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration."""
        if not self.layers:
            raise ValueError("layers cannot be empty")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")


# Predefined configurations
SWITCH_BASE_8 = ModelConfig(
    name="google/switch-base-8",
    layers=[1, 3, 5, 7, 9, 11],
    num_experts=8,
    device="cuda",
)
