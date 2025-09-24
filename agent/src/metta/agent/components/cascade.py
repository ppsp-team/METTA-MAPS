"""
Modular cascade and comparison components for TensorDict-based policies.
These components can be added to any policy to enable cascade and comparison functionality.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for cascade component."""

    in_key: str = "core"  # Input key in TensorDict
    out_key: str = "core_cascaded"  # Output key in TensorDict
    cascade_rate: float = 1.0  # 1.0 disables cascade, lower values enable blending


@dataclass
class ComparisonConfig:
    """Configuration for comparison component."""

    input_key: str = "encoded_obs"  # Original input (e.g., CNN output)
    hidden_key: str = "core_cascaded"  # Hidden state to reconstruct from
    weights_source_key: str = "actor_1"  # Component to get reconstruction weights from
    out_key: str = "comparison"  # Output key for comparison result


class CascadeModule(TDM):
    """
    TensorDictModule component that applies cascade blending to hidden states.

    This component maintains memory of previous hidden states and blends them
    with current states based on the cascade_rate parameter.

    Usage:
        cascade = CascadeModule(CascadeConfig(
            in_key="core",
            out_key="core_cascaded",
            cascade_rate=0.5
        ))
    """

    def __init__(self, config: CascadeConfig):
        # Create a simple pass-through module for TDM interface
        identity_module = nn.Identity()

        super().__init__(module=identity_module, in_keys=[config.in_key], out_keys=[config.out_key])

        self.config = config
        self.previous_hidden = None
        self.cascade_rate = config.cascade_rate

    def forward(self, td: TensorDict) -> TensorDict:
        """Apply cascade blending and store result in TensorDict."""
        current_hidden = td[self.config.in_key]

        # Apply cascade if we have previous state and cascade is enabled
        if self.previous_hidden is not None and self.cascade_rate < 1.0:
            cascaded = self._apply_cascade(current_hidden, self.previous_hidden)
            td[self.config.out_key] = cascaded
        else:
            # No cascade - pass through unchanged
            td[self.config.out_key] = current_hidden

        # Store current state for next iteration (detach to prevent gradient accumulation)
        self.previous_hidden = td[self.config.out_key].detach().clone()

        return td

    def _apply_cascade(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """
        Apply cascade blending between current and previous hidden states.
        Formula: cascade_rate * current + (1 - cascade_rate) * previous
        """
        # Ensure same device
        previous = previous.to(current.device)

        # Handle batch size differences between training and inference
        if previous.shape[0] != current.shape[0]:
            if previous.shape[0] == 1 and current.shape[0] > 1:
                # Replicate single state for batch
                previous = previous.repeat(current.shape[0], 1)
            elif previous.shape[0] > current.shape[0]:
                # Truncate to current batch size
                previous = previous[: current.shape[0]]
            # Note: Other shape mismatches will cause an error, which is appropriate

        # Apply cascade blending
        return self.cascade_rate * current + (1 - self.cascade_rate) * previous

    def reset_memory(self):
        """Reset cascade memory (call when starting new episodes)."""
        self.previous_hidden = None

    def set_cascade_rate(self, rate: float):
        """Dynamically adjust cascade rate."""
        self.cascade_rate = rate


class ComparisonModule(TDM):
    """
    TensorDictModule component that computes comparison between original input
    and reconstructed input from hidden state.

    This implements the comparison logic from the original MettaAgent:
    - Takes original CNN output and cascaded hidden state
    - Reconstructs the CNN output using linear transformation
    - Computes difference as comparison signal

    Usage:
        comparison = ComparisonModule(ComparisonConfig(
            input_key="encoded_obs",
            hidden_key="core_cascaded",
            weights_source_key="actor_1",
            out_key="comparison"
        ))
    """

    def __init__(self, config: ComparisonConfig, policy_reference: Optional[nn.Module] = None):
        # Create identity module for TDM interface
        identity_module = nn.Identity()

        super().__init__(
            module=identity_module, in_keys=[config.input_key, config.hidden_key], out_keys=[config.out_key]
        )

        self.config = config
        self.policy_reference = policy_reference  # Reference to policy for accessing weights

    def forward(self, td: TensorDict) -> TensorDict:
        """Compute comparison between original input and reconstructed input."""
        try:
            original_input = td[self.config.input_key]
            hidden_state = td[self.config.hidden_key]

            # Get reconstruction weights
            weights = self._get_reconstruction_weights()
            if weights is None:
                logger.warning("Could not get reconstruction weights, skipping comparison")
                return td

            # Reconstruct original input from hidden state
            # This matches: Output_comparison = f.relu(f.linear(Hidden, weights.t()))
            reconstructed = torch.relu(torch.nn.functional.linear(hidden_state, weights.t()))

            # Check shape compatibility
            if reconstructed.shape != original_input.shape:
                logger.warning(f"Shape mismatch in comparison: {reconstructed.shape} vs {original_input.shape}")
                return td

            # Compute comparison: Input - Output_comparison
            comparison = original_input - reconstructed
            td[self.config.out_key] = comparison

        except Exception as e:
            logger.warning(f"Error computing comparison: {e}")

        return td

    def _get_reconstruction_weights(self) -> Optional[torch.Tensor]:
        """Get weights for reconstructing input from hidden state."""
        if self.policy_reference is None:
            return None

        try:
            # Get the component specified in config
            component = getattr(self.policy_reference, self.config.weights_source_key, None)
            if component is None:
                return None

            # Extract weights from TensorDictModule
            if hasattr(component, "module") and hasattr(component.module, "weight"):
                return component.module.weight

            # Alternative: search for Linear modules
            for module in component.modules():
                if isinstance(module, nn.Linear):
                    return module.weight

            return None

        except Exception as e:
            logger.warning(f"Error accessing reconstruction weights: {e}")
            return None
