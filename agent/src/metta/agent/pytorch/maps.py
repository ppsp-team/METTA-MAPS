import logging
import math
import warnings

import einops
import numpy as np
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.pytorch.base import LSTMWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class MAPS(PyTorchAgentMixin, LSTMWrapper):
    """Fast CNN-based policy with LSTM using PyTorchAgentMixin for shared functionality + MAPS"""

    def __init__(self, env, policy=None, cnn_channels=128, input_size=128, hidden_size=128, num_layers=2, **kwargs):
        """Initialize Fast CNN-based policy with LSTM and mixin support."""
        mixin_params = self.extract_mixin_params(kwargs)  # Extract mixin parameters before passing to parent

        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)  # Pass num_layers=2 to match YAML

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None, cascade_rate: float = 1.0):
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Determine dimensions from observations
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
            # Reshape TD for training if needed
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Use mixin method to set TensorDict fields properly
        self.set_tensordict_fields(td, observations)

        # === STORE FLATTENED CNN OUTPUT FOR COMPARISON ===
        # We need to capture the CNN output before fc1 for comparison logic
        # This requires calling encode_observations with intermediate capture
        cnn_flattened = self._encode_with_intermediate_capture(observations, state)

        # Standard encoding for LSTM input
        hidden = self.policy.encode_observations(observations, state)

        # Use base class method for LSTM state management
        lstm_h, lstm_c, env_id = self._manage_lstm_state(td, B, TT, observations.device)
        lstm_state = (lstm_h, lstm_c)

        # Forward LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)

        # Use base class method to store state with automatic detachment
        self._store_lstm_state(new_lstm_h, new_lstm_c, env_id)

        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # === CASCADE LOGIC (matches old MettaAgent encoded_obs level) ===
        current_hidden = flat_hidden.clone()  # Store current for state management

        if state.get("hidden") is not None:
            # Ensure previous hidden state is on same device and correct shape
            prev_hidden = state["hidden"].to(flat_hidden.device)

            # Handle batch size differences between training and inference
            if prev_hidden.shape[0] != flat_hidden.shape[0]:
                if prev_hidden.shape[0] == 1 and flat_hidden.shape[0] > 1:
                    # Replicate single state for batch
                    prev_hidden = prev_hidden.repeat(flat_hidden.shape[0], 1)
                elif prev_hidden.shape[0] > flat_hidden.shape[0]:
                    # Truncate to current batch size
                    prev_hidden = prev_hidden[: flat_hidden.shape[0]]

            # Apply cascade: cascade_rate * current + (1-cascade_rate) * previous
            # This matches: Hidden = cascade_rate*Hidden + (1-cascade_rate)*prev_h2
            cascaded_hidden = cascade_rate * flat_hidden + (1 - cascade_rate) * prev_hidden
            flat_hidden = cascaded_hidden  # Use cascaded version for action decoding
            current_hidden = cascaded_hidden.clone()  # Update current for state storage

        # === COMPARISON LOGIC (matches old MettaAgent comparison method) ===
        comparison = None
        if cnn_flattened is not None:
            # This matches: Output_comparison = f.relu(f.linear(Hidden, self.fc_hidden.weight.t()))
            fc1_weight_transposed = self.policy.fc1.weight.t()  # Get fc1 weights transposed
            reconstructed_input = torch.relu(torch.nn.functional.linear(current_hidden, fc1_weight_transposed))
            # This matches: Comparison = Input - Output_comparison
            comparison = cnn_flattened - reconstructed_input

            # Store in TensorDict for downstream use
            td["comparison"] = comparison

        # Update state with current hidden for next iteration (detach to avoid gradients)
        state["hidden"] = current_hidden.detach()

        # Decode using the (potentially cascaded) hidden state
        logits_list, value = self.policy.decode_actions(flat_hidden, B * TT)

        # Use mixin for mode-specific processing (handles all TD reshaping)
        if action is None:
            # Mixin handles inference mode
            td = self.forward_inference(td, logits_list, value)
        else:
            # Mixin handles training mode with proper reshaping
            td = self.forward_training(td, action, logits_list, value)

        return td

    def _encode_with_intermediate_capture(self, observations, state=None):
        """
        Encode observations and capture intermediate CNN output for comparison logic.

        This replicates the Policy.encode_observations flow but captures the flattened
        CNN output (equivalent to obs_flattener in the old code) before fc1.
        """
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        B_TT = B * TT

        if token_observations.dim() != 3:
            import einops

            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        # Use the same token-to-box conversion as the main encode_observations
        # This is a simplified version - you might want to extract this to a shared method
        try:
            # Call the policy's encode method to get box_obs, then run CNN manually
            box_obs = self._token_to_box(token_observations, B_TT)

            # Run CNN pipeline up to the flatten step (before fc1)
            x = box_obs / self.policy.max_vec
            x = self.policy.cnn1(x)
            x = torch.nn.functional.relu(x)
            x = self.policy.cnn2(x)
            x = torch.nn.functional.relu(x)
            cnn_flattened = self.policy.flatten(x)  # This is equivalent to obs_flattener

            return cnn_flattened

        except Exception as e:
            # If we can't capture intermediate, return None (comparison will be disabled)
            logger.warning(f"Could not capture CNN intermediate for comparison: {e}")
            return None

    def _token_to_box(self, token_observations, B_TT):
        """
        Extract the token-to-box conversion logic from Policy.encode_observations.
        This avoids code duplication when we need the intermediate CNN output.
        """
        import warnings

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

        # Extract coordinates and attributes (matching Policy.encode_observations exactly)
        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = token_observations[..., 1].long()
        atr_values = token_observations[..., 2].float()

        # Create mask for valid tokens
        valid_tokens = coords_byte != 0xFF
        valid_atr = atr_indices < self.policy.num_layers
        valid_mask = valid_tokens & valid_atr

        # Log warning for out-of-bounds indices
        invalid_atr_mask = valid_tokens & ~valid_atr
        if invalid_atr_mask.any():
            invalid_indices = atr_indices[invalid_atr_mask].unique()
            warnings.warn(
                f"Found observation attribute indices {sorted(invalid_indices.tolist())} "
                f">= num_layers ({self.policy.num_layers}). These tokens will be ignored.",
                stacklevel=2,
            )

        # Scatter-based write to avoid multi-dim advanced indexing
        flat_spatial_index = x_coord_indices * self.policy.out_height + y_coord_indices
        dim_per_layer = self.policy.out_width * self.policy.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index

        # Mask out invalid entries
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scatter values into flattened buffer, then reshape
        box_flat = torch.zeros(
            (B_TT, self.policy.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.policy.num_layers, self.policy.out_width, self.policy.out_height)

        return box_obs


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        # Dynamically determine num_layers from environment features
        # This matches what ComponentPolicy does via ObsTokenToBoxShaper
        self.num_layers = max(env.feature_normalizations.keys()) + 1

        # Define layer dimensions that are used multiple times
        self.cnn_channels = 64  # Used in cnn1 and cnn2
        self.critic_hidden_dim = 1024  # Used in critic_1 and value_head
        self.actor_hidden_dim = 512  # Used in actor_1, actor_W, and bilinear calculations
        self.action_embed_dim = 16  # Used in action_embeddings, actor_W, and bilinear calculations

        # Match YAML component initialization more closely
        # Use dynamically determined num_layers as input channels
        # Note: YAML uses orthogonal with gain=1, not sqrt(2) like pufferlib default
        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=self.num_layers, out_channels=self.cnn_channels, kernel_size=5, stride=3),
            std=1.0,  # Match YAML orthogonal gain=1
        )
        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=self.cnn_channels, out_channels=self.cnn_channels, kernel_size=3, stride=1),
            std=1.0,  # Match YAML orthogonal gain=1
        )

        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Match YAML: Linear layers use orthogonal with gain=1
        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(self.flattened_size, 128), std=1.0)
        self.encoded_obs = pufferlib.pytorch.layer_init(nn.Linear(128, self.input_size), std=1.0)

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        self.critic_1 = pufferlib.pytorch.layer_init(
            nn.Linear(self.hidden_size, self.critic_hidden_dim), std=np.sqrt(2)
        )
        # value_head has no nonlinearity (YAML: nonlinearity: null), so gain=1
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.critic_hidden_dim, 1), std=1.0)

        # Actor branch
        # actor_1 uses gain=1 (YAML default for Linear layers with ReLU)
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.actor_hidden_dim), std=1.0)

        # Action embeddings - will be properly initialized via activate_action_embeddings
        self.action_embeddings = nn.Embedding(100, self.action_embed_dim)
        self._initialize_action_embeddings()

        # Bilinear layer to match MettaActorSingleHead
        self._init_bilinear_actor()

        # Build normalization vector dynamically from environment
        # This matches what ObservationNormalizer does in ComponentPolicy
        max_values = [1.0] * self.num_layers  # Default to 1.0
        for feature_id, norm_value in env.feature_normalizations.items():
            if feature_id < self.num_layers:
                max_values[feature_id] = norm_value if norm_value > 0 else 1.0
        max_vec = torch.tensor(max_values, dtype=torch.float32)[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        # Track active actions
        self.active_action_names = []
        self.num_active_actions = 100  # Default

        self.effective_rank_enabled = True  # For critic_1 matching YAML

    def _initialize_action_embeddings(self):
        """Initialize action embeddings to match YAML ActionEmbedding component."""
        # Match the YAML component's initialization (orthogonal then scaled to max 0.1)
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)

    def _init_bilinear_actor(self):
        """Initialize bilinear actor head to match MettaActorSingleHead."""
        # Bilinear parameters matching MettaActorSingleHead
        self.actor_W = nn.Parameter(
            torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).to(dtype=torch.float32)
        )
        self.actor_bias = nn.Parameter(torch.Tensor(1).to(dtype=torch.float32))

        # Kaiming (He) initialization
        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)

    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize to environment, setting up action embeddings to match the available actions."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

        # Could implement proper action name to index mapping here if needed
        # For now, we'll use the first N embeddings

    def network_forward(self, x):
        x = x / self.max_vec
        x = self.cnn1(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after cnn1
        x = self.cnn2(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after cnn2
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after fc1
        x = self.encoded_obs(x)
        x = F.relu(x)  # ComponentPolicy has ReLU after encoded_obs
        return x

    def encode_observations(self, observations, state=None):
        """Encode observations into a hidden representation.

        This implementation matches ComponentPolicy's ObsTokenToBoxShaper exactly,
        using scatter operation for efficient token placement."""
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        B_TT = B * TT

        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

        # Don't modify original tensor - ComponentPolicy doesn't do this

        # Extract coordinates and attributes (matching ObsTokenToBoxShaper exactly)
        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = token_observations[..., 1].long()  # Shape: [B_TT, M]
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # Create mask for valid tokens (matching ComponentPolicy)
        valid_tokens = coords_byte != 0xFF

        # Additional validation: ensure atr_indices are within valid range
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr

        # Log warning for out-of-bounds indices (matching ComponentPolicy)
        invalid_atr_mask = valid_tokens & ~valid_atr
        if invalid_atr_mask.any():
            invalid_indices = atr_indices[invalid_atr_mask].unique()
            warnings.warn(
                f"Found observation attribute indices {sorted(invalid_indices.tolist())} "
                f">= num_layers ({self.num_layers}). These tokens will be ignored. "
                f"This may indicate the policy was trained with fewer observation channels.",
                stacklevel=2,
            )

        # Use scatter-based write to avoid multi-dim advanced indexing (matching ComponentPolicy)
        # Compute flattened spatial index and a combined index that encodes (layer, x, y)
        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices  # [B_TT, M]
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index  # [B_TT, M]

        # Mask out invalid entries by directing them to index 0 with value 0
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scatter values into a flattened buffer, then reshape to [B_TT, L, W, H]
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        return self.network_forward(box_obs)

    def decode_actions(self, hidden, batch_size):
        """Decode actions using bilinear interaction to match MettaActorSingleHead."""
        # Critic branch (unchanged)
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        # Actor branch with bilinear interaction
        actor_features = self.actor_1(hidden)  # [B*TT, 512]
        actor_features = F.relu(actor_features)  # ComponentPolicy has ReLU after actor_1

        # Get action embeddings for all actions
        # Use only the active actions (first num_active_actions embeddings)
        action_embeds = self.action_embeddings.weight[: self.num_active_actions]  # [num_actions, 16]

        # Expand action embeddings for each batch element
        action_embeds = action_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B*TT, num_actions, 16]

        # Bilinear interaction matching MettaActorSingleHead
        num_actions = action_embeds.shape[1]

        # Reshape for bilinear calculation
        # actor_features: [B*TT, 512] -> [B*TT * num_actions, 512]
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)  # [B*TT, num_actions, 512]
        actor_reshaped = actor_repeated.reshape(-1, self.actor_hidden_dim)  # [B*TT * num_actions, 512]
        action_embeds_reshaped = action_embeds.reshape(-1, self.action_embed_dim)  # [B*TT * num_actions, 16]

        # Perform bilinear operation using einsum (matching MettaActorSingleHead)
        query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)  # [N, 1, 16]
        query = torch.tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # [N, 1]

        biased_scores = scores + self.actor_bias  # [N, 1]

        # Reshape back to [B*TT, num_actions]
        logits = biased_scores.reshape(batch_size, num_actions)

        return logits, value
