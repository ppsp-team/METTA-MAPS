"""
Complete FastPolicy implementation with modular cascade and comparison components.
"""

from typing import Optional

import numpy as np
import pufferlib.pytorch
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.actor import (
    ActionProbs,
    ActionProbsConfig,
    ActorKey,
    ActorKeyConfig,
    ActorQuery,
    ActorQueryConfig,
)
from metta.agent.components.cascade import CascadeConfig, CascadeModule, ComparisonConfig, ComparisonModule
from metta.agent.components.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.components.lstm import LSTM, LSTMConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimBoxConfig
from metta.agent.policy import Policy, PolicyArchitecture


class FastCascadeConfig(PolicyArchitecture):
    """FastPolicy configuration with cascade and comparison components."""

    class_path: str = "metta.agent.policies.fast_cascade.FastCascadePolicy"

    # Standard components
    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(in_key="obs_normalizer", out_key="encoded_obs")
    lstm_config: LSTMConfig = LSTMConfig(
        in_key="encoded_obs", out_key="core", latent_size=128, hidden_size=128, num_layers=2
    )

    # Network dimensions
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512

    # Actor components
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig(out_key="action_embedding")
    actor_query_config: ActorQueryConfig = ActorQueryConfig(in_key="actor_1", out_key="actor_query")
    actor_key_config: ActorKeyConfig = ActorKeyConfig(
        query_key="actor_query", embedding_key="action_embedding", out_key="logits"
    )
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    # Cascade and comparison components
    cascade_config: CascadeConfig = CascadeConfig(
        in_key="core",
        out_key="core_cascaded",
        cascade_rate=1.0,  # Default disabled
    )
    comparison_config: ComparisonConfig = ComparisonConfig(
        input_key="encoded_obs", hidden_key="core_cascaded", weights_source_key="actor_1", out_key="comparison"
    )

    # Control flags
    enable_cascade: bool = True
    enable_comparison: bool = True


class FastCascadePolicy(Policy):
    """
    FastPolicy with modular cascade and comparison components.

    This policy demonstrates how to integrate cascade functionality using
    the modular component approach. Components can be easily enabled/disabled
    or reconfigured without modifying the core policy logic.
    """

    def __init__(self, env, config: Optional[FastCascadeConfig] = None):
        super().__init__()
        self.config = config or FastCascadeConfig()
        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        # Policy metadata
        self.active_action_names = []
        self.num_active_actions = 100
        self.action_index_tensor = None
        self.cum_action_max_params = None
        self.out_width = env.obs_width
        self.out_height = env.obs_height

        # Initialize standard components
        self._init_standard_components()

        # Initialize cascade and comparison components
        self._init_cascade_components()

        # Initialize actor/critic heads
        self._init_actor_critic_heads()

    def _init_standard_components(self):
        """Initialize the standard policy components."""
        self.obs_shim = ObsShimBox(env=self.env, config=self.config.obs_shim_config)
        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=self.env)
        self.lstm = LSTM(config=self.config.lstm_config)

    def _init_cascade_components(self):
        """Initialize cascade and comparison components."""
        # Cascade component
        if self.config.enable_cascade:
            self.cascade = CascadeModule(self.config.cascade_config)
        else:
            self.cascade = None

        # Comparison component (initialized after actor_1 is created)
        if self.config.enable_comparison:
            self.comparison = ComparisonModule(self.config.comparison_config, policy_reference=self)
        else:
            self.comparison = None

    def _init_actor_critic_heads(self):
        """Initialize actor and critic network heads."""
        # Determine input key for actor/critic (cascaded or regular core)
        core_key = "core_cascaded" if self.config.enable_cascade else "core"

        # Actor head
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.actor_hidden_dim), std=1.0
        )
        self.actor_1 = TDM(module, in_keys=[core_key], out_keys=["actor_1"])

        # Critic head
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = TDM(module, in_keys=[core_key], out_keys=["critic_1"])
        self.critic_activation = nn.Tanh()

        module = pufferlib.pytorch.layer_init(nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor components
        self.action_embeddings = ActionEmbedding(config=self.config.action_embedding_config)

        # Configure actor query and key components
        self.config.actor_query_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.config.actor_query_config.hidden_size = self.config.actor_hidden_dim
        self.actor_query = ActorQuery(config=self.config.actor_query_config)

        self.config.actor_key_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=self.config.actor_key_config)
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        """
        Forward pass with modular cascade and comparison components.

        The pipeline flow:
        1. obs_shim: env_obs -> obs_normalizer
        2. cnn_encoder: obs_normalizer -> encoded_obs
        3. lstm: encoded_obs -> core
        4. [cascade]: core -> core_cascaded (if enabled)
        5. [comparison]: encoded_obs + core_cascaded -> comparison (if enabled)
        6. actor/critic: core_cascaded -> logits, values
        7. action_probs: logits -> action probabilities
        """
        # Standard component pipeline
        self.obs_shim(td)
        self.cnn_encoder(td)
        self.lstm(td)

        # Apply cascade if enabled
        if self.cascade is not None:
            self.cascade(td)
        else:
            # Pass through without cascade
            td["core_cascaded"] = td["core"]

        # Apply comparison if enabled
        if self.comparison is not None:
            self.comparison(td)

        # Actor-critic pipeline
        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])

        self.critic_1(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
        self.value_head(td)

        # Action generation
        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)
        self.action_probs(td, action)

        td["values"] = td["values"].flatten()
        return td

    def set_cascade_rate(self, rate: float):
        """Dynamically adjust cascade rate."""
        if self.cascade is not None:
            self.cascade.set_cascade_rate(rate)
            # Also update config for consistency
            self.config.cascade_config.cascade_rate = rate

    def get_cascade_rate(self) -> float:
        """Get current cascade rate."""
        if self.cascade is not None:
            return self.cascade.cascade_rate
        return 1.0  # No cascade = rate of 1.0

    def reset_memory(self):
        """Reset all memory components."""
        self.lstm.reset_memory()
        if self.cascade is not None:
            self.cascade.reset_memory()

    def initialize_to_environment(self, env, device) -> List[str]:
        """Initialize policy to environment."""
        device = torch.device(device)
        self.to(device)

        logs = []
        logs.append(self.obs_shim.initialize_to_environment(env, device))
        self.action_embeddings.initialize_to_environment(env, device)
        self.action_probs.initialize_to_environment(env, device)
        return logs

    def get_agent_experience_spec(self) -> Composite:
        """Return the policy's experience specification."""
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
