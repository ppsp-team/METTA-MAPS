import torch
import torch.nn as nn
import torch.nn.init as init


class SecondOrderNetwork(nn.Module):
    def __init__(self, in_channels, alpha):
        super(SecondOrderNetwork, self).__init__()

        # Define a linear layer for comparing the difference between input and output of the first-order network
        self.comparison_layer = nn.Linear(in_features=in_channels, out_features=in_channels)

        # Linear layer for determining wagers
        self.wager = nn.Linear(in_channels, 2)
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        self.alpha = float(alpha / 100)  # EMA hyperparameter

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for stability
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def target_wager(self, rewards):
        flattened_rewards = rewards
        EMA = 0.0

        batch_size = rewards.size(0)  # Get the batch size (first dimension of rewards)
        new_tensor = torch.zeros(
            batch_size, 2, device=rewards.device
        )  # Create [batch_size, 2] tensor on the same device

        for i in range(batch_size):
            G = flattened_rewards[i]  # Current reward
            EMA = self.alpha * G + (1 - self.alpha) * EMA  # Update EMA

            # Set values based on comparison with EMA
            if G > EMA:
                new_tensor[i] = torch.tensor([1, 0], device=rewards.device)
            else:
                new_tensor[i] = torch.tensor([0, 1], device=rewards.device)

        return new_tensor

    def forward(self, comparison_matrix, prev_comparison, cascade_rate, rewards=None):
        # Pass the input through the comparison layer and apply dropout and activation
        comparison_out = self.dropout(torch.nn.functional.relu(self.comparison_layer(comparison_matrix)))

        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        # Pass through wager layer
        wager = self.wager(comparison_out)

        if rewards is not None:
            return wager, comparison_out, self.target_wager(rewards)

        else:
            return wager, comparison_out
