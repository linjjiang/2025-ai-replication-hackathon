"""
Model definitions for the retro-cue RNN replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions.normal import Normal


@dataclass
class ModelConfig:
    input_dim: int = 36  # 2 cue + 17 + 17 colour channels
    hidden_dim: int = 200
    output_dim: int = 17
    noise_std: float = 0.07
    device: str = "cpu"


class RetroCueRNN(nn.Module):
    """Single-layer vanilla RNN with ReLU activations and softmax output."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rnn = nn.RNN(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            nonlinearity="relu",
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim, config.output_dim)
        self._init_parameters()
        self.to(config.device)

    def _init_parameters(self) -> None:
        # Orthogonal init for recurrent weights, Xavier for others (Methods).
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        noise: Optional[torch.distributions.Distribution] = None,
        noise_std: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: shape (batch, time, input_dim)
            noise: optional distribution for additive recurrent noise
            noise_std: convenience stddev for Gaussian noise
            initial_state: optional initial hidden state (1, batch, hidden_dim)
        Returns:
            logits over time (batch, time, output_dim)
            hidden states (batch, time, hidden_dim)
        """
        batch, time, _ = inputs.shape
        device = inputs.device
        h_t = torch.zeros(1, batch, self.config.hidden_dim, device=device)
        if initial_state is not None:
            h_t = initial_state
        if noise is None and noise_std is not None and noise_std > 0:
            noise = Normal(loc=0.0, scale=noise_std)

        outputs = []
        states = []
        weight_hh = self.rnn.weight_hh_l0
        weight_ih = self.rnn.weight_ih_l0
        bias_hh = self.rnn.bias_hh_l0
        bias_ih = self.rnn.bias_ih_l0
        for t in range(time):
            x_t = inputs[:, t, :]
            pre = torch.matmul(h_t, weight_hh.T) + torch.matmul(x_t, weight_ih.T)
            pre = pre + bias_hh + bias_ih
            if noise is not None:
                pre = pre + noise.sample(pre.shape).to(pre.device)
            h_t = torch.relu(pre)
            states.append(h_t.squeeze(0))
            outputs.append(self.output(h_t.squeeze(0)))
        logits = torch.stack(outputs, dim=1)
        hidden = torch.stack(states, dim=1)
        return logits, hidden

    def output_probs(self, inputs: torch.Tensor, noise_std: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = None
        if noise_std is not None and noise_std > 0:
            dist = torch.distributions.Normal(loc=0.0, scale=noise_std)
        logits, hidden = self(inputs, noise=dist, noise_std=noise_std)
        probs = torch.softmax(logits, dim=-1)
        return probs, hidden

