import os
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple, List, Dict
from torch.distributions import Categorical

from flatland.envs.rail_env_action import RailEnvActions


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_size: int | None = None,
        n_actions: int | None = None,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        checkpoint_path: str | None = "./submission/checkpoint.pt",
    ):
        super().__init__()
        ckpt = None
        ckpt_config = {}
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        obs_size = int(obs_size if obs_size is not None else ckpt_config.get("obs_size", 36))
        n_actions = int(n_actions if n_actions is not None else ckpt_config.get("n_actions", 5))
        hidden_size = int(
            hidden_size
            if hidden_size is not None
            else ckpt_config.get("hidden_size", 128)
        )
        num_hidden_layers = int(
            num_hidden_layers
            if num_hidden_layers is not None
            else ckpt_config.get("num_hidden_layers", 3)
        )
        self.obs_size = obs_size
        self.n_actions = n_actions
        layers: list[nn.Module] = [nn.Linear(obs_size, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        if ckpt is not None:
            self._load_compatible_state_dict(ckpt["model"])
            self.eval()
        else:
            """Orthogonal init for an ActorCritic-style net with trunk + policy/value heads."""
            for module in self.trunk:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
            nn.init.zeros_(self.policy_head.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

    def _load_compatible_state_dict(self, checkpoint_state: dict[str, torch.Tensor]) -> None:
        current_state = self.state_dict()
        merged_state = dict(current_state)
        for name, checkpoint_tensor in checkpoint_state.items():
            if name not in current_state:
                continue
            current_tensor = current_state[name]
            if current_tensor.shape == checkpoint_tensor.shape:
                merged_state[name] = checkpoint_tensor
                continue
            if name == "trunk.0.weight" and current_tensor.ndim == checkpoint_tensor.ndim == 2:
                expanded = current_tensor.clone()
                rows = min(expanded.shape[0], checkpoint_tensor.shape[0])
                cols = min(expanded.shape[1], checkpoint_tensor.shape[1])
                expanded.zero_()
                expanded[:rows, :cols] = checkpoint_tensor[:rows, :cols]
                merged_state[name] = expanded
        self.load_state_dict(merged_state)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def _features_and_valid_actions(
        self,
        observations: Any,
        action_masks: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.policy_head.weight.device
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        features = obs_t[:, : self.obs_size]
        if action_masks is None:
            if obs_t.shape[1] >= self.obs_size + self.n_actions:
                mask = obs_t[:, -self.n_actions :]
            else:
                mask = torch.ones(
                    (obs_t.shape[0], self.n_actions),
                    dtype=torch.float32,
                    device=obs_t.device,
                )
        else:
            mask = torch.as_tensor(action_masks, dtype=torch.float32, device=obs_t.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)

        valid_actions = mask >= 0.5
        no_valid_actions = ~valid_actions.any(dim=1)
        if no_valid_actions.any():
            valid_actions = valid_actions.clone()
            valid_actions[no_valid_actions, 0] = True

        return features, valid_actions

    def masked_forward(
        self,
        observations: Any,
        action_masks: Any | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, valid_actions = self._features_and_valid_actions(
            observations,
            action_masks,
        )
        logits, values = self(features)
        return logits.masked_fill(~valid_actions, float("-inf")), values

    def masked_logits(
        self,
        observations: Any,
        action_masks: Any | None = None,
    ) -> torch.Tensor:
        logits, _ = self.masked_forward(observations, action_masks)
        return logits

    def action_distribution(
        self,
        observations: Any,
        action_masks: Any | None = None,
    ) -> Categorical:
        return Categorical(logits=self.masked_logits(observations, action_masks))

    def sample_actions(
        self,
        observations: Any,
        action_masks: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.masked_forward(observations, action_masks)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), values

    def evaluate_actions(
        self,
        observations: Any,
        actions: Any,
        action_masks: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.masked_forward(observations, action_masks)
        distribution = Categorical(logits=logits)
        action_t = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
        return distribution.log_prob(action_t), distribution.entropy(), values

    def act_many(
        self, handles: List[int], observations: List[Any], **kwargs
    ) -> Dict[int, RailEnvActions]:
        if not handles:
            return {}

        with torch.no_grad():
            logits = self.masked_logits(np.asarray(observations, dtype=np.float32))
            actions = logits.argmax(dim=-1).cpu().numpy()
        return {
            handle: int(action)
            for handle, action in zip(handles, actions)
        }

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        with torch.no_grad():
            logits = self.masked_logits(observation)
        return int(logits.argmax(dim=-1).item())


MyPolicy = ActorCritic
