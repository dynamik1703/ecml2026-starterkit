import os
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple, List, Dict

from flatland.envs.rail_env_action import RailEnvActions


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_size: int = 36,
        n_actions: int = 5,
        hidden_size: int = 128,
        num_hidden_layers: int = 3,
        checkpoint_path: str | None = "./submission/checkpoint.pt",
    ):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions
        layers: list[nn.Module] = [nn.Linear(obs_size, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            self.load_state_dict(ckpt["model"])
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

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def masked_logits(
        self,
        observations: Any,
        action_masks: Any | None = None,
    ) -> torch.Tensor:
        obs_t = torch.as_tensor(observations, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        features = obs_t[:, : self.obs_size]
        if action_masks is None:
            if obs_t.shape[1] >= self.obs_size + self.n_actions:
                mask = obs_t[:, self.obs_size : self.obs_size + self.n_actions]
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

        logits, _ = self(features)
        return logits.masked_fill(~valid_actions, float("-inf"))

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
