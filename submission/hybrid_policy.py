from __future__ import annotations

from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.my_policy import ActorCritic
from submission.reservation_policy import ReservationPolicy


class HybridPolicy:
    def __init__(self):
        self.rl_policy = ActorCritic()
        self.reservation_policy = ReservationPolicy()

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return self.rl_policy.act(observation, **kwargs)

    def act_many(
        self, handles: List[int], observations: List[Any], **kwargs
    ) -> Dict[int, RailEnvActions]:
        context = runtime_context.get()
        env = context.env
        if env is not None and env.get_num_agents() <= 2:
            return self.reservation_policy.act_many(handles, observations, **kwargs)
        return self.rl_policy.act_many(handles, observations, **kwargs)


MyPolicy = HybridPolicy
