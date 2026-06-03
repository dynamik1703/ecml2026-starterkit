# RL Training Notes

This repo contains a local masked PPO trainer for the torch `ActorCritic` policy:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_masked_ppo.py \
  --updates 10 \
  --steps-per-update 512 \
  --ppo-epochs 4 \
  --minibatch-size 256 \
  --output-checkpoint checkpoints/masked_ppo.pt
```

For route-conflict features and the conflict-priority reward, use the 52-feature
observation builder. The flag keeps `obs_builder` and `obs_size` in sync:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_masked_ppo.py \
  --use-route-conflict-obs \
  --updates 10 \
  --steps-per-update 512 \
  --ppo-epochs 4 \
  --minibatch-size 256 \
  --teacher-ce-coef 0.02 \
  --conflict-priority-penalty-coef 0.05 \
  --global-slack-reward-coef 0.2 \
  --output-checkpoint checkpoints/masked_ppo_route.pt
```

Evaluate a trained checkpoint with the matching observation builder:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/evaluate_sampled.py \
  --policy submission.my_policy.MyPolicy \
  --policy-checkpoint checkpoints/masked_ppo_route.pt \
  --obs-builder submission.my_observation_builder.MyRouteConflictObservationBuilder \
  --episodes 50 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2
```

Current submission policy is `submission.hybrid_policy.MyPolicy`, which wraps the
torch actor with deterministic safety heuristics. A route-conflict checkpoint
needs the route-conflict observation builder at inference time; do not drop a
52-feature checkpoint into the current 36-feature submission path without also
switching the observation builder.
