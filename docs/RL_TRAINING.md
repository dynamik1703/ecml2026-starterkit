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

To evaluate the same checkpoint with the deterministic hybrid safety layer,
switch the policy class:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/evaluate_sampled.py \
  --policy submission.hybrid_policy.MyPolicy \
  --policy-checkpoint checkpoints/masked_ppo_route.pt \
  --obs-builder submission.my_observation_builder.MyRouteConflictObservationBuilder \
  --episodes 50 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2
```

Trace long-corridor decisions where a move candidate points toward an opposing
train and label those decisions with the final episode outcome:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/analyze_long_corridor_decisions.py \
  --policy submission.hybrid_policy.MyPolicy \
  --episodes 200 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --output-csv /private/tmp/ecml_long_corridor_events.csv
```

Trace medium-term route intersections where a candidate route prefix crosses or
opposes another train's planned prefix with a small ETA gap:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/analyze_route_intersection_decisions.py \
  --policy submission.hybrid_policy.MyPolicy \
  --episodes 200 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --output-csv /private/tmp/ecml_route_intersection_events.csv
```

Current Docker submission policy is `submission.rerank_policy.MyPolicy`, which
wraps the torch actor with deterministic safety heuristics and a conservative
side-detour reranker. A route-conflict checkpoint needs the route-conflict
observation builder at inference time; do not drop a 52-feature checkpoint into
the current 36-feature submission path without also switching the observation
builder.

Rejected shortcut: a hard future-head-on yield rule that stopped the lower
priority train for reverse-edge conflicts with ETA gap <= 1 and own step <= 20
looked promising on some failures but degraded the 50-seed benchmark from
`reward_mean=0.912794, success_rate_mean=0.946667` to
`reward_mean=0.699032, success_rate_mean=0.94`. Treat future intersections as
features or training labels first, not as broad hard masking.

Current candidate/default policy: `submission.rerank_policy.MyPolicy` keeps the
hybrid safety pipeline but tries a conservative side-detour rerank when a
`MOVE_FORWARD` candidate creates a near future reverse-edge conflict with a
higher-priority train. It never reranks an existing left/right decision and does
not introduce extra stop/yield actions.

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/evaluate_sampled.py \
  --policy submission.rerank_policy.MyPolicy \
  --episodes 200 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2
```

Current local measurements:
- Seeds 10-59: hybrid `0.912794 / 0.946667`, rerank `0.912459 / 0.946667`.
- Seeds 10-209: hybrid `0.913843 / 0.938333`, rerank `0.914221 / 0.939167`.
- Seeds 210-309: hybrid `0.924562 / 0.946667`, rerank `0.928594 / 0.948333`.
- Seeds 310-509: hybrid `0.911885 / 0.935000`, rerank `0.913035 / 0.934167`.
- Seeds 510-709: hybrid `0.920141 / 0.937500`, rerank `0.924645 / 0.942500`.
- Aggregate seeds 10-709 over the non-overlapping windows above:
  hybrid `0.916614 / 0.938333`, rerank `0.918914 / 0.940000`.

Use the seedwise comparison tool to find reranker gains and regressions:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/compare_policies.py \
  --episodes 50 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --output-csv /private/tmp/ecml_compare_10_50.csv
```

For seeds 10-59, the reranker changes only two seeds: seed 12 improves
`+0.080593` reward with unchanged success, while seed 54 regresses `-0.097352`
reward with unchanged success. This makes seed 54 the next target for a
conservative reranker guard.
