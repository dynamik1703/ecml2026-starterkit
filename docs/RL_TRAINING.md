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

The default teacher for behavior cloning and PPO teacher regularization is the
current guarded rerank policy, `submission.rerank_policy.MyPolicy`. Override
`--teacher-policy` to compare against older teachers.

Behavior-clone the current teacher into a torch checkpoint:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_behavior_clone.py \
  --episodes 20 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --epochs 3 \
  --output-checkpoint /private/tmp/ecml_bc_rerank_20.pt
```

Inspect first synchronized action differences between guarded rerank and a BC
checkpoint:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/analyze_policy_action_diffs.py \
  --candidate-checkpoint /private/tmp/ecml_bc_rerank_20.pt \
  --seeds 55,56,207,305,519,548,626 \
  --num-agents 6 \
  --line-length 2 \
  --output-csv /private/tmp/ecml_bc_action_diffs_known.csv
```

Collect one-step counterfactual labels at critical guarded-rerank decisions.
The tool first runs the baseline episode, records critical decisions, then
reruns the same seed while forcing one valid alternative for exactly one step:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/counterfactual_decision_eval.py \
  --episodes 5 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --max-decisions-per-seed 8 \
  --max-alternatives-per-decision 2 \
  --output-csv /private/tmp/ecml_counterfactual_decisions.csv
```

For movement-only gate labels, restrict alternatives to route-changing actions:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/counterfactual_decision_eval.py \
  --episodes 20 \
  --seed 50 \
  --num-agents 6 \
  --line-length 2 \
  --forced-actions LEFT,FORWARD,RIGHT \
  --max-decisions-per-seed 6 \
  --max-alternatives-per-decision 2 \
  --output-csv /private/tmp/ecml_counterfactual_move_50_20.csv
```

Each row is a candidate training example for a learned gate/reranker:
`reward_delta`, `success_delta`, and `failed_agents_delta` label whether the
forced action improved the baseline. Feature columns include observation
scalars, action masks, per-action target distances, corridor lengths,
future-head-on risk, policy logit deltas, and route-prefix conflict features.
The prefix features compare baseline versus forced lookahead routes against
other agents' planned prefixes: shared cells, same/opposing/crossing direction
intersections, same-edge conflicts, head-on edge conflicts, first intersection
step, and ETA gaps.

First fresh prefix-feature batches:
- Seeds 740-749 produced 107 rows with reward wins/losses/ties `0/8/99` and
  success wins/losses/ties `0/0/107`; useful mostly as hard negatives.
- Seeds 720-729 produced 120 rows with reward wins/losses/ties `8/8/104` and
  success wins/losses/ties `10/0/110`.
- Combined 227-row diagnostic set: 10 good, 14 bad, 203 neutral. With five
  shuffled seed splits, a 32-hidden weighted binary gate accepted
  good/neutral/bad `4/0/0` at threshold 0.90 and `3/0/0` at threshold 0.95.
  The multiclass gate with `--max-bad-probability 0.01` accepted `4/1/0` at
  threshold 0.90 and `3/1/0` at threshold 0.95. This is a small, mined sample,
  but the zero-bad validation result is the first positive sign that explicit
  prefix-conflict features may make a learned gate viable.

Initial smoke test on seeds 10 and 55 with three decisions per seed produced
13 labels: reward wins/losses/ties `1/8/4`, success wins/losses/ties `0/0/13`.
The positive label was the known useful seed-55 `MOVE_RIGHT -> MOVE_FORWARD`
override with reward delta `+0.084175`; most negative labels were unnecessary
one-step `STOP_MOVING` overrides.

Initial movement-only sample on seeds 50-69 produced 240 labels:
reward wins/losses/ties `26/11/203`, success wins/losses/ties `7/7/226`.
Using a conservative gate label (`success_delta > 0` or positive reward without
success loss) yielded 25 positives. A first tiny MLP gate fit this small sample
too easily, so we expanded the data and switched the trainer to seed-grouped
shuffle validation.

Expanded movement-only sample across seed windows 50-69, 100-111, 200-211,
320-331 and 648-659 produced 814 labels from 68 seeds:
- Conservative labels: 67 good, 24 bad, 723 neutral.
- By transition: `MOVE_FORWARD->MOVE_LEFT` 16/9/237, `MOVE_FORWARD->MOVE_RIGHT`
  24/8/316, `MOVE_LEFT->MOVE_FORWARD` 14/2/88, `MOVE_RIGHT->MOVE_FORWARD`
  12/5/82, `STOP_MOVING->MOVE_FORWARD` 1/0/0 (good/bad/neutral).
- Linear gate at threshold 0.90 on validation accepted good/neutral/bad
  `1/1/1`; too risky because it accepted a bad override.
- 32-hidden-unit MLP at threshold 0.90 on validation accepted
  good/neutral/bad `1/4/0`; at threshold 0.95 it accepted `1/2/0`.
  This is directionally useful as a very conservative gate, but recall is still
  too low to justify changing the submission default.
- Cross-split validation over split seeds 1, 3, 5, 7, 11, 13, 17 and 19 showed
  the unweighted positive-vs-rest MLP was not robust enough: at threshold 0.95
  it still accepted aggregate good/neutral/bad `8/22/2`.
- The trainer now uses category-specific sample weights, defaulting to
  good/neutral/bad `8.0/0.25/8.0`, so bad counterfactual overrides are not
  drowned out by the many neutral negatives. With the same 8 split seeds:
  - threshold 0.90 accepted `11/48/7`;
  - threshold 0.95 accepted `9/17/0`;
  - threshold 0.97 accepted `6/3/0`.
  This satisfies the first safety criterion for analysis (`accepted_bad=0` at
  high threshold), but recall remains low. Keep this as an experimental learned
  gate signal, not the Docker default.
- A later movement-only batch on seeds 720-739 added 240 labels with reward
  wins/losses/ties `17/21/202` and success wins/losses/ties `10/3/227`.
  The combined 1054-row set has 80 good, 82 bad and 892 neutral labels.
  Re-running the weighted binary gate on this broader set showed that the
  earlier zero-bad result was not robust:
  - threshold 0.90 accepted `20/55/4`;
  - threshold 0.95 accepted `13/12/3`;
  - threshold 0.97 accepted `7/2/1`.
  Stronger bad weights and higher thresholds did not remove all bad accepts
  without collapsing useful recall.
- The trainer also supports `--objective multiclass`, learning separate
  neutral/good/bad logits and accepting only actions with high `P(good)` and
  low `P(bad)`. On the 1054-row set this still left isolated bad accepts even
  with `--max-bad-probability 0.001`, so the current feature set is not yet
  separable enough for an online learned gate deployment.

Train a first gate classifier from one or more counterfactual CSVs:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_counterfactual_gate.py \
  /private/tmp/ecml_counterfactual_move_50_20.csv \
  --epochs 500 \
  --hidden-size 32 \
  --thresholds 0.75 0.9 0.95 0.97 \
  --output-checkpoint /private/tmp/ecml_counterfactual_gate_move.pt
```

For the three-class diagnostic variant:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_counterfactual_gate.py \
  /private/tmp/ecml_counterfactual_move_50_20.csv \
  /private/tmp/ecml_counterfactual_move_720_20.csv \
  --objective multiclass \
  --max-bad-probability 0.01 \
  --epochs 500 \
  --hidden-size 32
```

Current behavior-cloning findings:
- `seed=10`, `episodes=20`, `epochs=3` collected 50,408 valid samples and only
  28 teacher/reference disagreements. Pure `ActorCritic` evaluation on seeds
  10-59 was weak: `0.857456 / 0.920000`.
- The same checkpoint inside `HybridPolicy` scored `0.915853 / 0.946667` on
  seeds 10-59.
- The same checkpoint inside `RerankPolicy` scored `0.917465 / 0.946667` on
  seeds 10-59, versus current guarded rerank `0.914406 / 0.946667`.
- On seeds 10-209, that BC+Rerank checkpoint scored `0.917233 / 0.941667`
  versus current guarded rerank `0.915676 / 0.940000`: reward delta
  `+0.001556`, success delta `+0.001667`, reward wins/losses `3/1`, success
  wins/losses `2/0`.
- Broader BC+Rerank validation of `/private/tmp/ecml_bc_rerank_20.pt` did not
  justify direct submission replacement:
  - Seeds 210-309: `0.925978 / 0.946667` versus guarded rerank
    `0.928594 / 0.948333`; reward wins/losses `1/3`, success wins/losses
    `0/1`.
  - Seeds 310-509: `0.912208 / 0.935000` versus guarded rerank
    `0.912328 / 0.934167`; reward wins/losses `3/4`, success wins/losses
    `1/0`.
  - Seeds 510-709: `0.923788 / 0.941667` versus guarded rerank
    `0.925131 / 0.944167`; reward wins/losses `4/6`, success wins/losses
    `1/4`.
  - Aggregate seeds 10-709: approximately `-0.000347` reward and `-0.000238`
    success versus guarded rerank, with several hard success regressions. Keep
    this as a training signal, not as the current submission checkpoint.
- First-difference action diagnostics show that many true BC changes are
  `MOVE_RIGHT -> MOVE_FORWARD`; this creates wins such as seeds 55, 56 and 207,
  but also hard losses such as seed 305. Several broad-window losses are
  `MOVE_FORWARD -> MOVE_LEFT` changes, e.g. seeds 519, 548, 626, 497 and 603.
  A useful learned-action gate must distinguish these contexts before accepting
  BC actions online.
- Experimental `submission.bc_gated_policy.MyPolicy` keeps guarded rerank as
  the default action source and accepts only narrow BC checkpoint deviations.
  With `/private/tmp/ecml_bc_rerank_20.pt`, broad validation versus guarded
  rerank found no regressions over seeds 10-709:
  - Seeds 10-209: `0.916097 / 0.940000` versus `0.915676 / 0.940000`;
    reward wins/losses `1/0`, success wins/losses `0/0`.
  - Seeds 210-309: unchanged `0.928594 / 0.948333`; reward wins/losses `0/0`,
    success wins/losses `0/0`.
  - Seeds 310-509: `0.912683 / 0.935000` versus `0.912328 / 0.934167`;
    reward wins/losses `1/0`, success wins/losses `1/0`.
  - Seeds 510-709: `0.925526 / 0.944167` versus `0.925131 / 0.944167`;
    reward wins/losses `1/0`, success wins/losses `0/0`.
  - Aggregate seeds 10-709: approximately `+0.000335` reward and `+0.000238`
    success, with reward wins on seeds 55, 325 and 653 and no observed losses.
    This is promising as an ensemble/gating component, but still too narrow to
    replace the Docker default before hidden-distribution validation.
  - OOD canaries on seeds 800-849 found one pre-guard regression on `scene_1`
    seed 836: a `STOP_MOVING -> MOVE_LEFT` accept had an infinite candidate
    distance to the current waypoint. The gate now rejects BC actions whose
    post-action waypoint distance is not finite. After this guard:
    - `scene_1`, 6 agents, line length 2: `0.863753 / 0.840000` versus
      `0.863739 / 0.840000`; reward wins/losses `1/0`, success wins/losses
      `0/0`.
    - `scene_3`, 6 agents, line length 2: unchanged `0.863514 / 0.900000`;
      reward wins/losses `0/0`, success wins/losses `0/0`.
    - `scene_5`, 4 agents, line length 2: unchanged `0.900808 / 0.945000`;
      reward wins/losses `0/0`, success wins/losses `0/0`.
    - `scene_5`, 8 agents, line length 2: unchanged `0.896894 / 0.917500`;
      reward wins/losses `0/0`, success wins/losses `0/0`.
    - `scene_5`, 6 agents, line length 3: unchanged `0.896815 / 0.916667`;
      reward wins/losses `0/0`, success wins/losses `0/0`.
    - `scene_5`, 6 agents, line length 1 failed during environment reset with
      Flatland `line_generators.py` `UnboundLocalError: cur_agent_target`,
      before policy evaluation.
- A disjoint BC run trained on seeds 1000-1049 was also net-positive on
  seeds 10-209 (`0.916828 / 0.940833`) but had more downside: reward
  wins/losses `8/3`, success wins/losses `4/1`. Treat the BC checkpoint as a
  promising candidate, not yet as the submission checkpoint.

Rejected PPO smoke after BC:
- Init checkpoint `/private/tmp/ecml_bc_rerank_20.pt`, 3 PPO updates,
  512 steps/update, `learning_rate=0.00005`, `teacher_ce_coef=0.02`,
  `ent_coef=0.005`.
- `RerankPolicy` evaluation on seeds 10-59 dropped to `0.907533 / 0.930000`
  versus guarded rerank `0.914406 / 0.946667`.
- The run had reward wins/losses `2/7` and success wins/losses `0/4`.
  PPO needs a more conservative setup before it is useful: complete-episode
  rollouts, reward/value normalization, and stronger KL/teacher anchoring.

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

Current guarded rerank local measurements:
- Seeds 10-59: hybrid `0.912794 / 0.946667`, rerank `0.914406 / 0.946667`.
- Seeds 10-209: hybrid `0.913843 / 0.938333`, rerank `0.915676 / 0.940000`.

Historical unguarded rerank measurements before the left-detour slack guard:
- Seeds 10-209: hybrid `0.913843 / 0.938333`, rerank `0.914221 / 0.939167`.
- Seeds 210-309: hybrid `0.924562 / 0.946667`, rerank `0.928594 / 0.948333`.
- Seeds 310-509: hybrid `0.911885 / 0.935000`, rerank `0.913035 / 0.934167`.
- Seeds 510-709: hybrid `0.920141 / 0.937500`, rerank `0.924645 / 0.942500`.
- Historical aggregate seeds 10-709 over the non-overlapping windows above:
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

Before the guard, seeds 10-59 changed only two seeds: seed 12 improved
`+0.080593` reward with unchanged success, while seed 54 regressed `-0.097352`
reward with unchanged success. The current guard keeps seed 12 and blocks seed
54.

For seeds 10-209, the current guarded reranker changes five seeds, all positive:
seed 12 `+0.080593`, seed 79 `+0.093583`, seed 131 `+0.071347`, seed 138
`+0.091575`, and seed 181 `+0.029551`. It blocks the previous regressions on
seed 54 and seed 118.
