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

For better signal density, mine failure-rich seeds first:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/mine_failure_seeds.py \
  --episodes 40 \
  --seed 720 \
  --num-agents 6 \
  --line-length 2 \
  --top-k 12 \
  --print-counterfactual-command \
  --output-csv /private/tmp/ecml_failure_720_40.csv \
  --output-json /private/tmp/ecml_failure_720_40.json
```

Then focus the counterfactual sampler on the failed agents from that JSON:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/counterfactual_decision_eval.py \
  --seeds 723,742,754,744,722,725,758,748,733,729,749,736 \
  --policy submission.rerank_policy.MyPolicy \
  --num-agents 6 \
  --line-length 2 \
  --forced-actions LEFT,FORWARD,RIGHT \
  --max-decisions-per-seed 6 \
  --max-alternatives-per-decision 2 \
  --focus-failures-json /private/tmp/ecml_failure_720_40.json \
  --output-csv /private/tmp/ecml_counterfactual_failure_focus_720_40_top12.csv
```

For late-arrival failures, focus decisions around each failed agent's deadline
instead of the whole failed-agent trajectory:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/counterfactual_decision_eval.py \
  --seeds 723,742,754,744,722,725,758,748,733,729,749,736 \
  --policy submission.rerank_policy.MyPolicy \
  --num-agents 6 \
  --line-length 2 \
  --forced-actions LEFT,FORWARD,RIGHT \
  --max-decisions-per-seed 6 \
  --max-alternatives-per-decision 2 \
  --focus-failures-json /private/tmp/ecml_failure_720_40.json \
  --focus-window-before-deadline 300 \
  --focus-window-after-deadline 60 \
  --output-csv /private/tmp/ecml_counterfactual_failure_deadline_720_40_top12.csv
```

For hard-negative regression data, select completed episodes with the lowest
normalized reward. These are full-success baselines where small forced changes
can still break a later arrival:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/mine_failure_seeds.py \
  --episodes 80 \
  --seed 1000 \
  --num-agents 6 \
  --line-length 2 \
  --selection-mode full-success-low-reward \
  --top-k 24 \
  --print-counterfactual-command \
  --output-csv /private/tmp/ecml_eval_1000_80.csv \
  --output-json /private/tmp/ecml_eval_1000_80.json \
  --counterfactual-output /private/tmp/ecml_counterfactual_success_hardneg_1000_80_top24.csv
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

Rank feature contrasts between outcome classes:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache \
  .venv/bin/python tools/analyze_counterfactual_features.py \
  /private/tmp/ecml_counterfactual_success_hardneg_1000_80_top24.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_1080_80_top20.csv \
  --comparison bad:rest \
  --top-k 20 \
  --output-csv /private/tmp/ecml_success_hardneg_feature_bad_rest.csv
```

Evaluate simple prefix/ETA safety rules against counterfactual labels:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache \
  .venv/bin/python tools/evaluate_counterfactual_safety_rules.py \
  /private/tmp/ecml_counterfactual_failure_deadline_720_40_top12.csv \
  /private/tmp/ecml_counterfactual_failure_deadline_760_80_top20.csv \
  /private/tmp/ecml_counterfactual_failure_deadline_840_80_top20.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_920_80_top20.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_1000_80_top24.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_1080_80_top20.csv \
  --top-k 20 \
  --output-csv /private/tmp/ecml_safety_rules_all_1193.csv
```

Evaluate monotone rescue rules that require the forced action to improve or not
worsen selected conflict/risk signals relative to the baseline:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache \
  .venv/bin/python tools/evaluate_counterfactual_rescue_rules.py \
  /private/tmp/ecml_counterfactual_failure_deadline_720_40_top12.csv \
  /private/tmp/ecml_counterfactual_failure_deadline_760_80_top20.csv \
  /private/tmp/ecml_counterfactual_failure_deadline_840_80_top20.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_920_80_top20.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_1000_80_top24.csv \
  /private/tmp/ecml_counterfactual_success_hardneg_1080_80_top20.csv \
  --top-k 20 \
  --output-csv /private/tmp/ecml_rescue_rules_all_1193.csv
```

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
- Expanding the fresh prefix-feature set with seeds 730-739 and 750-759
  produced 467 rows: 32 good, 403 neutral, 32 bad. Eight shuffled split seeds
  showed the 227-row result was not yet robust:
  - Binary gate, all 83 features: at thresholds 0.90/0.95 accepted
    `8/22/7` and `5/14/3` good/neutral/bad.
  - Binary gate, without prefix features: at thresholds 0.90/0.95 accepted
    `8/30/3` and `5/16/2`.
  - Multiclass gate, all features: at thresholds 0.90/0.95 accepted
    `6/12/5` and `6/11/4`.
  - Multiclass gate, without prefix features: at thresholds 0.90/0.95 accepted
    `11/25/4` and `6/11/2`.
  More conservative thresholds and stronger bad weights only reached zero-bad
  when useful accepts collapsed to zero. Feature summaries show good and bad
  overrides still overlap strongly on prefix-conflict deltas. Conclusion:
  prefix features help analysis, but the learned gate still needs either more
  targeted data, better temporal/conflict features, or a different model before
  online deployment.
- Failure-focused mining is a better data source than broad critical-decision
  sampling. On seeds 720-759, `tools/mine_failure_seeds.py` found 16 incomplete
  seeds and selected the 12 worst (`723,742,754,744,722,725,758,748,733,729,
  749,736`). Sampling only the failed agents in those seeds produced 81 rows:
  19 good, 59 neutral, 3 bad, with reward wins/losses/ties `19/3/59` and
  success wins/losses/ties `4/0/77`. Five shuffled split seeds on this mined
  set were much cleaner:
  - Binary gate, all features: threshold 0.90 accepted `11/2/0`;
    threshold 0.95 accepted `11/1/0`.
  - Binary gate, without prefix features: threshold 0.90 accepted `11/1/0`;
    threshold 0.95 accepted `11/0/0`.
  - Multiclass gate, all features: threshold 0.90 accepted `12/1/0`;
    threshold 0.95 accepted `11/0/0`.
  This is not deployment evidence yet because the mined set contains only
  three bad examples, but it validates the next data strategy: mine failure
  seeds and failed-agent decisions first, then expand bad/hard-negative coverage.
- Deadline-window focusing on the same 12 seeds with
  `--focus-window-before-deadline 300 --focus-window-after-deadline 60`
  produced 67 rows: 20 good, 45 neutral, 2 bad, with reward wins/losses/ties
  `20/2/45` and success wins/losses/ties `4/0/63`. Five shuffled split seeds:
  - Binary gate, all features: threshold 0.90 accepted `14/3/0`;
    threshold 0.95 accepted `11/2/0`.
  - Binary gate, without prefix features: threshold 0.90 accepted `11/1/0`;
    threshold 0.95 accepted `11/0/0`.
  - Multiclass gate, all features: threshold 0.90 accepted `11/2/0`;
    threshold 0.95 accepted `11/1/0`.
  - Multiclass gate, without prefix features: threshold 0.90 accepted `11/3/0`;
    threshold 0.95 accepted `11/0/0`.
  This deadline window is better than a last-stationary-progress window for
  Flatland late-arrival failures, because many failed agents keep moving until
  episode end and simply miss `latest_arrival`.
- A fresh failure-mined range 760-839 produced 32 incomplete seeds from 80
  episodes. The selected top-20 seeds
  (`803,779,761,836,763,773,815,829,783,812,762,781,809,792,789,819,782,780,824,814`)
  yielded 121 deadline-window rows: 19 good, 94 neutral, 8 bad, with reward
  wins/losses/ties `21/6/94` and success wins/losses/ties `7/2/112`.
  This range is harder and more useful for hard-negative coverage than the
  earlier 720-759 range.
- Combining the 720-759 and 760-839 deadline-window sets gives 188 rows:
  39 good, 139 neutral, 10 bad. Eight shuffled split seeds:
  - Binary only-prefix gate: threshold 0.90 accepted `28/23/1`;
    threshold 0.95 accepted `20/15/0`.
  - Multiclass only-prefix gate: threshold 0.90 accepted `21/18/0`;
    threshold 0.95 accepted `17/16/0`.
  This is the first broader zero-bad result with nontrivial recall, but it is
  still mined-data validation, not deployment validation.
- Explicit window-holdout is stricter and exposes remaining generalization
  risk:
  - Train 720-759, validate 760-839, binary only-prefix: threshold 0.95
    accepted `5/5/2`; not safe.
  - Train 760-839, validate 720-759, binary only-prefix: threshold 0.95
    accepted `3/4/0`; safe but low recall.
  - Train 720-759, validate 760-839, multiclass only-prefix: threshold 0.95
    accepted `4/4/4`; not safe.
  Conclusion: the 760-839 hard negatives are important, and we need more
  independent failure-mined windows before considering an online learned gate.
- A third independent failure-mined range 840-919 produced 25 incomplete seeds
  from 80 episodes. The selected top-20 seeds
  (`896,901,877,864,885,915,855,879,853,870,909,847,891,898,906,892,903,916,840,886`)
  yielded 144 deadline-window rows: 38 good, 103 neutral, 3 bad, with reward
  wins/losses/ties `37/4/103` and success wins/losses/ties `15/0/129`.
  The range is strongly positive-heavy; useful for recall, but weak for bad
  coverage.
- Combining the 720-759, 760-839 and 840-919 deadline-window sets gives
  332 rows: 77 good, 242 neutral, 13 bad. Eight shuffled split seeds:
  - Binary all features: threshold 0.95 accepted `74/27/5`; not safe.
  - Multiclass all features: threshold 0.95 accepted `83/22/4`; not safe.
  - Binary only-prefix: threshold 0.95 accepted `44/30/7`; not safe.
  - Multiclass only-prefix: threshold 0.95 accepted `42/31/7`; not safe.
  Random seed splits over mined data are still too optimistic for deployment.
- Leave-window-out validation over the three mined ranges:
  - Train 720-759 + 760-839, validate 840-919, multiclass all features:
    threshold 0.95 accepted `11/4/0`; clean holdout.
  - Train 720-759 + 840-919, validate 760-839, multiclass all features:
    threshold 0.95 accepted `8/7/3`; not safe.
  - Train 760-839 + 840-919, validate 720-759, multiclass all features:
    threshold 0.95 accepted `15/6/1`; threshold 0.97 accepted `13/6/0`.
  Conclusion: the current gate is improving as a diagnostic/ranking model, but
  760-839-like hard negatives still break generalization. The next data step
  should deliberately mine more hard-negative failure windows, not just more
  positive rescue examples.
- Successful-seed hard-negative mining from evaluated seeds 920-999 selected
  20 full-success seeds with lower baseline reward
  (`997,985,957,983,974,977,931,933,996,941,959,950,991,922,947,934,965,937,975,940`)
  and sampled broad movement counterfactuals. This produced 237 rows: 16 good,
  200 neutral, 21 bad, with reward wins/losses/ties `19/18/200` and success
  wins/losses/ties `0/7/230`. Several bad examples have positive immediate
  reward but lose a successful arrival later, so they are exactly the kind of
  hard negatives a safe learned gate must reject.
- On the successful-seed hard-negative set alone, random seed splits can learn
  some safe low-recall decisions: binary only-prefix at threshold 0.95 accepted
  `6/5/0`, and multiclass all-features at threshold 0.95 accepted `2/2/0`.
  This is useful local signal, not deployment evidence.
- Training on the three failure-deadline windows and validating on the
  successful-seed hard negatives fails the safety requirement: multiclass
  all-features accepted `9/34/8` at threshold 0.95 and `8/22/5` at threshold
  0.97. The failure-focused positives do not teach the gate enough about
  preserving already-successful episodes.
- Combining the three failure-deadline windows with the successful-seed hard
  negatives gives 569 rows: 93 good, 442 neutral, 34 bad. Random split
  validation is still not safe: multiclass all-features at threshold 0.95
  accepted `39/36/13`; binary all-features accepted `42/42/12`.
- Adding successful-seed hard negatives to the training side improves some
  leave-window-out safety but reduces recall and still does not solve the
  hardest 760-839 holdout:
  - Train 720-759 + 760-839 + successful hard negatives, validate 840-919:
    threshold 0.75 accepted `15/3/0`; threshold 0.95 accepted `8/1/0`.
  - Train 720-759 + 840-919 + successful hard negatives, validate 760-839:
    threshold 0.95 accepted `5/5/1`; still not safe.
  - Train 760-839 + 840-919 + successful hard negatives, validate 720-759:
    threshold 0.97 accepted `4/5/0`, but lower thresholds accept one bad.
  Conclusion: successful-seed hard negatives are valuable regression data, but
  the learned gate should remain offline-only. The next best data step is to
  mine more 760-like success-loss negatives and validate with explicit
  seed-window holdouts before any online gate/reranker deployment.
- A second successful-seed hard-negative range 1000-1079 was evaluated with
  reward mean `0.909273` and success-rate mean `0.939583`. The 24 lowest-reward
  full-success seeds
  (`1003,1051,1072,1077,1033,1019,1050,1020,1078,1025,1015,1059,1047,1048,1065,1027,1076,1007,1037,1035,1000,1056,1057,1066`)
  produced 369 broad movement counterfactual rows: 21 good, 306 neutral,
  42 bad, with reward wins/losses/ties `23/40/306` and success
  wins/losses/ties `0/19/350`. This is a much harder negative set than the
  920-999 block; several alternatives lose one or two successful arrivals.
- Random seed splits on the 1000-1079 successful hard-negative block confirm
  that this data is difficult: binary only-prefix at threshold 0.90 accepted
  `6/6/1`, and threshold 0.97 accepted only `1/1/0`; multiclass variants mostly
  accepted bad actions or collapsed to zero-good accepts.
- Combining the three failure-deadline windows plus both successful hard-negative
  blocks gives 938 rows: 114 good, 748 neutral, 76 bad. Random splits can be
  made safe only at very low recall: multiclass without-prefix at threshold
  0.97 accepted `7/0/0`, while threshold 0.95 still accepted `18/0/2`.
- Explicit holdouts remain the deciding evidence:
  - Train the three failure-deadline windows plus 920-999 hard negatives,
    validate on 1000-1079 hard negatives: multiclass all-features threshold
    0.95 accepted `3/14/7`, threshold 0.97 accepted `3/9/5`; not safe.
  - With `--max-bad-probability 0.01` and thresholds up to 0.995, the same
    holdout reaches zero-bad only when it accepts zero good actions
    (`0/1/0` without prefix features at threshold 0.995).
  - Train 720-759 + 840-919 + both successful hard-negative blocks, validate
    on the hard 760-839 failure window: default multiclass threshold 0.95
    accepted `3/6/1`; stricter bad-probability gating reaches zero-bad only
    with zero-good accepts.
  Conclusion: full-success low-reward mining is the right source for hard
  negatives, but the current small MLP gate and feature set still do not
  generalize into a useful online gate. Keep it as an offline diagnostic until
  a holdout can accept useful good actions with `accepted_bad=0`.
- Feature-contrast analysis over all 938 counterfactual rows shows that bad
  actions are most strongly associated with prefix conflict features, not raw
  policy confidence:
  - `forced_prefix_min_intersection_eta_gap`: bad median `1`, rest median `999`;
    effect size about `-0.96`.
  - `forced_prefix_conflict_agents`: bad mean `1.22`, rest mean `0.54`;
    effect size about `+0.87`.
  - `baseline_prefix_cell_intersections`, `baseline_prefix_conflict_agents`,
    and head-on/opposing prefix counts are also high for bad rows.
  This supports using prefix/ETA features as a safety filter, but also shows
  why the current MLP is fragile: good and bad rows can both occur in late,
  tight-slack states, so thresholding policy logits or immediate reward is not
  enough.
- A third successful low-reward range 1080-1159 was mined with
  `--selection-mode full-success-low-reward --max-reward 0.91`. It produced
  16 selected full-success seeds
  (`1111,1157,1115,1125,1081,1141,1114,1108,1083,1143,1120,1159,1134,1103,1107,1092`)
  from 80 episodes, with reward mean `0.918627` and success-rate mean
  `0.941667`. Broad movement counterfactuals on those seeds produced 255 rows:
  21 good, 216 neutral, 18 bad, with reward wins/losses/ties `23/16/216` and
  success wins/losses/ties `0/5/250`. The low-reward filter found fewer bad
  rows than 1000-1079, but includes high-value cases where immediate reward
  improves while one later arrival is lost.
- The 1080-1159 block remains hard for the learned gate:
  - Random splits on this block alone are unsafe; binary only-prefix at
    threshold 0.97 accepted `1/2/4`, and multiclass variants still accepted
    bad actions.
  - Combining all current counterfactual CSVs gives 1193 rows: 135 good,
    964 neutral, 94 bad. Random splits are still unsafe at useful recall:
    multiclass all-features threshold 0.95 accepted `31/35/10`.
  - Train on the previous 938 rows and validate on 1080-1159: default
    multiclass threshold 0.95 accepted `1/6/2`; threshold 0.97 accepted
    `0/5/2`.
  Conclusion: keep mining full-success low-reward hard negatives, but the next
  model improvement should be a safety-first prefix/ETA rule or monotonic
  reranker around the learned score, not just more MLP threshold tuning.
- A grid search over simple prefix/ETA safety rules on all 1193 current
  counterfactual rows found 7092 zero-bad rules out of 48600, but the best
  useful zero-bad rule accepted only `3/0/0` good/neutral/bad. The best rule
  required no increase in conflict agents, head-on ETA gap at least 20,
  at most one head-on edge conflict, at most one opposing-direction
  intersection, and `future_head_on_risk_delta <= -1`. This is safe but too
  low-recall for an online override.
- Prefix/ETA-only rules are not safe. When `future_head_on_risk_delta` is
  unconstrained, the best grid result on all 1193 rows accepted `14/28/6`.
  Requiring only non-increasing future risk (`<= 0`) still accepted `14/28/6`.
  On the successful hard-negative rows only, the best non-increasing-risk rule
  accepted `8/25/5`.
- A fixed best zero-bad rule selected on the previous 938 rows accepted
  `3/0/0` on those rows, but `0/0/0` on the 1080-1159 holdout and `0/0/0` on
  the hard 760-839 failure window. Conclusion: simple safety rules are useful
  as diagnostic filters, but not yet as a winning-action generator. The next
  practical policy step should combine a positive rescuer signal with these
  safety features, then validate with strict seed-window holdouts.
- A monotone rescue-rule grid over the same 1193 rows also failed to produce a
  useful deployment rule. It tested conditions such as baseline conflict/risk
  presence, non-increasing forced conflict agents, improved ETA gaps, reduced
  head-on/opposing prefix deltas, non-increasing future head-on risk, positive
  policy logit delta, and bounded forced-distance delta. The best zero-bad
  rules again accepted only `3/0/0` good/neutral/bad.
- The best strict rescue rule selected on the previous 938 rows accepted
  `0/0/0` on the 1080-1159 holdout. A relaxed rule that allowed no conflict
  increase and required positive logit delta accepted `3/8/1` on 1080-1159 and
  `1/2/2` on the hard 760-839 failure window. Conclusion: simple monotone
  rescuer+filter rules are still too brittle; useful positives and bad
  success-loss counterfactuals overlap under the current one-step feature set.
  Next, focus on better candidate generation and temporal/global features
  before deploying any learned or rule-based override.

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

For explicit window-holdout validation, train on one CSV and validate on another:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_counterfactual_gate.py \
  /private/tmp/ecml_counterfactual_failure_deadline_720_40_top12.csv \
  --validation-csv /private/tmp/ecml_counterfactual_failure_deadline_760_80_top20.csv \
  --include-feature-regex '^(baseline_prefix_|forced_prefix_)' \
  --objective binary \
  --thresholds 0.75 0.9 0.95 0.97 \
  --epochs 300 \
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
