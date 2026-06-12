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
step, and ETA gaps. New counterfactual CSVs also include deadline-aware prefix
features: minimum own/other/pair deadline slack at conflict points, deadline
miss conflict counts, tight-deadline conflict counts, and forced-minus-baseline
deltas for each of those signals. Older `/private/tmp` CSVs created before this
feature extension do not contain these columns; regenerate counterfactual rows
before using the deadline-aware features for training or rule selection.

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
- Deadline-aware prefix features were added to regenerated counterfactual rows.
  A smoke run on known hard-negative seeds `1157,1114,1081` produced 48 rows:
  11 good, 34 neutral, 3 bad, with reward wins/losses/ties `13/1/34` and
  success wins/losses/ties `0/3/45`. On this small sample, deadline-aware
  features were among the strongest bad-vs-rest signals:
  - `forced_prefix_min_other_deadline_slack`: bad median `182`, rest median
    `999`, effect size about `-1.59`.
  - `forced_prefix_min_pair_deadline_slack`: bad median `182`, rest median
    `999`, effect size about `-1.49`.
  - `baseline_prefix_min_other_deadline_slack`: bad median `178`, rest median
    `999`, effect size about `-1.45`.
  These are still smoke-test numbers, but they support the hypothesis that
  success-loss bad actions are not just spatial conflicts; they are conflicts
  involving agents with less deadline room. The next data step is to regenerate
  the failure-deadline and full-success hard-negative CSVs with these columns
  and re-run the gate/rule holdouts.
- The hard 760-839 failure-deadline set and the 1080-1159 full-success
  hard-negative set were regenerated with deadline-aware prefix features:
  - 760-839 regenerated 121 rows with unchanged label counts: 19 good,
    94 neutral, 8 bad; reward wins/losses/ties `21/6/94`; success
    wins/losses/ties `7/2/112`.
  - 1080-1159 regenerated 255 rows with unchanged label counts: 21 good,
    216 neutral, 18 bad; reward wins/losses/ties `23/16/216`; success
    wins/losses/ties `0/5/250`.
- On the combined regenerated 760+1080 data, deadline-aware features are strong
  bad-vs-rest signals:
  - `baseline_prefix_min_other_deadline_slack`: bad median `172.5`, rest
    median `999`, effect size about `-0.97`.
  - `forced_prefix_min_pair_deadline_slack`: bad median `145.5`, rest median
    `999`, effect size about `-0.97`.
  - `forced_prefix_min_other_deadline_slack`: bad median `164`, rest median
    `999`, effect size about `-0.97`.
  The delta features are weaker than absolute slack at conflict points, which
  suggests that deadline pressure identifies risky contexts rather than a
  simple forced-minus-baseline rescue direction.
- Explicit gate holdouts with the regenerated deadline-aware rows improved
  when training only on prefix/risk features
  (`baseline_prefix_*`, `forced_prefix_*`, and future-head-on risk columns):
  - Train 760-839, validate 1080-1159: threshold 0.97 accepted `2/9/0`;
    threshold 0.99 accepted `1/5/0`.
  - Train 1080-1159, validate 760-839: threshold 0.95 accepted `3/1/0`;
    threshold 0.97 accepted `2/1/0`.
  With `--max-bad-probability 0.01`, the same setup remains zero-bad but loses
  recall: `2/8/0` on 1080 at threshold 0.97, and `2/1/0` on 760 at threshold
  0.90.
- Random seed splits over the combined 376 regenerated rows are still unsafe:
  multiclass all-features at threshold 0.99 accepted `7/5/4`, and multiclass
  only-prefix at threshold 0.99 accepted `8/1/6`. Conclusion: deadline-aware
  prefix features are a real improvement for explicit holdouts, but not enough
  for deployment. The next step is to regenerate at least one more independent
  failure window and one more full-success low-reward window with the new
  columns before considering a learned/rule gate integration.
- The full deadline-aware regeneration now covers all six current windows:
  720-759 failure-deadline `67` rows (`20/2/45` reward wins/losses/ties,
  `4/0/63` success wins/losses/ties), 760-839 failure-deadline `121` rows
  (`21/6/94`, `7/2/112`), 840-919 failure-deadline `144` rows (`37/4/103`,
  `15/0/129`), 920-999 success-hard-negative `237` rows (`19/18/200`,
  `0/7/230`), 1000-1079 success-hard-negative `369` rows (`23/40/306`,
  `0/19/350`), and 1080-1159 success-hard-negative `255` rows (`23/16/216`,
  `0/5/250`). Combined, this gives `1193` rows: `135` good, `964` neutral,
  and `94` bad labels.
- Full-set deadline feature contrast confirms the same bad-vs-rest signal at
  larger scale: `forced_prefix_min_pair_deadline_slack` has bad median `149.5`
  versus rest median `999` with effect size about `-0.94`;
  `forced_prefix_min_other_deadline_slack` has bad median `168.5` versus `999`
  with effect size about `-0.93`; `forced_prefix_min_own_deadline_slack` has
  bad median `190` versus `999` with effect size about `-0.93`.
- Random splits over all `1193` rows remain unsafe even with deadline-aware
  features. At threshold `0.99`, multiclass all-features accepted `13/14/3`,
  multiclass without-prefix accepted `2/5/1`, and multiclass only-prefix
  accepted `9/9/3` good/neutral/bad. Binary variants also leaked bad actions
  unless they accepted no actions.
- Strict window holdouts with prefix/risk features are still not deployable:
  - Train all except 760-839, validate 760-839: threshold `0.95+` accepted
    `0/0/0`; threshold `0.90` accepted `0/1/0`.
  - Train all except 1080-1159, validate 1080-1159: thresholds
    `0.90/0.95/0.97/0.99` accepted `2/10/3`, `1/4/1`, `0/1/1`, and `0/0/0`.
  - Train failure windows 720-919, validate success-hard-negative windows
    920-1159: thresholds `0.90/0.95/0.97/0.99` accepted `37/153/40`,
    `34/120/35`, `27/95/22`, and `11/40/14`.
  - Train success-hard-negative windows 920-1159, validate failure windows
    720-919: thresholds `0.90/0.95/0.97/0.99` accepted `5/7/0`, `5/6/0`,
    `3/5/0`, and `3/5/0`.
  Adding `--max-bad-probability 0.01` did not fix the unsafe
  success-hard-negative holdouts; bad actions still leaked at useful
  thresholds.
- Conclusion from the full deadline-aware sweep: deadline pressure is a strong
  explanatory feature and should be integrated into richer observations or
  candidate ranking diagnostics, but the current one-step learned gate is still
  offline-only. It is zero-bad only on some failure-window holdouts or when it
  accepts too few/no useful actions; it does not yet generalize to
  success-hard-negative windows where a single alternative can destroy a
  successful schedule.

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

Route-conflict RL training path:
- `submission.my_observation_builder.MyRouteConflictObservationBuilder` exposes
  the 36 base features plus 16 route-conflict features (`obs_size=52`). These
  include near route occupancy, future route intersections, ETA overlap,
  head-on/crossing flags and whether another train has tighter slack.
- `submission.my_observation_builder.MyTrajectoryConflictObservationBuilder`
  is the next experimental RL observation. It keeps the 52 route-conflict
  features at the same indices and appends 12 trajectory/priority features
  (`obs_size=64`): relative priority rank, tighter-slack peers, same-state
  priority peers, effective slack, whether a side detour exists, whether the
  best side detour rejoins the forward greedy prefix, detour divergence length,
  prefix overlap, distance delta and conflict-count/head-on/opposing deltas.
  This is intended for learned PPO/BC policies, not for the current Docker
  default.
- `tools/train_behavior_clone.py --use-route-conflict-obs` now mirrors
  `tools/train_masked_ppo.py --use-route-conflict-obs`, automatically selecting
  the 52-feature builder and `obs_size=52`.
- `tools/train_behavior_clone.py --use-trajectory-conflict-obs` and
  `tools/train_masked_ppo.py --use-trajectory-conflict-obs` select the 64-feature
  builder and `obs_size=64`.
- Smoke validation on one BC episode and one PPO update showed the full path is
  executable: BC collected 1360 valid samples on seed 10 and wrote
  `/private/tmp/ecml_bc_route_conflict_smoke.pt`; PPO then wrote
  `/private/tmp/ecml_ppo_route_conflict_smoke.pt`; `tools/evaluate_sampled.py`
  evaluated that checkpoint with the route-conflict builder over seeds 30-31.
- The trajectory-conflict path also passed smoke validation: one BC episode
  collected 1360 valid samples and wrote
  `/private/tmp/ecml_bc_trajectory_conflict_smoke.pt`; one 32-step PPO update
  wrote `/private/tmp/ecml_ppo_trajectory_conflict_smoke.pt`; a two-episode
  load/eval smoke with `MyTrajectoryConflictObservationBuilder` completed over
  seeds 30-31. The smoke checkpoint is not a performance candidate.
- A 40-episode trajectory-conflict BC warm-start
  (`/private/tmp/ecml_bc_trajectory_conflict_40.pt`) collected 96950 valid
  samples from seeds 10-49 with teacher reward/success `0.922663 / 0.950000`.
  The teacher disagreed with the expanded init checkpoint only 63 times, mostly
  on `MOVE_RIGHT` and `MOVE_LEFT`, so the learned policy is still dominated by
  the old actor. Pure `submission.my_policy.MyPolicy` evaluation on validation
  seeds 100-129 regressed versus guarded rerank: baseline
  `0.936044 / 0.950000`, candidate `0.878421 / 0.933333`; reward
  wins/losses/ties `2/15/13`, success `3/4/23`. Wrapping the same checkpoint in
  `submission.rerank_policy.MyPolicy` reduced but did not remove the problem:
  seeds 100-129 gave `0.931983 / 0.944444` with reward `4/6/20`, success
  `1/1/28`; seeds 130-159 gave baseline `0.882811 / 0.950000`, candidate
  `0.858286 / 0.938889`, reward `2/8/20`, success `2/4/24`. Do not deploy this
  BC checkpoint or use it as an unconstrained PPO starting point. The next RL
  attempt should start from the stable default actor with the 64-feature
  observation, use strong teacher/KL anchoring, and require seedwise regression
  gates after each short PPO run.

Minimal BC warm-start:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_behavior_clone.py \
  --use-route-conflict-obs \
  --teacher-policy submission.rerank_policy.MyPolicy \
  --episodes 40 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --epochs 4 \
  --class-balanced-loss \
  --output-checkpoint /private/tmp/ecml_bc_route_conflict.pt
```

PPO fine-tuning on the route-conflict observation:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_masked_ppo.py \
  --use-route-conflict-obs \
  --init-checkpoint /private/tmp/ecml_bc_route_conflict.pt \
  --updates 20 \
  --episodes-per-update 4 \
  --num-agents 6 \
  --line-length 2 \
  --teacher-ce-coef 0.1 \
  --conflict-priority-penalty-coef 0.05 \
  --output-checkpoint /private/tmp/ecml_ppo_route_conflict.pt
```

Trajectory-conflict BC warm-start:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_behavior_clone.py \
  --use-trajectory-conflict-obs \
  --teacher-policy submission.rerank_policy.MyPolicy \
  --episodes 40 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2 \
  --epochs 4 \
  --class-balanced-loss \
  --output-checkpoint /private/tmp/ecml_bc_trajectory_conflict.pt
```

Trajectory-conflict PPO fine-tuning:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_masked_ppo.py \
  --use-trajectory-conflict-obs \
  --init-checkpoint /private/tmp/ecml_bc_trajectory_conflict.pt \
  --updates 20 \
  --episodes-per-update 4 \
  --num-agents 6 \
  --line-length 2 \
  --teacher-ce-coef 0.1 \
  --anchor-kl-coef 2.0 \
  --conflict-priority-penalty-coef 0.05 \
  --output-checkpoint /private/tmp/ecml_ppo_trajectory_conflict.pt
```

`tools/train_masked_ppo.py --anchor-kl-coef` adds a KL penalty against the
initial checkpoint policy. This is useful for 64-feature PPO because the stable
36-feature actor is expanded with zero weights for the new features; the KL term
lets PPO learn small feature-dependent deviations without drifting far from the
known-safe actor.
`--terminal-success-bonus`, `--terminal-failure-penalty`,
`--terminal-team-success-bonus` and `--terminal-team-failure-penalty` add
optional episode-end shaping. Defaults are zero, so older runs are unchanged.
Use these only with `--reward-clip` to keep value targets bounded.

Initial 64-feature PPO anchor experiments:
- Strict run `/private/tmp/ecml_ppo_trajectory_anchor_u3.pt`: 3 updates, 2
  complete episodes/update, `learning_rate=1e-5`, `teacher_ce_coef=0.5`,
  `anchor_kl_coef=20.0`, `conflict_priority_penalty_coef=0.02`,
  `global_slack_reward_coef=0.02`. The run stayed extremely close to the
  anchor (`anchor_kl` up to `3.45e-6`) and was exactly tied with guarded rerank
  on seeds 100-129: reward/success wins/losses/ties `0/0/30`.
- Looser run `/private/tmp/ecml_ppo_trajectory_anchor2_u3.pt`: same rollout
  setup with `learning_rate=2e-5`, `teacher_ce_coef=0.2`, `anchor_kl_coef=2.0`.
  This produced the first small positive PPO gate result. Against guarded rerank
  on seeds 100-129 it improved seed 120 to full success and had no success
  losses: baseline `0.936044 / 0.950000`, candidate `0.939039 / 0.955556`,
  reward wins/losses/ties `1/2/27`, success `1/0/29`. On seeds 130-159 it was
  exactly tied: `0/0/30` reward and success wins/losses/ties.
- Broader validation rejects that loose PPO checkpoint for deployment. Across
  seeds 100-249, guarded rerank baseline/candidate means were
  `0.920969 / 0.940000` versus `0.920777 / 0.940000`; reward
  wins/losses/ties were `3/6/141` and success `2/2/146`. The two success gains
  were seeds 120 and 215, but seeds 205 and 206 regressed by one train each.
  Treat this as useful evidence that 64-feature anchored PPO can find real
  rescues, but the current rollout objective is not yet safe enough. Do not
  scale this checkpoint; the next RL iteration needs a success-regression
  penalty or hard validation-aware selection before longer training.
- Terminal-failure shaping was added and smoke-tested. A short run
  `/private/tmp/ecml_ppo_trajectory_terminal_anchor2_u3.pt` reused the loose
  anchor setup plus `terminal_failure_penalty=0.3` and
  `terminal_team_failure_penalty=0.2`. It matched the previous checkpoint on
  seeds 100-129 (`1/2/27` reward and `1/0/29` success) but still failed the
  critical validation seeds: 205 and 206 remained one-train success
  regressions, while 215 remained a one-train success gain. Terminal shaping in
  this mild form is not enough; next try either stronger outcome shaping with
  validation after every update, or checkpoint selection that only keeps
  zero-success-regression candidates.
- `tools/validate_policy_gate.py` automates that checkpoint-selection gate. It
  reuses the seedwise compare logic and exits with status `1` when a candidate
  violates the configured limits. Defaults require zero success regressions,
  non-negative mean success delta and non-negative mean reward delta. A smoke
  run on the strict tied checkpoint passed on seeds 100-101; the known loose PPO
  checkpoint failed on critical seeds 205,206,215 with two success regressions.
- Stronger terminal shaping and a more regularized mid-anchor variant both
  reproduced the same critical failure pattern. The stronger terminal run
  `/private/tmp/ecml_ppo_trajectory_terminal_stronger_u3.pt` used
  `terminal_failure_penalty=0.8`, `terminal_team_failure_penalty=0.5` and
  `reward_clip=1.5`; the mid-anchor run
  `/private/tmp/ecml_ppo_trajectory_anchor_mid_u3.pt` used
  `learning_rate=1.5e-5`, `teacher_ce_coef=0.3`, `anchor_kl_coef=5.0`,
  `terminal_failure_penalty=0.5` and `terminal_team_failure_penalty=0.3`.
  Both failed seeds 205,206,215 with success wins/losses/ties `1/2/0`.
- `tools/analyze_policy_action_diffs.py --max-diffs-per-seed` can now inspect
  more than the first synchronized action difference. `tools/counterfactual_decision_eval.py`
  can focus directly on those action-diff CSVs via `--focus-action-diff-csv`.
  On the stronger terminal PPO checkpoint, the first diff was
  `MOVE_FORWARD -> MOVE_LEFT` on all critical seeds. One-step counterfactuals
  labelled this exact diff as bad on seed 205 (`success_delta=-0.166667`),
  neutral on seed 206, and good on seed 215 (`success_delta=+0.166667`). For
  seed 206, the second synchronized diff was `MOVE_LEFT -> STOP_MOVING`; forcing
  stop at that single step only reduced reward, not success. This means the PPO
  success regression is caused by trajectory drift or action sequences, not a
  single trivially separable one-step decision. Next RL work should use
  sequence-level acceptance/training signals instead of more one-step terminal
  reward tuning.
- `tools/evaluate_policy_diff_prefixes.py` measures that sequence effect
  directly. It runs the baseline policy, but adaptively replaces the first `k`
  current-trajectory disagreements with the candidate action. On critical seeds
  `205,206,215` for `/private/tmp/ecml_ppo_trajectory_terminal_stronger_u3.pt`:
  seed 205 regressed after the first diff
  (`1@188:a3:MOVE_FORWARD->MOVE_LEFT`), seed 215 improved after the first diff
  (`1@194:a4:MOVE_FORWARD->MOVE_LEFT`), and seed 206 only lost success after
  the third diff:
  `1@243:a0:MOVE_FORWARD->MOVE_LEFT;2@334:a4:MOVE_LEFT->STOP_MOVING;3@335:a4:MOVE_LEFT->STOP_MOVING`.
  Prefix summary over these three seeds: `k=1` had success wins/losses/ties
  `1/1/1`; `k>=3` had `1/2/0`, matching the full candidate's failure count.
  This gives us a better next training target: learn or regularize against
  harmful diff-prefixes, especially repeated stop overrides after an initially
  neutral route deviation.
- Adding seed 120 to the same prefix run separates reward improvements from
  success improvements. The candidate improved reward from `0.832620` to
  `0.964311`, but success stayed at `0.833333`; the adaptive prefix needed
  three diffs:
  `1@34:a0:MOVE_FORWARD->MOVE_RIGHT;2@178:a0:MOVE_FORWARD->MOVE_LEFT;3@249:a1:MOVE_FORWARD->MOVE_LEFT`.
  Over seeds `120,205,206,215`, prefix `k=1` had success wins/losses/ties
  `1/1/2`; `k>=3` had `1/2/1`. This reinforces that the next objective should
  prioritize zero success regressions over positive mean reward deltas.
- `submission.ppo_sequence_gated_policy.MyPolicy` is the first online wrapper
  for these prefix findings. It keeps guarded rerank as the baseline, queries a
  PPO checkpoint in parallel, rejects all non-side-detour candidate deviations
  (including the harmful repeated `STOP_MOVING` pattern), and accepts only
  `MOVE_FORWARD -> MOVE_LEFT/RIGHT` when the target is finite/free,
  distance-neutral, and the candidate prefix has no cell or edge interaction
  with the current planned prefixes. The current guard also rejects detours for
  already-late agents, and requires `MOVE_LEFT` candidates to improve the
  observation's direction-to-target signal over `MOVE_FORWARD`. `MOVE_RIGHT`
  candidates must either improve that direction signal or reduce planned
  prefix interactions. This removed the only observed `100-249` reward-only
  regression, seed 211, and the `line_length=3` OOD success regression seed
  839, while keeping the success gains on seeds 207 and 215. With
  `/private/tmp/ecml_ppo_trajectory_terminal_stronger_u3.pt` and
  `MyTrajectoryConflictObservationBuilder`:
  - Critical seeds `205,206,215`: baseline `0.834180 / 0.833333`,
    sequence-gated PPO `0.873129 / 0.888889`, reward `1/0/2`, success `1/0/2`.
  - Seeds `100-249`: baseline `0.920969 / 0.940000`, sequence-gated PPO
    `0.923129 / 0.942222`, reward `2/0/148`, success `2/0/148`.
  - Mined 96-seed failure-deadline + success-hard-negative stress set:
    exact tie at `0.878832 / 0.920139`, reward and success `0/0/96`.
  - OOD canaries on seeds `800-849`:
    `scene_1`, 6 agents, line length 2 tied exactly at
    `0.863739 / 0.840000`; `scene_5`, 8 agents, line length 2 tied exactly at
    `0.898640 / 0.917500`; `scene_5`, 6 agents, line length 3 initially
    regressed seed 839 via `MOVE_FORWARD -> MOVE_RIGHT`, but after the
    prefix-interaction guard tied exactly at `0.896815 / 0.916667`.
  This is not the Docker default yet, but it is the first PPO-derived wrapper
  that passes the local zero-success-regression gate on known critical cases,
  a 150-seed contiguous validation range, the mined stress set, and the main
  OOD canaries.

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/evaluate_policy_diff_prefixes.py \
  --candidate-checkpoint /private/tmp/ecml_ppo_trajectory_terminal_stronger_u3.pt \
  --obs-builder submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --seeds 205,206,215 \
  --prefix-lengths 1 2 3 5 10 \
  --num-agents 6 \
  --line-length 2 \
  --output-csv /private/tmp/ecml_diff_prefix_terminal_stronger_critical.csv \
  --output-json /private/tmp/ecml_diff_prefix_terminal_stronger_critical.json
```

`tools/mine_diff_prefix_dataset.py` turns the same adaptive diff-prefix replay
into sequence-level training rows. It stores the final episode outcome deltas
plus aggregated event features from `tools/analyze_policy_action_diffs.py`,
including route-prefix conflicts, ETA/deadline slack at conflict points, raw
policy logits, observation route-conflict features, action-transition counts,
and first/last/mean/min/max values across the forced prefix. The CSV is directly
readable by `tools/train_counterfactual_gate.py`.

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/mine_diff_prefix_dataset.py \
  --candidate-policy submission.rerank_policy.MyPolicy \
  --candidate-checkpoint /private/tmp/ecml_ppo_trajectory_terminal_stronger_u3.pt \
  --obs-builder submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --seeds 205,206,207,215 \
  --prefix-lengths 1 2 3 \
  --num-agents 6 \
  --line-length 2 \
  --only-changed \
  --output-csv /private/tmp/ecml_diff_prefix_sequence_train.csv \
  --output-json /private/tmp/ecml_diff_prefix_sequence_train.json
```

Smoke results:
- Seeds `207,215`, line length 2, prefix lengths `1,2`: four good rows, reward
  wins/losses/ties `2/0/0` per prefix and success wins/losses/ties `2/0/0`.
  The resulting CSV loaded successfully in `tools/train_counterfactual_gate.py`
  with roughly `700` numeric feature columns.
- Seed `839`, line length 3: prefix `1` was bad
  (`reward_delta=-0.019179`, `success_delta=-0.166667`) while prefix `2` was
  neutral. This is a useful warning: the same first deviation can be harmful
  alone but harmless when followed by the candidate's next correction, so the
  next learned model should use sequence labels rather than only one-step
  counterfactual labels.

The feature surface is intentionally rich and easy to overfit. For first
sequence-gate experiments, use explicit seed-window holdouts and feature regex
filters, for example:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/train_counterfactual_gate.py \
  /private/tmp/ecml_diff_prefix_sequence_train.csv \
  --include-feature-regex '^(event_(baseline_prefix_|candidate_prefix_|candidate_.*_delta|obs_route_|obs_time_slack|slack_|candidate_raw_|baseline_raw_|transition_|count|unique))' \
  --objective multiclass \
  --max-bad-probability 0.01 \
  --thresholds 0.9 0.95 0.97 0.99 \
  --epochs 500 \
  --hidden-size 32
```

Current assessment: this is the right next direction because it converts the
PPO wrapper's hand-coded acceptance logic into labelled temporal data. It is
not yet evidence that an online learned gate is safe; require zero accepted bad
rows on explicit held-out seed windows before replacing deterministic guards.

First explicit sequence-gate holdouts:
- Critical training set `120,205,206,207,211,215`, line length 2, prefix lengths
  `1,2,3,5`: 24 rows with labels `12 good`, `11 bad`, `1 neutral`. This is a
  deliberately dense sanity set, not an unbiased benchmark.
- OOD holdout `839-843`, line length 3, prefix lengths `1,2,3`: 15 rows with
  labels `7 bad`, `8 neutral`. All-features multiclass gate overfit the small
  training set and accepted holdout good/neutral/bad `0/3/1` at all thresholds
  from `0.75` to `0.99`. The tighter sequence/prefix feature regex was safer:
  at thresholds `0.75/0.90` it accepted `0/0/1`, and at `0.95+` it accepted
  `0/0/0`.
- Same-distribution neutral holdout `216-225`, line length 2, prefix lengths
  `1,2,3`: 9 rows, all neutral. All-features gate still accepted neutral rows
  at every threshold (`0/1/0` at `0.99`); the tighter sequence/prefix feature
  gate accepted `0/5/0` up to `0.97` and `0/0/0` at `0.99`.

Assessment after these holdouts: sequence labels are the right diagnostic
direction, but the learned gate is not ready. The all-feature model memorizes
and leaks bad actions; the safer feature set reaches zero bad only by becoming
mostly inert. Next useful step is not online deployment; it is mining larger,
balanced sequence windows with both rescue positives and hard negative
success-preserving/reward-losing prefixes, then validating with strict
seed-window holdouts.

Targeted sequence mining is much more useful than broad blind mining:
- Broad window `100-129`, line length 2, prefix lengths `1,2,3,5`: 80 rows
  from 20 changed seeds, labels `4 good`, `6 bad`, `70 neutral`. The only good
  seed was `120`; bad labels came from `108` and `111`. This took several
  minutes and is too neutral-heavy for rapid iteration.
- A relaxed `validate_policy_gate.py` scan over `130-249` found only 9
  outcome-changing seeds out of 120: `174,182,184,205,206,207,211,215,232`.
  The raw candidate had reward wins/losses/ties `4/5/111` and success
  wins/losses/ties `3/2/115`.
- Mining only those 9 changed seeds produced 36 dense sequence rows:
  `12 good`, `23 bad`, `1 neutral`. Per prefix, there were consistently three
  good rows and five or six bad rows. This is a better data source than broad
  windows because it concentrates on the actual candidate decision boundary.

Gate checks from these targeted rows:
- Train on broad `100-129`, validate on targeted `130-249`: all-features
  multiclass gate accepted `4/0/0` good/neutral/bad at every tested threshold
  (`0.75..0.99`). This is the first sign that the seed-120 rescue pattern can
  transfer to later rescue seeds without accepting bad targeted rows. The
  tighter sequence/prefix feature regex accepted `4/0/1`, so it was less safe
  here.
- The same all-features train-`100-129` model accepted `0/0/0` on the
  line-length-3 holdout, so it was safe but did not recover any OOD positives.
  On the same-distribution neutral holdout it accepted `0/2/0` at thresholds
  up to `0.97` and `0/1/0` at `0.99`.
- Training on broad `100-129` plus targeted `130-249` made the line-length-3
  holdout fully inert (`0/0/0`) and still accepted `0/2/0` neutral rows on the
  neutral holdout.

Assessment after targeted mining: the sequence-gate idea is no longer clearly
hopeless; targeted changed-seed mining gave useful transfer on targeted
in-distribution positives. It is still not a deployment candidate because it
has no demonstrated OOD positive recall and still accepts neutral actions. The
right next loop is: scan for outcome-changing seeds, mine only those as dense
sequence rows, hold out whole seed windows and scene/line-length variants, then
deploy only if accepted bad is zero and accepted good remains nonzero across
multiple disjoint holdouts.

Next disjoint scan `250-369` is a cautionary result:
- Relaxed scan over 120 seeds found 10 outcome-changing seeds:
  `251,273,274,289,294,311,335,361,363,365`. The raw PPO candidate was worse
  on average: reward wins/losses/ties `2/7/111`, success wins/losses/ties
  `1/3/116`, with success regressions on `251,274,335` and one strong
  success win on `294`.
- Targeted prefix mining of those 10 seeds produced 40 dense rows:
  `8 good`, `31 bad`, `1 neutral`. Per prefix, there were two good rows and
  seven or eight bad rows.
- Train on `100-129 + targeted 130-249`, validate on targeted `250-369`:
  both all-features and sequence/prefix feature gates accepted `0/0/0` on the
  validation rows at all tested thresholds. Safety is fine, but useful recall
  is zero.
- Train on `100-129 + targeted 250-369`, validate on targeted `130-249`:
  all-features gate also accepted `0/0/0` on validation. Adding the
  negative-heavy `250-369` block makes the learned gate inert on an earlier
  targeted positive block.
- Train on all current sequence rows (`100-129`, targeted `130-249`,
  targeted `250-369`) and validate on OOD/neutral canaries: line-length-3
  holdout accepted `0/0/0` at threshold `0.90+`; neutral holdout accepted
  `0/0/0` at every threshold.

Assessment after `250-369`: targeted mining is the right data workflow, but the
current PPO candidate is not a strong enough source of diverse positive
rescues. More gate training mostly learns to reject everything once the
negative-heavy window is included. Next best effort should shift toward better
candidate generation and RL objective design: produce multiple PPO/BC
candidates, mine changed-seed prefix labels per candidate, and only then train
an ensemble/reranker. Continuing to tune one learned gate on this single PPO
checkpoint is low expected value.

Candidate-diversity PPO test:
- Trained `/private/tmp/ecml_ppo_trajectory_mix293_u4.pt` as a small
  64-feature PPO candidate around seed `293`, with strong teacher/anchor
  regularization (`learning_rate=1e-5`, `teacher_ce_coef=0.4`,
  `anchor_kl_coef=8.0`), trajectory-conflict observation, conflict-priority
  shaping, and terminal success/failure shaping. This was intentionally a
  short candidate-generation test, not a long training run.
- On `100-129`, it matched the previous useful seed-120 rescue pattern:
  baseline/candidate `0.936044 / 0.950000` -> `0.939066 / 0.955556`, reward
  wins/losses/ties `1/1/28`, success wins/losses/ties `1/0/29`; changed seeds
  were `111` and `120`.
- On `130-249`, it was mixed and still unsafe: reward wins/losses/ties
  `2/3/115`, success wins/losses/ties `1/2/117`; changed seeds
  `174,205,206,211,215`. The same hard success regressions `205` and `206`
  remained, and it did not recover the old candidate's additional positives
  `184` and `207`.
- On `250-369`, it was worse than useful: reward wins/losses/ties `0/5/115`,
  success wins/losses/ties `0/1/119`; changed seeds
  `311,335,361,363,365`.
- Assessment: simply changing PPO seed mix with stronger regularization did
  not create useful candidate diversity. It kept some known positives but lost
  others and still produced hard regressions. Do not mine prefix labels for
  this checkpoint as a main path. The next candidate-generation experiment
  should be meaningfully different: e.g. a BC/RL ensemble candidate, a
  supervised sequence-rescue head trained from targeted prefix rows, or PPO
  with an explicit offline penalty/regularizer against mined bad prefixes.

Sequence value/risk ensemble on targeted prefix rows:
- `tools/evaluate_counterfactual_value_ensemble.py` now supports
  `--validation-csv`, so value/risk models can be tested on explicit
  seed-window or OOD holdouts instead of only random seed splits.
- It also supports `--output-audit-csv` for explicit validation runs. The audit
  writes per validation row: seed, prefix length, event string, true deltas,
  utility, value mean/std/lower-confidence-bound, bad probability mean/std/upper
  confidence bound, and accept/reject for each threshold combination.
- The tool now has an optional auxiliary Success-rescue classifier head. It is
  enabled for acceptance only when `--min-success-probability` is provided.
  The default path without that flag keeps the old value/risk acceptance logic.
  Success acceptance uses a lower confidence bound on predicted
  `success_delta > 0`, still gated by the bad-probability upper confidence
  bound. The audit CSV includes success probability mean/std/LCB.
- Simple all-feature value/risk training on broad `100-129` and validation on
  targeted `130-249` leaked bad rows. With sequence/prefix features only, the
  same train/validation split accepted `16/0/0` good/neutral/bad aggregated
  over three ensemble split seeds at `min_utility=0,max_bad_probability<=0.2`;
  at `min_utility=0.02` it accepted `4/0/0`.
- Train on `100-129 + targeted 130-249`, validate on targeted `250-369`:
  the default value/risk ensemble leaked bad rows (`3/0/1` at
  `min_utility=0.05`). A stricter configuration with sequence/prefix features,
  higher bad weights, stronger bad-risk uncertainty, and stronger value lower
  confidence bounds accepted `2/0/0` at
  `min_utility=0.2,max_bad_probability=0.005..0.02` and `1/0/0` at
  `min_utility=0.2,max_bad_probability=0.001`.
- Train on all current sequence rows and validate on line-length-3 OOD
  holdout: the strict value/risk model accepted `0/0/0` across all tested
  thresholds, so it did not leak the known line-length-3 bad prefix. On the
  same-distribution neutral holdout, the same strict setup still accepted
  neutral rows at lower utility thresholds, but `min_utility=0.4` accepted
  `0/0/0`.
- Assessment: this is the first supervised sequence-rescue result that is
  better than the simple gate classifier. It can find some held-out good
  targeted prefixes with zero bad leakage under strict settings. It still has
  low recall, no demonstrated OOD positive recall, and some neutral acceptance
  unless the utility threshold is high. Keep it offline, but this is now a
  plausible basis for a future online reranker once we add row-level audit
  output and validate on more candidate/window combinations.
- Row-level audit of the strict `100-249 -> targeted 250-369` run showed the
  accepted rows were not two distinct rescues. They were the same concrete
  prefix accepted by different bootstrap split seeds: seed `273`, prefix `1`,
  event `1@166:a4:MOVE_FORWARD->MOVE_RIGHT`, with true
  `reward_delta=+0.0574713` and unchanged success. At
  `min_utility=0.2,max_bad_probability=0.001`, only split seed `3` accepted it;
  at bad-probability thresholds `0.005..0.02`, split seeds `3` and `5`
  accepted it. This is plausible and safe, but still very low diversity.
- Next disjoint PPO-candidate scan on seeds `370-489` with the same
  checkpoint produced a difficult holdout: gate summary baseline/candidate
  `0.917194/0.933333 -> 0.913898/0.936111`, reward wins/losses/ties
  `2/8/110`, success wins/losses/ties `3/1/116`, with a clear success
  regression on seed `402`. The changed seeds were
  `383,390,391,392,394,398,402,406,419,452,484`.
- Targeted prefix mining over those changed seeds yielded 44 rows:
  `16 good / 7 neutral / 21 bad`. By prefix length: prefix `1` had
  `2/5/4` good/neutral/bad, prefix `2` had `4/2/5`, prefix `3` had
  `5/0/6`, and prefix `5` had `5/0/6`.
- Train on all current sequence rows through targeted `250-369`, validate on
  targeted `370-489`: the previous strict sequence/prefix value/risk setting
  accepted `0/0/0` at `min_utility=0.2,max_bad_probability=0.001..0.02`.
  A diagnostic frontier showed the model is safe but too pessimistic: at
  `min_utility=-0.25,max_bad_probability=0.02` it accepted `4/0/0`, all four
  rows being the same reward rescue on seed `419` across prefix lengths
  `1,2,3,5` (`1@61:a5:MOVE_FORWARD->MOVE_RIGHT`,
  `reward_delta=+0.15873`, unchanged success). At `min_utility=-0.5` it also
  accepted neutral rows from seed `398`; no tested frontier leaked bad rows on
  this holdout.
- Assessment after `370-489`: the learned value/risk route is directionally
  useful and still safer than deploying raw PPO or a hard heuristic gate, but
  it remains narrow. It recognizes a second reward-rescue pattern with no bad
  leakage under a softer LCB threshold, while success-rescue rows are still
  underestimated. The best next step is not to mainline this gate yet; mine
  more diverse PPO candidates and train the sequence value model on richer
  conflict/priority features or separate reward-vs-success heads.
- Success-rescue diagnostic on the same `370-489` holdout: increasing
  `success_weight` to `4.0` did not help; it accepted only `0/1/0` at the soft
  `min_utility=-0.25` threshold and nothing at non-negative utility thresholds.
  Reducing value uncertainty from `value_std_coef=2.0` to `1.0` accepted
  `4/2/0` at `min_utility=0,max_bad_probability<=0.1`, still only the reward
  rescue pattern plus neutral rows. Using all 716 numeric features increased
  apparent recall to `8/0/0` at `min_utility=0,max_bad_probability=0.05`, but
  the accepted rows were again only seed `419` across bootstrap splits and
  prefix lengths; no accepted row had positive `success_delta`.
- The rejected success-positive validation rows are from seeds `390`, `394`,
  and `398`. The strict sequence/prefix model gives them negative value LCBs
  around `-0.65..-1.21` and very high bad-probability UCBs around
  `1.0..1.9`. The current training set through `250-369` has only 16
  success-positive rows from four seeds (`184`, `207`, `215`, `294`), while
  the `370-489` holdout has eight success-positive rows from three new seeds.
  This is too little diversity for the bad-risk head to distinguish risky
  stop/side-action sequences from real success rescues. Treat Success rescue
  as a separate data/model problem rather than a utility-weight tuning issue.
- Auxiliary Success-head test: with sequence/prefix features, training through
  targeted `250-369` and validating on `370-489`, conservative Success-LCB
  acceptance still found `0/0/0`. Removing the Success uncertainty penalty
  found seed `398` prefixes `3/5` as success-positive good rows only when
  `max_bad_probability=1.0`; the same rows had bad UCB about `0.90`, so they
  are not deployable. The head also overpredicted the neutral prefix `1` of
  the same seed. Using all features accepted many reward-positive rows but
  still `accepted_success_positive=0`, so this was not real Success-rescue
  recall.
- Earlier holdout check for the auxiliary Success head, trained on
  `100-129 + targeted 130-249` and validated on targeted `250-369`, was also
  not deployable. At low `min_success_probability=0.05`, it leaked bad rows
  (`accepted_bad=2` in aggregate) and had negative accepted success delta under
  conservative bad thresholds. At thresholds `0.1..0.6`, it accepted only
  reward-positive rows with `accepted_success_positive=0`. Conclusion: the
  Success head confirms the diagnosis but does not solve it yet; the next
  useful work is more diverse Success-positive mining and/or a better bad-risk
  model around stop/side-action sequences.
- Next disjoint scan on seeds `490-609` with
  `ecml_ppo_trajectory_terminal_stronger_u3.pt`: baseline/candidate
  `0.918937/0.947222 -> 0.917667/0.945833`, reward wins/losses/ties
  `5/6/109`, success wins/losses/ties `1/2/117`. Changed seeds were
  `499,500,519,529,535,537,578,588,589,603,605`; seed `589` was the only
  success-positive case (`reward_delta=+0.150893`,
  `success_delta=+0.166667`), while seeds `519` and `578` were
  success-negative.
- Targeted prefix mining for `490-609` produced 44 rows:
  `19 good / 4 neutral / 21 bad`. Prefix `1` and `2` each had
  `5/2/4` good/neutral/bad and one success-positive row with no success
  losses; prefix `3` had `5/0/6` with one success win and one success loss;
  prefix `5` had `4/0/7` with one success win and two success losses. All four
  success-positive rows are the same seed `589` event
  `1@198:a0:MOVE_FORWARD->MOVE_LEFT`; success-negative rows came from seed
  `519` prefixes `3/5` and seed `578` prefix `5`.
- Train through targeted `370-489`, validate on targeted `490-609`: the
  normal value/risk branch still only found reward-positive rows. At
  `min_utility=-0.25,max_bad_probability=0.001` it accepted `4/0/0`, but
  `accepted_success_positive=0`; relaxing bad-risk thresholds leaked bad rows.
  The auxiliary Success branch did not generalize to seed `589`. With
  sequence/prefix features, its score for seed `589` was near zero
  (`success_probability_mean` around `1e-5..0.002`) and bad UCB was high
  (`1.08..1.89`). Increasing the negative weight for non-success rows did not
  fix this. This is a useful hard positive, but still not enough to deploy a
  learned Success gate.
- Assessment after `490-609`: continuing to mine the same PPO candidate gives
  useful hard labels but not enough Success-positive diversity. Best next
  step is to evaluate/mine a different checkpoint family for distinct
  success-positive action patterns, then retrain the Success/Risk audit with
  the combined windows.
- Alternative checkpoint check on the same `490-609` range:
  `ecml_ppo_trajectory_anchor2_u3.pt` produced no new Success-positive seed.
  Gate summary was baseline/candidate `0.918937/0.947222 ->
  0.917403/0.945833`, reward wins/losses/ties `2/4/114`, success
  wins/losses/ties `1/2/117`. Changed seeds were
  `519,578,588,589,603,605`; the only success-positive seed was again `589`,
  and success regressions were again `519` and `578`.
- Targeted Anchor2 mining over those six changed seeds produced 24 rows:
  `8 good / 3 neutral / 13 bad`. The Success-positive rows were the same
  event as the stronger-terminal candidate:
  `1@198:a0:MOVE_FORWARD->MOVE_LEFT` on seed `589` across prefixes
  `1,2,3,5`. The Success-negative rows came from seed `519` prefixes `3/5`
  and seed `578` prefixes `2/3/5`. Conclusion: Anchor2 is not useful
  candidate diversity for the current Success-rescue dataset; the next
  candidate should come from a meaningfully different training objective or
  selection criterion, not only a lighter terminal-shaping variant.
- New success-diversity candidate
  `/private/tmp/ecml_ppo_trajectory_successdiv_seed610_u4.pt`: a short
  64-feature trajectory PPO run with seed `610`, `updates=4`,
  `episodes_per_update=4`, `learning_rate=2e-5`, stronger terminal
  success/failure shaping (`terminal_success_bonus=0.9`,
  `terminal_failure_penalty=1.1`, team success/failure `0.5/0.8`), plus
  teacher/anchor regularization (`teacher_ce=0.35`, `anchor_kl=4.0`) and
  conflict/slack shaping. Rollout success was mixed
  (`0.958, 0.833, 0.583, 0.958`), so this is not a deployment candidate by
  itself; it is a candidate generator for diverse prefix labels.
- On seeds `100-129`, the success-diversity candidate produced
  baseline/candidate `0.936044/0.950000 -> 0.937439/0.961111`, reward
  wins/losses/ties `1/2/27`, success wins/losses/ties `2/0/28`. Changed seeds
  were `111,119,120`; seed `119` is a new success-positive pattern with
  `reward_delta=-0.0488281, success_delta=+0.166667`, and seed `120` is the
  known success-positive rescue.
- Targeted mining on those `100-129` changed seeds produced 12 rows:
  `8 good / 4 bad`. Success-positive rows were seed `119`
  `1@70:a1:MOVE_RIGHT->MOVE_FORWARD` plus later
  `MOVE_LEFT->MOVE_FORWARD` events, and seed `120`
  `1@180:a0:MOVE_FORWARD->MOVE_LEFT`; both stayed positive from prefix `1`.
- On seeds `130-249`, the same candidate was mixed but useful as data:
  baseline/candidate `0.917200/0.937500 -> 0.917720/0.938889`, reward
  wins/losses/ties `3/4/113`, success wins/losses/ties `2/1/117`. Changed
  seeds were `174,184,205,207,211,218,232`; success positives were `184` and
  `207`, and the known success regression was `205`.
- Targeted mining on those `130-249` changed seeds produced 28 rows:
  `8 good / 20 bad`. Success-positive rows were seed `184`
  `1@96:a4:MOVE_FORWARD->MOVE_RIGHT` and seed `207`
  `1@86:a3:MOVE_FORWARD->MOVE_RIGHT`, both positive from prefix `1`.
  Success-negative rows were seed `205`
  `1@188:a3:MOVE_FORWARD->MOVE_LEFT`, also negative from prefix `1`.
- First positive auxiliary Success-head holdout: train on broad `100-129`,
  targeted `250-369`, targeted `370-489`, targeted `490-609`, and the new
  success-diversity targeted `100-129`; validate on success-diversity targeted
  `130-249` while excluding the old targeted `130-249` rows to avoid same-seed
  leakage. With sequence/prefix features, strict bad UCB
  `max_bad_probability=0.001`, and Success LCB threshold
  `min_success_probability=0.02`, the model accepted `12/0/0`
  good/neutral/bad aggregated over three split seeds, all 12 rows
  success-positive. The accepted rows were only seed `207` across prefixes and
  split seeds; seed `184` was still missed. At threshold `0.3`, only split
  seed `5` accepted seed `207`.
- Canary after adding the success-diversity rows: validating the same
  Success-head setup on the older terminal-targeted `490-609` holdout did not
  open bad leaks under strict bad UCB (`max_bad_probability=0.001`), but it
  still accepted only reward-positive rows and `accepted_success_positive=0`.
  So the Success head is finally learning one held-out success-rescue pattern,
  but the recall is still narrow and candidate/motif-specific.
- Assessment after success-diversity candidate: changing the PPO objective did
  produce valuable new Success-positive patterns (`119`, `184`, `207`) and a
  first clean Success-head transfer on seed `207`. This validates the direction:
  invest in candidate generation and mined sequence labels. It is still not an
  online gate; next best step is train one or two more short, differently
  seeded success-diversity candidates and require each to add at least one new
  success-positive seed or a new hard negative before spending time on longer
  RL runs.
- Second success-diversity candidate
  `/private/tmp/ecml_ppo_trajectory_successdiv_seed730_u4.pt`: same short
  64-feature PPO pattern with seed `730`, slightly higher rollout temperature
  (`1.05`), stronger team success bonus, and slightly lighter teacher/anchor
  (`teacher_ce=0.30`, `anchor_kl=3.0`). Rollout success was more stable than
  seed610 (`0.875, 0.875, 0.917, 0.958`).
- On seeds `100-129`, seed730 produced
  baseline/candidate `0.936044/0.950000 -> 0.939996/0.961111`, reward
  wins/losses/ties `1/1/28`, success wins/losses/ties `2/0/28`. Changed seeds
  were only `119` and `120`, both success-positive. Targeted mining over those
  two seeds produced 8 good rows and no bad/neutral rows; the events matched
  the useful seed610 positives (`119`: `MOVE_RIGHT->MOVE_FORWARD` followed by
  `MOVE_LEFT->MOVE_FORWARD`; `120`: `MOVE_FORWARD->MOVE_LEFT` then
  `MOVE_LEFT->MOVE_FORWARD`).
- On seeds `130-249`, seed730 did not add useful diversity:
  baseline/candidate `0.917200/0.937500 -> 0.917328/0.936111`, reward
  wins/losses/ties `1/1/118`, success wins/losses/ties `0/1/119`; changed
  seeds were `205` and `211`, with the known success regression on `205` and
  no success-positive rows. Do not mine this block further.
- Adding the clean seed730 `100-129` positives to the Success-head training set
  did not broaden holdout seed coverage beyond seed `207`, but it made the
  signal more robust: on the same success-diversity `130-249` validation, the
  model still accepted `12/0/0` good/neutral/bad under strict bad UCB
  `0.001`, and the acceptance remained stable up to
  `min_success_probability=0.4` instead of dropping after `0.05..0.3`. This is
  useful as data reinforcement for the `MOVE_*->MOVE_FORWARD` rescue motif,
  but it is not a new online-policy improvement.
- High-temperature success-diversity candidate
  `/private/tmp/ecml_ppo_trajectory_successdiv_hitemp_seed840_u4.pt`: a more
  exploratory short PPO run with rollout temperature `1.25`, lower teacher/kl
  anchoring (`teacher_ce=0.18`, `anchor_kl=1.5`), stronger team success
  shaping and the same 64-feature trajectory observation. Rollout success was
  reasonable (`0.875, 0.917, 0.958, 0.917`), but the gate results show that
  more exploration did not add positive diversity.
- On seeds `100-129`, seed840 produced
  baseline/candidate `0.936044/0.950000 -> 0.931859/0.955556`, reward
  wins/losses/ties `0/2/28`, success wins/losses/ties `1/0/29`. Changed seeds
  were `111` and `119`; the only success-positive seed was the already-known
  `119`, plus the known reward-negative `111`. Do not mine this window further
  as new positive diversity.
- On seeds `130-249`, seed840 produced
  baseline/candidate `0.917200/0.937500 -> 0.918542/0.937500`, reward
  wins/losses/ties `1/2/117`, success wins/losses/ties `1/1/118`. Changed
  seeds were `198,207,211`: known positive `207`, neutral/reward-negative
  `211`, and a new success regression on seed `198`.
- Targeted mining of seed840 `130-249` produced 12 rows:
  `4 good / 1 neutral / 7 bad`. The positive rows were seed `207`
  `1@39:a3:MOVE_RIGHT->MOVE_FORWARD`, a useful variant of the
  `MOVE_*->MOVE_FORWARD` rescue motif. The new hard negative was seed `198`:
  prefix `1` was neutral, but prefixes `2/3/5` became success-negative after
  `1@285:a2:MOVE_LEFT->MOVE_FORWARD;2@288:a0:MOVE_FORWARD->MOVE_LEFT`.
- Adding the seed840 mined rows to the Success-head training set did not hurt
  the existing success-diversity `130-249` validation. The strict bad-UCB run
  still accepted `12/0/0` and remained stable up to
  `min_success_probability=0.5`. Seed840 is therefore useful as
  hard-negative/robustness data, but not as a source of new success-positive
  coverage.
- `tools/train_masked_ppo.py` now supports `--training-seeds`, a comma-separated
  complete-episode seed list. This lets PPO train directly on curated rescue
  positives and hard negatives instead of contiguous seed blocks, while keeping
  the old seed behavior unchanged when the option is omitted. Smoke validation
  with `--training-seeds 119,120,198,205,207` passed and wrote
  `/private/tmp/ecml_ppo_training_seeds_smoke.pt`.
- Targeted-seed PPO candidate
  `/private/tmp/ecml_ppo_trajectory_targeted_seed901_u4.pt`: trained for
  4 updates x 4 complete episodes on positives `119,120,184,207,589` plus hard
  negatives `111,198,205,211,232,519,578`, with trajectory-conflict
  observation, stronger teacher/anchor regularization (`teacher_ce=0.35`,
  `anchor_kl=4.0`), rollout temperature `1.05`, conflict-priority penalty
  `0.025`, slack-progress reward `0.025`, global slack reward `0.03`, terminal
  success/failure shaping, and reward clip `1.5`.
- Holdout `250-369` versus guarded rerank:
  baseline/candidate `0.913467/0.938889 -> 0.916290/0.937500`, reward
  wins/losses/ties `3/3/114`, success wins/losses/ties `0/1/119`.
  Changed seeds were `258,264,280,311,335,343`; seed `335` was a success
  regression (`success_delta=-0.166667`). This candidate is not safe for
  deployment.
- Holdout `370-489` versus guarded rerank:
  baseline/candidate `0.917194/0.933333 -> 0.918245/0.933333`, reward
  wins/losses/ties `2/1/117`, success wins/losses/ties `0/0/120`.
  Changed seeds were `392,452,477`, all reward-only. This is acceptable as a
  diagnostic result but still not enough to justify deploying the raw PPO
  checkpoint.
- Assessment after targeted-seed PPO: explicit seed control is useful
  infrastructure, but this PPO candidate did not create new held-out
  success-positive diversity. The immediate value is hard-negative curation
  (`335`) and a compact fast decision suite (`258,264,280,311,335,343,392,452,477`)
  for candidate screening. The next winner-oriented step should shift from
  more small raw-PPO deployment attempts to offline sequence Success/Risk
  learning and candidate generation evaluated first on this fast suite, then
  on full disjoint holdouts.
- Fast-suite screening of the same candidate with strict success-loss gate
  (`max_success_losses=0`, seeds `258,264,280,311,335,343,392,452,477`)
  failed quickly: baseline/candidate `0.765301/0.851852 ->
  0.816960/0.833333`, reward wins/losses/ties `5/4/0`, success
  wins/losses/ties `0/1/8`, with the same seed `335` regression. Use this
  compact suite as the first filter for future PPO/BC candidate generators;
  only candidates with zero success loss there should be promoted to full
  120-seed holdout scans.
- Targeted prefix mining on the fast-suite changed seeds produced 36 sequence
  rows: `20 good / 2 neutral / 14 bad`. The bad rows were seed `264`
  prefixes `2/3/5` (`MOVE_RIGHT->MOVE_FORWARD` twice), seed `311` all
  prefixes (`MOVE_FORWARD->MOVE_LEFT`), seed `335` prefixes `2/3/5`
  (`MOVE_FORWARD->MOVE_LEFT` twice, success-negative), and seed `477` all
  prefixes (`MOVE_RIGHT->MOVE_FORWARD`). The good rows are reward-only
  positives on seeds `258,280,343,392,452`. This confirms the core sequence
  issue: seed `335` is neutral at prefix `1` but success-negative from prefix
  `2`, so a one-step gate cannot reliably protect it. Add these rows to the
  next offline Success/Risk training pool as hard-negative and reward-positive
  screening data, not as deployment proof for the raw PPO checkpoint.
- Fast-suite holdout for the existing sequence Success/Risk ensemble, trained
  on prior prefix rows but not on the new fast-suite rows, is not deployable:
  with sequence/prefix features and strict thresholds
  `min_utility=0.2,max_bad_probability=0.001,min_success_probability=0.02`,
  it accepted `36/0/12` good/neutral/bad validation rows. The accepted bad rows
  were seed `477` across prefixes `1/2/3/5`; softer threshold combinations
  also accepted seed `264`. The good news is that the success-negative seed
  `335` prefixes `2/3/5` were rejected with high bad UCB (`>1.0`) and negative
  value LCB. The model has learned some success-risk structure, but the reward
  bad-risk head is still too optimistic on reward-only detours, so the next
  training pass should include the fast-suite rows and validate on disjoint
  success-diversity windows.
- Training the sequence Success/Risk ensemble with the fast-suite rows added
  and validating on the disjoint success-diversity `130-249` rows made the
  standard utility gate too conservative at strict thresholds: at
  `min_utility=0.2,max_bad_probability=0.001,min_success_probability=0.02` it
  accepted `0/0/0`. Relaxing to `min_utility=-0.25` recovered Success-positive
  rows but leaked bad rows, so the mixed utility-or-success acceptance is still
  not deployable with this data.
- A better diagnostic was a Success-only acceptance branch:
  `min_utility=999`, `success_std_coef=0`, and a finer bad-UCB threshold.
  With `max_bad_probability=0.0004..0.0005` and
  `min_success_probability=0.02..0.05`, validation on success-diversity
  `130-249` accepted `8/0/0` good/neutral/bad, all success-positive rows from
  seed `207` across prefixes `1/2/3/5` and split seeds `3/5`. Tighter bad UCB
  `0.0002..0.0003` accepted `4/0/0`; looser `0.0007` leaked four bad rows.
  The same Success-only configuration on the hitemp `130-249` holdout accepted
  `0/0/0`, so it is safe there but has no extra recall.
- Assessment after fast-suite training: adding hard negatives improved risk
  calibration enough to create a narrow zero-bad Success-only frontier, but the
  frontier is threshold-fragile and still recognizes only seed `207`. The next
  useful step is to treat Success rescue as a separate online gate candidate:
  train/persist a Success-only sequence model, require extremely low bad UCB,
  and validate it across the fast suite, success-diversity holdouts, and
  full disjoint seed windows before any deployment.
- `tools/evaluate_counterfactual_value_ensemble.py` now supports
  `--output-model`. It trains a final ensemble on the training CSV rows and
  saves model weights, feature columns, normalization tensors, split seeds,
  ensemble size, config, and train summary. A Success-only export smoke with
  the fast-suite training pool wrote
  `/private/tmp/ecml_success_only_sequence_fasttrain.pt`: 15 models
  (`split_seeds=1,3,5`, `ensemble_size=5`), 464 features, 264 training rows
  with categories `83 good / 84 neutral / 97 bad`. This is reproducible
  offline model state for the next online gate prototype, not an activated
  submission policy.
- `tools/score_counterfactual_value_ensemble.py` loads an exported ensemble and
  scores prefix CSV rows with explicit thresholds. Scoring the exported
  Success-only model showed that the validation-audit threshold `0.0004` is too
  strict for the final 15-model export (`0/0/0` on successdiv `130-249`), while
  `max_bad_probability=0.0025`, `min_utility=999`,
  `min_success_probability=0.02`, `success_std_coef=0` accepts `4/0/0` on
  successdiv `130-249`: seed `207`, prefixes `1/2/3/5`, all
  success-positive. The same exported threshold accepts `0/0/0` on the hitemp
  `130-249` holdout and `4/0/0` on the fast-suite training rows, only seed
  `280` reward-positive rows. This is the first persisted Success-only
  sequence candidate with no observed bad acceptance in these offline checks,
  but it still has very low recall and needs online prefix-feature integration
  plus full seed-window validation.
- `submission.sequence_success_policy.MyPolicy` is the first online prototype
  for that persisted Success-only sequence candidate. It keeps guarded rerank
  as the baseline, queries the success-diversity PPO checkpoint in parallel,
  builds cumulative diff-prefix features through the existing mining feature
  pipeline, and accepts a candidate deviation only when the exported sequence
  ensemble passes the Success-only thresholds. It is not the Docker/default
  policy; it depends on the local exported model and PPO checkpoint paths, and
  the online feature path is currently slow.
- Online validation of this prototype with trajectory-conflict observations:
  seed `207` improved from `0.783854/0.833333` to `0.991030/1.000000`
  (`reward_delta=+0.207176`, `success_delta=+0.166667`). The fast-suite seeds
  `258,264,280,311,335,343,392,452,477` stayed exactly neutral
  (`0/0/9` reward and success changes). Full disjoint windows were also safe
  but neutral: `250-369` stayed `0/0/120`, and `370-489` stayed `0/0/120`.
  On `130-249`, the only changed seed was `207`, giving aggregate
  baseline/candidate `0.917200/0.937500 -> 0.918927/0.938889`,
  reward and success wins/losses/ties `1/0/119`.
- Assessment after online integration: this is a real learned online gate now,
  not just a CSV audit, and it safely recovers one known Success rescue.
  However, recall is still extremely low and runtime is high because the
  wrapper rebuilds the broad prefix feature row and scores a 15-model ensemble
  online. The next useful work is either to distill this Success-only gate into
  a smaller feature/model path, or to mine/train more Success-positive motifs
  before spending effort on deployment optimization.
- Candidate-source recall check: with the default success-diversity PPO
  candidate, the online gate also opens seed `119` on `100-129`
  (`reward_delta=-0.0488281`, `success_delta=+0.166667`) and seed `207` on
  `130-249` (`reward_delta=+0.207176`, `success_delta=+0.166667`). With the
  terminal-stronger PPO candidate
  (`ECML_SEQUENCE_SUCCESS_CANDIDATE_CHECKPOINT=/private/tmp/ecml_ppo_trajectory_terminal_stronger_u3.pt`),
  seed `589` improves from `0.849107/0.833333` to `1.000000/1.000000`, and
  seed `529` is reward-positive without success change.
- The terminal-stronger candidate initially leaked seed `578`
  (`reward_delta=-0.063615`, `success_delta=-0.166667`) because the online gate
  accepted a late `MOVE_FORWARD->MOVE_LEFT` at time `198`, while the old mined
  prefix rows for that seed were only harmful after a longer prefix. The online
  wrapper now has two conservative safety guards: `ECML_SEQUENCE_MAX_ACCEPTED_EVENTS`
  defaults to `1`, and learned `MOVE_LEFT` deviations are rejected when true
  slack exceeds `ECML_SEQUENCE_LEFT_MAX_SLACK` (default `115`, matching the
  existing rerank left-detour slack guard). This blocks seed `578` while keeping
  seed `589` (`slack=89`) and seed `529`.
- Full guarded terminal-candidate validation on `490-609` after that guard:
  baseline/candidate `0.918937/0.947222 -> 0.920793/0.948611`, reward
  wins/losses/ties `2/0/118`, success wins/losses/ties `1/0/119`. Changed
  seeds were `529` (`reward_delta=+0.0718391`, success unchanged) and `589`
  (`reward_delta=+0.150893`, `success_delta=+0.166667`). This is the best
  online Success-only result so far, but still candidate-source-specific; the
  next useful prototype is a multi-candidate sequence gate that can combine the
  default success-diversity source (`119/207`) with the terminal-stronger source
  (`529/589`) under the same safety guards.
- `submission.sequence_success_policy.MyPolicy` now supports
  `ECML_SEQUENCE_SUCCESS_CANDIDATE_CHECKPOINTS`, a comma-separated list of PPO
  candidate checkpoints. It checks candidates in order and accepts the first
  action that passes the same exported Sequence Success gate and safety guards;
  without the env var it keeps the previous single-candidate default. With
  success-diversity seed610 first and terminal-stronger second, targeted known
  seeds produced wins on `119`, `207`, `529`, and `589`, while `578` stayed
  neutral after the left-slack guard.
- Multi-candidate validation so far:
  - Fast-suite `258,264,280,311,335,343,392,452,477`: exactly neutral
    (`0/0/9` reward and success changes).
  - `100-129`: baseline/candidate `0.936044/0.950000 ->
    0.934417/0.955556`, reward wins/losses/ties `0/1/29`, success
    wins/losses/ties `1/0/29`; only seed `119` changed
    (`reward_delta=-0.0488281`, `success_delta=+0.166667`).
  - `130-249`: baseline/candidate `0.917200/0.937500 ->
    0.918927/0.938889`, reward and success wins/losses/ties `1/0/119`;
    only seed `207` changed.
  - `490-609`: same as the guarded terminal-only result,
    `0.918937/0.947222 -> 0.920793/0.948611`, reward wins/losses/ties
    `2/0/118`, success wins/losses/ties `1/0/119`; changed seeds `529` and
    `589`.
- Assessment after multi-candidate sequence gate: this is the best current
  learned online direction by benefit/effort. It combines multiple PPO
  generators while preserving the conservative Sequence Success filter, giving
  three success-positive seeds (`119`, `207`, `589`) and one reward-positive
  seed (`529`) in checked windows with no observed success loss. It is still
  not ready as a default submission because recall is low, seed `119` trades
  reward for success, and the online feature/scoring path is slow.
- Candidate-source audit after the multi-candidate prototype:
  - Offline scoring with the exported Success-only sequence model at
    `min_utility=999,max_bad_probability=0.0025,min_success_probability=0.02`
    showed that several mined candidate sources contain apparently safe rows.
    The useful accepted motifs were: `targeted_seed901` seed `280`
    (`reward_delta=+0.133120`, success unchanged), successdiv seed610/730
    seeds `119/120`, and Anchor2 seed `589`. Hitemp seed840 added no accepted
    rows under these strict thresholds.
  - Online validation is stricter than the offline row audit. Before the
    first-diff guard, adding `targeted_seed901` as a third candidate source
    recovered seed `280`, but also introduced a pure reward regression on seed
    `264`
    (`reward_delta=-0.116713`, success unchanged). The same seeds with only
    the existing two candidates (`successdiv_seed610`,
    `terminal_stronger`) stayed neutral.
  - Adding successdiv seed730 or Anchor2 as extra candidate sources did not
    add new online wins beyond the existing two-candidate list. Seed `120`
    remains blocked online by the current guards even though it appears in
    offline prefix rows.
  - The existing two-candidate list was then validated on the previously
    missing full windows `250-369` and `370-489`; both were exactly neutral and
    passed the zero-loss gate (`reward 0/0/120`, success `0/0/120` in each
    window). At this point, the best candidate list remained
    `successdiv_seed610 + terminal_stronger`.
- Assessment after this audit: candidate recall should not be expanded by
  simply appending PPO checkpoints. The online gate can accept actions whose
  local prefix row looks safe but whose rollout effect is reward-negative, so
  every new candidate source needs a fast-suite screen plus full disjoint
  windows before promotion. The next winner-oriented step is to improve the
  learned risk model or candidate generator around these newly identified
  motifs, especially distinguishing seed `280`-like reward rescues from
  seed `264`-like reward leaks.
- `submission.sequence_success_policy.MyPolicy` now has two debug/safety
  additions for this exact failure mode. `ECML_SEQUENCE_TRACE_PATH` writes
  accepted Sequence-Gate decisions as JSONL, and
  `ECML_SEQUENCE_TRACE_ALL=1` also writes rejected scored decisions. The trace
  showed seed `264` first rejected `1@18:a1:MOVE_RIGHT->MOVE_FORWARD`, then
  later accepted `1@122:a5:MOVE_RIGHT->MOVE_FORWARD` as a singleton event with
  low bad UCB. That singleton is out-of-distribution relative to the mined
  `first k candidate diffs` prefix rows, where seed `264` becomes bad once
  the first and second diffs are combined.
- The policy therefore defaults `ECML_SEQUENCE_FIRST_DIFF_ONLY=1`: each
  candidate checkpoint may contribute at most its first episode-level
  baseline-vs-candidate diff to the learned sequence gate. This is per
  candidate source, so an early rejected diff from one PPO checkpoint does not
  block a later first diff from another checkpoint. With only
  `targeted_seed901`, this blocks seed `264` while keeping seed `280`
  (`reward_delta=+0.133120`, success unchanged).
- After the first-diff guard, the three-candidate list
  `successdiv_seed610 + terminal_stronger + targeted_seed901` is the best
  current experimental online Sequence-Gate configuration:
  - Known hard/fast suite `119,120,207,280,529,578,589,258,264,311,335,343,392,452,477`:
    reward wins/losses/ties `4/1/10`, success `3/0/12`; changed seeds were
    `119` (success win with reward tradeoff), `207`, `280`, `529`, and `589`.
    The previous `264` leak stayed neutral.
  - `100-129`: changed only seed `119`
    (`reward_delta=-0.0488281`, `success_delta=+0.166667`). This fails the
    strict non-negative mean-reward gate but has no success loss.
  - `130-249`: changed only seed `207`
    (`reward_delta=+0.207176`, `success_delta=+0.166667`), gate pass.
  - `250-369`: changed only seed `280`
    (`reward_delta=+0.133120`, success unchanged), gate pass.
  - `370-489`: exactly neutral, gate pass.
  - `490-609`: changed seeds `529` (`reward_delta=+0.0718391`) and `589`
    (`reward_delta=+0.150893`, `success_delta=+0.166667`), gate pass.
  - Aggregate over `100-609`: baseline/candidate
    `0.917837/0.939869 -> 0.918845/0.940850`, reward wins/losses/ties
    `4/1/505`, success wins/losses/ties `3/0/507`.
- Assessment after the first-diff guard: this is a meaningful recall gain with
  one known reward tradeoff but no observed success loss over the checked
  windows. The guard is not seed-specific; it aligns online acceptance with the
  prefix-mining distribution. The remaining open question is whether the
  `119` success-for-reward tradeoff should be accepted in competition scoring
  or controlled by a stricter reward-aware Success gate.
- Reward-aware tradeoff default: the Flatland book describes normalized return
  as the primary evaluation metric, and the starterkit reports
  `normalized_reward` alongside `success_rate`. Therefore the experimental
  Sequence-Gate policy now defaults `ECML_SEQUENCE_REJECT_TRANSITIONS` to
  `MOVE_RIGHT->MOVE_FORWARD`, because this removes the only observed
  reward-loss seed (`119`) while preserving the checked reward-positive
  changes. Set `ECML_SEQUENCE_REJECT_TRANSITIONS=` to recover the unblocked
  extra-Success variant.
- With the same three-candidate list and
  default `MOVE_RIGHT->MOVE_FORWARD` rejection, the known seed `119` tradeoff
  is blocked while the useful `207`, `280`, `529`, and `589` changes remain:
  - Known tradeoff seeds `119,207,280,529,589,264,578`: reward
    wins/losses/ties `4/0/3`, success `2/0/5`; changed seeds were `207`,
    `280`, `529`, and `589`.
  - `100-129`: exactly neutral, gate pass.
  - `130-249`: changed only seed `207`, gate pass.
  - `250-369`: changed only seed `280`, gate pass.
  - `370-489`: exactly neutral, gate pass.
  - `490-609`: changed seeds `529` and `589`, gate pass.
  - Aggregate over `100-609`: baseline/candidate
    `0.917837/0.939869 -> 0.918941/0.940523`, reward wins/losses/ties
    `4/0/506`, success wins/losses/ties `2/0/508`.
- Assessment of the reward-aware default: this is now the better default for
  `submission.sequence_success_policy.MyPolicy` because it improves the metric
  we can verify locally (`normalized_reward`) and removes the only checked
  reward regression. If a later official benchmark or leaderboard evidence
  values completion more heavily than normalized reward, the unblocked mode
  remains available and keeps one extra success-positive seed (`119`) at a
  small reward cost.
- Deployment packaging step: the exported Success-only sequence model and the
  three PPO candidate checkpoints are now stored under `submission/models/`.
  `submission.sequence_success_policy.MyPolicy` defaults to these repository
  paths and the three-candidate reward-oriented configuration; no `/private/tmp`
  model paths or env vars are required for the default behavior. The Dockerfile
  now uses `submission.sequence_success_policy.MyPolicy`,
  `MyTrajectoryConflictObservationBuilder`, and copies `tools/` into the image
  because the online gate reuses the local prefix-feature extraction code.
- Repo-default smoke without explicit model/candidate env vars on known seeds
  `119,207,280,529,589,264,578`: reward wins/losses/ties `4/0/3`, success
  `2/0/5`, gate pass. Seed `119` stayed neutral, while `207`, `280`, `529`,
  and `589` remained active.
- Independent holdout scan `610-659` found one Sequence-Gate regression before
  the head-on-prefix guard: rerank `0.932773 / 0.940000`, sequence
  `0.932017 / 0.936667`, reward `0/1/49`, success `0/1/49`; seed `631`
  regressed by `-0.037798` reward and `-0.166667` success. Trace showed one
  accepted `MOVE_FORWARD -> MOVE_RIGHT` event with
  `candidate_prefix_head_on_edge_conflicts=13`, while known positive accepted
  events had `0,0,0,4`. `submission.sequence_success_policy.MyPolicy` now
  rejects candidates above `ECML_SEQUENCE_MAX_HEAD_ON_EDGE_CONFLICTS=8` by
  default (`-1` disables the guard). With the guard, `610-659` became exactly
  neutral against rerank: reward `0/0/50`, success `0/0/50`. Known gains plus
  seed `631` (`119,207,280,529,589,264,578,631`) still pass with reward
  `4/0/4` and success `2/0/6`.
- Independent holdout scan `660-709` after the head-on guard passed but exposed
  a reward-only regression: rerank `0.928673 / 0.936667`, sequence
  `0.932205 / 0.936667`, reward `1/1/48`, success `0/0/50`. Seed `660`
  regressed by `-0.066348` reward, while seed `671` gained `+0.242954`.
  Trace showed seed `660` accepted a `MOVE_FORWARD -> MOVE_RIGHT` event with
  `candidate_prefix_same_edge_conflicts=18` and `value_lcb=-0.849489`; seed
  `671` remained a different high-confidence `MOVE_FORWARD -> MOVE_LEFT` gain.
  `SequenceSuccessPolicy` now also rejects same-edge-heavy candidates when
  `candidate_prefix_same_edge_conflicts >=
  ECML_SEQUENCE_SAME_EDGE_VALUE_GUARD_MIN_CONFLICTS=8` and
  `value_lcb < ECML_SEQUENCE_SAME_EDGE_VALUE_GUARD_MIN_VALUE_LCB=-0.25`
  (`min_conflicts=-1` disables this guard). With both guards, `660-709`
  improves to reward `1/0/49`, success `0/0/50`; known gains plus
  `631,660,671` pass with reward `5/0/5`, success `2/0/8`.
- Broader independent holdout scan `710-809` with both guards was exactly
  neutral against rerank: `0.911634 / 0.921667` for both policies, reward
  `0/0/100`, success `0/0/100`, and zero changed rows. This is good
  regression evidence for the current guards, but it also means the online
  Sequence-Gate did not find new gains in this block.
- Broader holdout `810-909` exposed the next same-edge regression under the
  older `-0.5` same-edge value threshold: seed `905` regressed by `-0.044209`
  reward and `-0.166667` success from one accepted `MOVE_FORWARD -> MOVE_LEFT`
  event with `candidate_prefix_same_edge_conflicts=30` and
  `value_lcb=-0.314352`. Tightening the default same-edge value guard to
  `-0.25` neutralized seed `905`; known gains plus `631,660,671,905` still
  pass with reward `5/0/4`, success `2/0/7`. With the tighter default,
  `810-909` becomes exactly neutral against rerank: reward `0/0/100`, success
  `0/0/100`.
- Broader holdout `910-1009` with the tighter default was also exactly neutral
  against rerank: both policies scored `0.944226 / 0.945000`, reward
  `0/0/100`, success `0/0/100`, and zero changed rows. Together with `710-809`
  and the stabilized `810-909`, this supports the current view that the
  Sequence-Gate default is now conservative and regression-resistant, but also
  too sparse to create broad gains by itself.
- Runtime note: a one-seed local policy-gate comparison took roughly `9.4s`
  for Sequence-Gate candidate versus `7.4s` for rerank candidate, including
  process startup and baseline episode cost. This is acceptable for the next
  Docker smoke/build check but still needs full container/runtime validation
  before treating the Sequence-Gate policy as final submission default.
- Docker is not available in the current local shell, so the build could not
  be executed here. A Docker-import simulator that copied only `submission/`
  and `tools/` to `/private/tmp` successfully instantiated
  `SequenceSuccessPolicy` with three candidate policies and the default
  rejected transition `(MOVE_RIGHT, MOVE_FORWARD)`. Setting `MPLCONFIGDIR`,
  `XDG_CACHE_HOME`, and `PYTHONPYCACHEPREFIX` to writable `/tmp` locations
  reduced first-import smoke time from about `29.5s` with Matplotlib/font-cache
  warnings to about `2.4s`; the Dockerfile now sets these env vars.
- `tools/policy_scoreboard.py` is the current steering tool for comparing
  policies. Unlike the older sampled benchmark helper, it supports a distinct
  observation builder per candidate and reports deltas, wins/losses/ties, and
  CSV/JSON output against a named baseline candidate. Use it before promoting
  any future PPO/gate change.
- Starterkit-vs-current scoreboard on seeds `100-129`, 6 agents, line length 2:
  starterkit PPO with `MyObservationBuilder` scored `0.907488 / 0.933333`;
  both guarded rerank and current reward-oriented Sequence-Gate with
  `MyTrajectoryConflictObservationBuilder` scored `0.936044 / 0.950000`.
  Versus starterkit, reward wins/losses/ties were `13/2/15` and success
  wins/losses/ties were `4/1/25`.
- Sequence-vs-rerank scoreboard smoke on known gain seeds `207,280,589`:
  rerank `0.788907 / 0.888889`, sequence `0.952637 / 1.000000`, reward
  wins/losses/ties `3/0/0`, success `2/0/1`.
- `tools/select_target_seeds.py` turns scoreboard or `validate_policy_gate.py`
  JSON rows into prioritized seed lists for the next mining/training loop.
  Use `failures` first to find incomplete episodes and `full-success-low-reward`
  second to find completed but inefficient episodes. For scoreboard JSONs with
  multiple policies, pass `--candidate sequence` or the policy name being
  mined. On the current
  reward-oriented Sequence-Gate scan over seeds `100-609`, the top failure
  targets were `452,153,239,556,403,184,345,466,504,343,137,294,321,215,328,
  198,394,475,468,291`; the top full-success low-reward targets were
  `281,322,109,136,573,605,164,241,310,146,274,335,478,361,441,297,244,158,
  497,127`. Mine the failure list first because it has the clearest upside for
  normalized reward and completion.

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl XDG_CACHE_HOME=/private/tmp/ecml_cache \
  .venv/bin/python tools/policy_scoreboard.py \
  --candidate starterkit=submission.my_policy.MyPolicy,obs=submission.my_observation_builder.MyObservationBuilder \
  --candidate rerank=submission.rerank_policy.MyPolicy,obs=submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --candidate sequence=submission.sequence_success_policy.MyPolicy,obs=submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --baseline starterkit \
  --blocks 100-129 \
  --quiet \
  --output-json /private/tmp/ecml_scoreboard_100_129.json \
  --output-csv /private/tmp/ecml_scoreboard_100_129.csv \
  --summary-csv /private/tmp/ecml_scoreboard_100_129_summary.csv
```

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache \
  .venv/bin/python tools/select_target_seeds.py \
  /private/tmp/ecml_sequence_success_reject_r2f_100_129.json \
  /private/tmp/ecml_sequence_success_reject_r2f_130_249.json \
  /private/tmp/ecml_sequence_success_reject_r2f_250_369.json \
  /private/tmp/ecml_sequence_success_reject_r2f_370_489.json \
  /private/tmp/ecml_sequence_success_reject_r2f_490_609.json \
  --mode failures \
  --success-threshold 0.834 \
  --top-k 20 \
  --output-csv /private/tmp/ecml_targets_current_failures_100_609_top20.csv \
  --output-json /private/tmp/ecml_targets_current_failures_100_609_top20.json
```

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache \
  .venv/bin/python tools/select_target_seeds.py \
  /private/tmp/ecml_sequence_success_reject_r2f_100_129.json \
  /private/tmp/ecml_sequence_success_reject_r2f_130_249.json \
  /private/tmp/ecml_sequence_success_reject_r2f_250_369.json \
  /private/tmp/ecml_sequence_success_reject_r2f_370_489.json \
  /private/tmp/ecml_sequence_success_reject_r2f_490_609.json \
  --mode full-success-low-reward \
  --max-reward 0.85 \
  --top-k 20 \
  --output-csv /private/tmp/ecml_targets_current_full_success_low_reward_100_609_top20.csv \
  --output-json /private/tmp/ecml_targets_current_full_success_low_reward_100_609_top20.json
```

Start the next counterfactual mining pass on the hardest current failure seeds:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl XDG_CACHE_HOME=/private/tmp/ecml_cache \
  .venv/bin/python tools/counterfactual_decision_eval.py \
  --policy submission.sequence_success_policy.MyPolicy \
  --obs-builder submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --seeds 452,153,239,556,403 \
  --num-agents 6 \
  --line-length 2 \
  --forced-actions LEFT,FORWARD,RIGHT \
  --max-decisions-per-seed 6 \
  --max-alternatives-per-decision 2 \
  --output-csv /private/tmp/ecml_counterfactual_current_failures_top5.csv \
  --output-json /private/tmp/ecml_counterfactual_current_failures_top5.json
```

The initial top-5 smoke on seeds `452,153,239,556,403` produced 60 rows with
reward wins/losses/ties `10/4/46` and success wins/losses/ties `8/1/51`. This
is a useful signal density for the next RL/gate dataset, especially because the
same small batch contains positive rescues, harmful side effects, and neutral
examples.

Then run the focused top-20 pass using the selector JSON. The counterfactual
tool accepts both detailed `mine_failure_seeds.py` records and compact
`select_target_seeds.py` rows with `failed_agent_ids`, so this samples only
agents that actually failed in the current Sequence-Gate rollout:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl XDG_CACHE_HOME=/private/tmp/ecml_cache \
  .venv/bin/python tools/counterfactual_decision_eval.py \
  --policy submission.sequence_success_policy.MyPolicy \
  --obs-builder submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --focus-failures-json /private/tmp/ecml_targets_current_failures_100_609_top20.json \
  --num-agents 6 \
  --line-length 2 \
  --forced-actions LEFT,FORWARD,RIGHT \
  --max-decisions-per-seed 6 \
  --max-alternatives-per-decision 2 \
  --output-csv /private/tmp/ecml_counterfactual_current_failures_top20_focus.csv \
  --output-json /private/tmp/ecml_counterfactual_current_failures_top20_focus.json
```

Focused top-5 smoke on the same seeds produced 54 rows with reward
wins/losses/ties `11/4/39` and success wins/losses/ties `9/0/45`. The
success-rescue signal is cleaner than the unfocused smoke, but failed-agent
focus is not a full replacement: on seed `153`, the unfocused run found
positive helper-agent decisions for agent `4`, while the failed-only focus
correctly sampled only baseline-failed agents `0,1,5`. Keep both data sources
until the learned model shows which generalizes better.

The full focused top-20 run produced 195 rows with reward wins/losses/ties
`39/14/142` and success wins/losses/ties `20/2/173`; 17 of 20 seeds had at
least one reward- or success-positive counterfactual. Outcome categories at
`reward_epsilon=1e-6` were good/neutral/bad `41/142/12`. Feature contrasts show
good versus bad rows differ most in route-prefix conflict-agent counts, policy
logit deltas, deadline slack, crossing intersections, and divergence/rejoin
features. A seed-split MLP gate is still not deployable at ordinary thresholds:
at `0.90`, binary all-feature validation accepted good/neutral/bad `24/21/9`.
At a very conservative `0.995`, all-feature binary validation reached zero bad
while accepting only `4/3/0`; multiclass all-feature reached `4/6/0`. Treat
this as a promising conservative learned-gate prototype, not as final online
evidence.
- Hard-case remine around the top-20 good/bad counterfactual windows with
  `--focus-counterfactual-window-before 8 --focus-counterfactual-window-after 8`
  produced 64 dense rows. Reward wins/losses/ties were `39/14/11`, success
  wins/losses/ties were `20/2/42`, and tool outcome categories were
  good/neutral/bad `41/11/12` because success-positive rows are treated as good
  even when reward decreases. Seed-split gate validation improved as a
  prototype but is still small: binary `without_prefix` at threshold `0.99`
  accepted `4/0/0`, and multiclass `without_prefix` at `0.995` accepted
  `4/0/0`. This is the best current learned-gate signal, but it is mined from
  the same top-20 failure pool and needs an independent target-seed holdout
  before online integration.
- Independent failure-focus mining from the guarded `660-709` holdout selected
  seeds `667,663,706,700,696,701,688,661,662,682` and produced 86 rows:
  reward wins/losses/ties `12/10/64`, success wins/losses/ties `10/1/75`,
  outcome categories good/neutral/bad `17/64/5`. Combining this with the
  earlier 64-row hard-case set breaks the apparent learned-gate safety: mixed
  seed-split validation over 150 rows still leaks bad rows at all tested
  thresholds. Explicit train-on-old, validate-on-`660-709` is worse: binary MLP
  training is near-perfect on the old set (`40/0/0` accepted at threshold
  `0.90`) but validates at only `7/42/2` good/neutral/bad. Conclusion: the
  current MLP gate is still overfitting local mined cases. Keep using these
  datasets for feature learning and diagnostics, but do not deploy the learned
  counterfactual gate online yet.
- Independent failure-focus mining from the neutral `710-809` holdout selected
  seeds `723,742,763,754,744,715,716,803,722,725,758,761` and produced 108
  rows: reward wins/losses/ties `23/7/78`, success wins/losses/ties `13/0/95`,
  outcome categories good/neutral/bad `26/78/4`. The labels include useful
  late rescue actions, reward-positive stop-to-move variants, and reward-vs-
  success tradeoffs. Combined split validation over the earlier hard-case set,
  `660-709`, and `710-809` still leaks bad rows at tested thresholds. Explicit
  train-on-old+`660-709`, validate-on-`710-809` is safer only at extreme
  thresholds: binary at `0.99` accepts `3/3/0`, multiclass at `0.995` accepts
  `3/1/0`. This is useful diagnostic signal, but still too little accepted
  good action volume for an online learned gate.
- Independent failure-focus mining from the stabilized `810-909` holdout
  selected seeds `896,901,877,867,864,885,855,836,815,829,879,812` and
  produced 87 rows: reward wins/losses/ties `17/3/67`, success
  wins/losses/ties `9/2/76`, outcome categories good/neutral/bad `15/67/5`.
  The block contains useful rescue rows, pure reward wins, hard negatives, and
  reward-positive/success-negative traps. Combined split validation across all
  four current mined datasets now has 345 rows but still leaks bad rows at all
  tested thresholds. Explicit train-on-old+`660-709`+`710-809`, validate-on
  `810-909` also leaks: binary validation at `0.99` accepts `5/4/3`, multiclass
  at `0.995` accepts `4/4/2`. Conclusion remains unchanged: learned-gate
  datasets are useful for diagnosis and future model design, but not yet for
  online deployment.
- Independent failure-focus mining from the neutral `910-1009` holdout selected
  seeds `924,995,1006,970,915,923,938,979,976,946,1005,972` and produced 100
  rows: reward wins/losses/ties `9/6/85`, success wins/losses/ties `7/0/93`,
  outcome categories good/neutral/bad `11/85/4`. Adding this fifth mined
  dataset brings the combined diagnostic pool to 445 rows, but the simple MLP
  gate degrades further: mixed split validation still leaks bad rows at all
  tested thresholds, and explicit train-on-previous, validate-on-`910-1009` is
  only clean at near-zero utility. Binary at `0.995` accepts `0/1/0`;
  multiclass at `0.99` accepts only `2/0/0`. This argues for a different
  learned objective or architecture, not for deploying the current gate.
- The value/risk ensemble was also tested on the same newest split: train on
  the previous four mined datasets, validate on `910-1009`. The Success-rescue
  variant still leaked bad rows (`3/5/3` good/neutral/bad at
  `min_utility=0.1,max_bad_probability=0.02,min_success_probability=0.9`).
  A stricter no-Success variant with higher bad weights and stronger
  uncertainty was safe only when useful accepts collapsed to zero
  (`0/0/0` at `min_utility>=0.2,max_bad_probability<=0.02`). Semantic-feature
  and no-prefix ablations did not fix this. The recurring leak was seed `946`,
  where a reward-negative forced move (`reward_delta=-0.115741`, unchanged
  success) was assigned positive utility LCB and near-zero bad probability.
  Conclusion: the current one-step supervised utility model is not reliable
  enough for online gating.
- A new conservative 64-feature PPO candidate
  `/private/tmp/ecml_ppo_trajectory_failurefocus_910_u4.pt` was trained on
  known positives plus the newest failure targets with stronger teacher/anchor
  regularization (`teacher_ce=0.45`, `anchor_kl=6.0`, `lr=1e-5`). Rollout
  success stayed stable enough (`0.958, 0.875, 0.792, 0.833`), but raw
  candidate validation on the fast suite failed because seed `335` regressed:
  reward `6/4/8`, success `4/1/13`. Adding this checkpoint as an extra
  Sequence-Gate candidate was safe but did not add coverage: default Sequence
  and default-plus-new both scored reward `4/0/14`, success `2/0/16` on the
  same 18-seed fast suite, accepting only the existing `207,280,529,589`
  motifs. Do not promote this checkpoint; use it only as another hard-negative
  diagnostic if needed.
- Next independent holdout `1010-1109` with the current default Sequence-Gate
  passed and was slightly positive: baseline `0.916344 / 0.941667`, candidate
  `0.916942 / 0.941667`, reward `1/0/99`, success `0/0/100`. The single online
  reward win was seed `1051` (`+0.059829`, unchanged success); there were no
  reward or success regressions. Failure targeting selected
  `1049,1095,1064,1079,1040,1071,1031,1012,1011,1018,1105,1017`.
- Focused counterfactual mining on those `1010-1109` failure seeds produced a
  clean positive dataset: 67 rows with reward wins/losses/ties `10/0/57`,
  success wins/losses/ties `6/0/61`, and no forced-action failures. Useful new
  rescue seeds include `1095` (`+0.245` reward, `+0.166667` success), `1079`
  (`+0.104918`, `+0.166667`), `1040` (`+0.078987`, `+0.166667`), and `1017`
  (`+0.151515`, `+0.166667`); seed `1064` contributed reward-positive
  unchanged-success rows. This is a better next training signal than the last
  PPO checkpoint: it shows new rescue actions exist, but the current candidate
  generators do not propose or pass them online yet.
- `tools/train_rescue_behavior_clone.py` now trains a checkpoint-compatible
  64-feature `ActorCritic` candidate directly from positive counterfactual
  rescue rows. It replays the baseline policy on each labelled seed, collects
  the exact real observation at each labelled `(seed, env_time, agent)` event,
  trains the `forced_action` with high weight, and mixes in low-weight baseline
  anchor samples so the candidate stays close to the default policy.
- First Rescue-BC candidate `/private/tmp/ecml_rescue_bc_1010_v1.pt` used all
  current mined counterfactual CSVs as positive labels. Collection reproduced
  all labels exactly: `120/120` rescue hits, zero misses, zero invalid actions,
  zero baseline mismatches, plus `45477` anchor samples. Raw validation on the
  extended fast suite found new useful behaviour but was not safe:
  reward `13/3/7`, success `7/1/15`, with a success regression on seed `529`.
  Through the current Sequence-Gate it became safe but added no accepted
  coverage: default Sequence and default-plus-Rescue-BC both scored reward
  `4/0/19`, success `2/0/21`, accepting only `207,280,529,589`.
- Trace on the new Rescue-BC-only positive seeds `1017,1079,1095` showed the
  blocker is the learned Sequence scorer rather than the action generator:
  the Rescue-BC checkpoint proposes plausible `MOVE_FORWARD -> MOVE_RIGHT`
  diffs, but the current sequence model assigns very high bad-risk UCB
  (`~1.3-1.6`) despite zero prefix conflicts and low or positive success LCB.
  Next step: retrain or redesign the Sequence Success/Risk scorer with these
  new positive motifs and matching hard negatives; do not relax the bad-risk
  threshold by hand.
- Retrained the Success/Risk sequence ensemble with the Rescue-BC diff-prefix
  rows added to the old sequence dataset. Offline scoring of the exported
  model accepted `14/3/0` good/neutral/bad Rescue-BC prefix rows at the online
  bad-risk threshold, accepted `4/0/0` success-diversity holdout rows, and
  accepted no hitemp holdout rows. Online validation showed why the old
  `FIRST_DIFF_ONLY=1` default was too restrictive for this candidate family:
  Rescue-BC needs later accepted diffs to realize new rescues.
- With `FIRST_DIFF_ONLY=0`, the first rescue-scored default leaked reward on
  seed `660`; raising `ECML_SEQUENCE_MIN_SUCCESS_PROBABILITY` to `0.5` blocked
  that but leaked seed `946`. The accepted `946` event had
  `success_lcb=0.556`, while the genuine `1079` rescue had
  `success_lcb=0.956`. A stricter `0.6` threshold fixed those and gave fast
  suite reward/success `5/0/18` and `3/0/20`, but a second new holdout
  `1160-1209` found a small reward-only loss on seed `1160`
  (`-0.002146`). That loss accepted at `success_lcb=0.666`; the useful
  holdout win on seed `1185` accepted at `0.791`.
- The promoted conservative Rescue-Sequence default therefore uses the new
  model `submission/models/ecml_success_only_sequence_with_rescue.pt`, adds
  `submission/models/ecml_rescue_bc_1010_v1.pt` as the fourth candidate,
  defaults `ECML_SEQUENCE_FIRST_DIFF_ONLY=0`, and defaults
  `ECML_SEQUENCE_MIN_SUCCESS_PROBABILITY=0.75`. Validation versus guarded
  rerank:
  - Extended fast suite
    `119,120,184,207,280,529,589,258,264,311,335,343,392,452,477,660,905,946,1017,1040,1064,1079,1095`:
    reward `3/0/20`, success `3/0/20`, mean reward delta `+0.020130`.
    Changed seeds were `207`, `589`, and `1079`; all three were Success wins.
  - New holdout `1110-1209`: reward `1/0/99`, success `0/0/100`, mean reward
    delta `+0.000943`. The only changed seed was `1185` with reward
    `+0.094268`, unchanged Success.
  - Current default Sequence on `1110-1159` was fully neutral (`0/0/50`), while
    the p06 ablation found one additional reward win on `1112`. We did not
    promote p06 because `1160-1209` exposed the small seed `1160` reward leak.
  This is a net RL/gate improvement with lower action coverage than p06; do
  not lower the Success threshold without adding the new hard negatives.
- Broader post-promotion holdout showed the remaining weak point of Rescue-BC
  right detours. Blocks `1210-1309` and `1310-1409` were fully neutral
  (`0/0/100` reward and success in each block). Block `1410-1509` before the
  extra value guard had reward `2/2/96`, success `0/1/99`; wins were `1438`
  (`+0.144698`) and `1509` (`+0.105219`), but seed `1442` lost Success
  (`-0.029656` reward, `-0.166667` Success) and seed `1463` lost reward
  (`-0.075620`). Both bad accepts were `MOVE_FORWARD -> MOVE_RIGHT` from the
  Rescue-BC checkpoint with very low bad UCB and high Success LCB; a pure
  Success-threshold increase would not catch seed `1463`.
- `SequenceSuccessPolicy` now adds `ECML_SEQUENCE_RIGHT_DETOUR_MIN_VALUE_LCB`
  with default `-0.4`, applied only to `MOVE_FORWARD -> MOVE_RIGHT` accepts.
  This blocks low-value right-detour accepts without touching `MOVE_LEFT` or
  `MOVE_LEFT -> MOVE_FORWARD` rescues. Targeted validation on
  `207,589,1079,1185,1438,1442,1463,1509` kept all six known wins and
  neutralized `1442/1463` (`6/0/2` reward, `3/0/5` Success). Re-running
  `1410-1509` after the guard produced reward `2/0/98`, success `0/0/100`,
  with only `1438` and `1509` changed.
- Next holdouts found one more Success regression pattern. `1510-1609` was
  fully neutral. Before the added left-to-forward confidence guard, `1610-1709`
  had seed `1671` lose reward `-0.039872` and Success `-0.166667`; the accepted
  event was a Rescue-BC `MOVE_LEFT -> MOVE_FORWARD` with high Success LCB but
  weak raw candidate margin (`0.202`). The known good `1079` rescue uses the
  same transition but has much stronger margin (`1.428`). The policy now adds
  `ECML_SEQUENCE_LEFT_TO_FORWARD_MIN_MARGIN=0.5` for this transition only.
  Targeted validation kept known wins `207,589,1079,1185,1438,1509` and
  neutralized `1671` (`6/0/1` reward, `3/0/4` Success). Re-running
  `1610-1709` after the guard was fully neutral; `1710-1809` was positive
  with reward `1/0/99`, success `0/0/100`, changing only seed `1740`
  (`+0.093214`, unchanged Success).
- Holdout `1810-1909` found a reward-only regression on seed `1814`
  (`-0.075758`, unchanged Success). The accepted event was again a Rescue-BC
  `MOVE_FORWARD -> MOVE_RIGHT`, but unlike the known good right-detours it had
  high observed route-intersection load (`obs_route_intersection_count=0.667`;
  known kept wins `207,1185,1438,1509` were `0.0..0.333`). The policy now adds
  `ECML_SEQUENCE_RIGHT_DETOUR_MAX_OBS_INTERSECTIONS=0.5`, only for
  `MOVE_FORWARD -> MOVE_RIGHT`. Targeted validation on
  `207,589,1079,1185,1438,1509,1740,1814` kept all seven known wins and
  neutralized `1814` (`7/0/1` reward, `3/0/5` Success). Re-running
  `1810-1909` after the guard was fully neutral (`0/0/100` reward and
  Success).
- Follow-up holdouts `1910-2209` were clean with the current committed default.
  `1910-2009` had reward `1/0/99`, success `0/0/100`, changing only seed
  `1929` (`+0.073486`, unchanged Success). `2010-2109` had reward `1/0/99`,
  success `0/0/100`, changing only seed `2046` (`+0.106838`, unchanged
  Success). `2110-2209` was fully neutral (`0/0/100` reward and Success).
  The policy remains sparse but has now passed the independent post-guard
  range `1910-2209` without regressions.

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/validate_policy_gate.py \
  --candidate-checkpoint /private/tmp/ecml_ppo_trajectory_anchor2_u3.pt \
  --obs-builder submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --blocks 100-129 130-159 160-189 190-219 220-249 \
  --num-agents 6 \
  --line-length 2 \
  --output-json /private/tmp/ecml_gate_candidate.json
```

Evaluate a 64-feature checkpoint with the matching observation builder:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/evaluate_sampled.py \
  --policy submission.my_policy.MyPolicy \
  --policy-checkpoint /private/tmp/ecml_ppo_trajectory_conflict.pt \
  --obs-builder submission.my_observation_builder.MyTrajectoryConflictObservationBuilder \
  --episodes 50 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2
```

Evaluate a 52-feature checkpoint with the matching observation builder:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/evaluate_sampled.py \
  --policy submission.my_policy.MyPolicy \
  --policy-checkpoint /private/tmp/ecml_ppo_route_conflict.pt \
  --obs-builder submission.my_observation_builder.MyRouteConflictObservationBuilder \
  --episodes 50 \
  --seed 10 \
  --num-agents 6 \
  --line-length 2
```

Initial route-conflict checkpoint results:
- Balanced 12-episode BC
  (`/private/tmp/ecml_bc_route_conflict_12.pt`) collected 30035 valid samples
  from the guarded rerank teacher with `0.929119 / 0.944444` teacher
  reward/success. Pure checkpoint evaluation versus guarded rerank on seeds
  100-129 regressed: guarded rerank `0.936044 / 0.950000`, pure BC
  `0.906905 / 0.933333`; reward wins/losses/ties `5/12/13`, success
  wins/losses/ties `2/4/24`.
- Unbalanced 12-episode BC
  (`/private/tmp/ecml_bc_route_conflict_12_unbalanced.pt`) copied the teacher
  more closely but was still not safe as a pure policy on seeds 100-129:
  `0.902228 / 0.927778`; reward wins/losses/ties `2/15/13`, success
  wins/losses/ties `2/5/23`.
- The existing `submission.bc_gated_policy.MyPolicy` neutralised both
  route-conflict BC checkpoints on seeds 100-129: reward and success
  wins/losses/ties `0/0/30`. This protects the default policy but also shows
  the current gate does not exploit the new route-conflict features.
- A short PPO fine-tune from the balanced BC checkpoint with teacher CE `0.2`,
  conflict-priority penalty `0.05`, slack-progress reward `0.02`, and global
  slack reward `0.02` collapsed badly on seeds 100-129:
  guarded rerank `0.936044 / 0.950000`, PPO checkpoint
  `0.177446 / 0.033333`; reward and success wins/losses/ties `0/30/0`.
  Do not continue with these PPO shaping coefficients. The next RL step should
  either stabilise PPO with much stronger teacher anchoring/smaller updates, or
  learn a new route-conflict-aware gate/reranker from first-difference labels.
- First-difference analysis of the balanced BC checkpoint on the same seeds
  showed most deviations are distance-neutral `MOVE_FORWARD -> MOVE_RIGHT`
  detours. All five reward-positive first deviations had zero route occupancy,
  zero future route-intersection signals and zero candidate prefix conflicts,
  but many reward-negative deviations had the same zero-conflict feature
  pattern. Prefix conflict filters would block obvious bad cases such as seeds
  102 and 114, but they do not separate the silent bad detours. A useful
  52-feature gate likely needs more labelled counterfactual decisions and
  trajectory context, not a single threshold on the current route-conflict
  observation.
- A focused route-conflict counterfactual dataset over the BC win/loss seeds
  102, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 118, 120, 126, 127,
  128 and 129 used `MyRouteConflictObservationBuilder` with forced `LEFT,RIGHT`
  alternatives and produced 204 labelled one-step counterfactuals: reward
  wins/losses/ties `15/14/175`, success wins/losses/ties `2/4/198`.
  This includes mixed labels within the same seed, for example seed 108 and
  seed 111 contain both helpful and harmful side detours.
- Linear, MLP and multiclass counterfactual gates trained on this 204-row
  dataset overfit the training split but failed seedwise cross-split
  validation. Across split seeds `1,3,5,7,11,13,17,19`, all tested objectives
  and feature ablations accepted zero validation good rows at thresholds
  `0.75,0.90,0.95,0.99`, while still accepting some neutral or bad rows. Do not
  deploy a learned 52-feature gate from this dataset alone.
- Broader route-conflict counterfactual mining added three non-overlapping
  seed windows using the same 52-feature builder and forced `LEFT,RIGHT`
  alternatives with four decisions per seed: seeds 130-149 produced 159 rows
  with good/neutral/bad `2/154/3`; seeds 230-249 produced 155 rows with
  `5/144/6`; seeds 430-449 produced 147 rows with `4/132/11`.
  Together with the focused BC win/loss dataset, this gives 665 rows with
  good/neutral/bad `26/607/32`.
- Leave-window-out gate validation across the four datasets is still not
  deployable. Binary MLP gates over all 126 numeric features overfit training
  folds but, on held-out windows, accepted at most one good row and leaked bad
  rows when they did. Multiclass gates with `max_bad_probability=0.05` found
  one held-out good row in three windows only at low thresholds, but always
  alongside neutral and/or bad rows; high thresholds again accepted no useful
  good rows.
- The monotone rescue-rule grid over the combined 665 rows found no non-empty
  zero-bad rule: every zero-bad rule accepted zero good and zero neutral rows.
  A learned or rule-based route-conflict side-detour gate should therefore not
  be deployed yet. The next useful data step is targeted hard-case mining
  around known helpful side-detours and known success-regression side-detours,
  then leave-window-out validation that accepts good rows with zero bad rows.
- `tools/counterfactual_decision_eval.py` now supports
  `--focus-counterfactual-csv`, which converts previously labelled
  counterfactual good/bad rows into seed/agent/time windows. This avoids
  collecting mostly early neutral decisions when the goal is hard-case mining.
- Focused hard-case reruns over all four route-conflict counterfactual CSVs
  produced much denser label sets: an 8-step window produced 60 rows with
  good/neutral/bad `26/2/32`; a 20-step window produced 67 rows with
  `27/7/33`.
- Even with this denser hard-case data, cross-split validation is not safe for
  deployment. For `hard_w8`, the best aggregated binary split result at
  threshold 0.50 accepted good/neutral/bad `37/4/30`; for `hard_w20`, the best
  comparable result accepted `36/14/29`. Higher thresholds reduce accepted
  good rows but still leak bad rows. The monotone rescue-rule grid again found
  no non-empty zero-bad rule on either hard-case dataset.
- The hard-case counterfactual tool now adds decision-time trajectory context:
  route-prefix rejoin features (`prefix_rejoin_*`, `prefix_divergence_len`,
  `prefix_overlap_ratio`) and relative priority-rank features
  (`priority_*`). Regenerating the focused hard-case datasets with these
  columns kept the same labels: `hard_w8_rejoin` has 60 rows with
  good/neutral/bad `26/2/32`; `hard_w20_rejoin` has 67 rows with `27/7/33`.
  MLP split validation still leaked bad rows. The best all-feature binary
  split at threshold 0.50 accepted good/neutral/bad `25/3/33` for w8 and
  `24/7/31` for w20. An expanded monotone rule grid over rejoin and priority
  thresholds found non-empty zero-bad rules for the first time, but only one
  accepted good row (`1/0/0`) in each window. Feature contrasts show weak but
  plausible signals: good side-detours have slightly shorter divergence,
  higher prefix overlap and fewer same-state-priority competitors. This is
  useful as model input, but still not enough for a deployable hard gate.
- A conservative counterfactual value/risk ensemble was added as an offline
  evaluation tool (`tools/evaluate_counterfactual_value_ensemble.py`). It
  trains bootstrap MLPs with a utility target
  `reward_delta + success_weight * success_delta - failed_weight * new_failures`
  and a separate bad-risk head. Acceptance uses a lower confidence bound on
  utility and an upper confidence bound on bad probability. On `hard_w8_rejoin`
  and `hard_w20_rejoin`, the relaxed all-feature ensemble still accepted bad
  rows (for example w8 accepted good/neutral/bad `1/0/1` at
  `min_utility=0.05,max_bad_probability=0.20`; w20 accepted `1/0/2` at
  `0.10,0.10`). More conservative settings with higher bad weights and
  stronger uncertainty penalties accepted nothing. A rerank-feature-only
  ablation over w8+w20 accepted bad rows with negative true utility, while a
  linear rejoin+priority-only model again accepted nothing. Do not integrate
  this offline value ensemble into the submission yet; the next useful RL step
  is to expose these trajectory/priority signals to rollout training rather
  than trying to hard-gate one-step counterfactual labels.

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
- Seeds 10-209 after the residual head-on deadline-slack guard: hybrid
  `0.913843 / 0.938333`, rerank `0.916138 / 0.940000`; reward
  wins/losses/ties `6/0/194`, success wins/losses/ties `2/0/198`.
- Seeds 210-309: hybrid `0.924562 / 0.946667`, rerank
  `0.928594 / 0.948333`; reward wins/losses/ties `5/0/95`, success
  wins/losses/ties `1/0/99`.
- Seeds 310-509: hybrid `0.911885 / 0.935000`, rerank
  `0.913340 / 0.935000`; reward wins/losses/ties `4/0/196`, success
  wins/losses/ties `0/0/200`.
- Seeds 510-709: hybrid `0.920141 / 0.937500`, rerank
  `0.925158 / 0.942500`; reward wins/losses/ties `8/0/192`, success
  wins/losses/ties `4/0/196`.
- Aggregate seeds 10-709 after the residual head-on deadline-slack guard:
  hybrid `0.916614 / 0.938333`, rerank `0.919695 / 0.940476`;
  reward wins/losses/ties `23/0/677`, success wins/losses/ties `7/0/693`.
- Mined failure-deadline + success-hard-negative stress seeds
  (`96` exact seeds from the counterfactual windows): hybrid
  `0.878231 / 0.918403`, rerank `0.878832 / 0.920139`; reward
  wins/losses/ties `2/0/94`, success wins/losses/ties `1/0/95`.

Experimental near-tie side-detour candidate:
- `submission.logit_detour_policy.MyPolicy` is not the Docker default. It adds
  a narrow candidate-generation rule on top of guarded rerank: when guarded
  rerank would move forward, try a distance-neutral side action only if the
  torch action logit is near-tie, current deadline slack is high, the candidate
  has no future head-on risk, no residual tight head-on pair, and no cell or
  edge interaction with planned route prefixes.
- Loose near-tie rules were unsafe: early locally neutral side detours can
  remove later rescue opportunities. On mined low-performing seeds, a loose
  rule regressed seed 452; on seeds 10-209, the tight rule before prefix
  interaction filtering improved four seeds but still regressed two reward-only
  cases.
- After requiring a completely non-interacting candidate prefix, validation
  versus `submission.rerank_policy.MyPolicy` found no local regressions:
  seeds 10-209 `0.916138 / 0.940000` -> `0.917174 / 0.940833`, reward
  wins/losses/ties `1/0/199`, success wins/losses/ties `1/0/199`; mined
  96-seed stress set `0.892065 / 0.878472` -> `0.894223 / 0.880208`, reward
  wins/losses/ties `1/0/95`, success wins/losses/ties `1/0/95`.
- The effect is very sparse and currently comes from seed 207 in these local
  validation sets. Keep this as an experimental policy or ablation candidate,
  not as evidence that broad side-detour expansion should replace the guarded
  rerank default.

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

For seeds 10-209, the current guarded reranker changes six seeds, all positive:
seed 12 `+0.080593`, seed 79 `+0.093583`, seed 88 `+0.092387`, seed 131
`+0.071347`, seed 138 `+0.091575`, and seed 181 `+0.029551`. It blocks the
previous regressions on seed 54 and seed 118.

The residual head-on deadline-slack guard came from the mined stress-set
regressions. Before this guard, seeds 896, 1019, and 1033 regressed by
`-0.133400`, `-0.074405`, and `-0.076453` reward while preserving success. The
bad detours had residual reverse-edge prefix conflicts whose minimum true pair
deadline slack was `80..89`; known good detours with residual reverse-edge
prefix conflicts had at least `95`. The online guard therefore rejects a
candidate side detour only when the residual head-on pair deadline slack is
below `90`, while retaining the broader deadline-conflict penalty as a score
penalty for ETA-near conflicts.

Rescue PPO continuation after the 1910-2209 holdout check:
- `tools/train_masked_ppo.py` was run from
  `submission/models/ecml_rescue_bc_1010_v1.pt` with trajectory conflict
  observations, the curated positive rescue seeds plus hard negatives, 6 PPO
  updates, 4 full episodes per update, low learning rate `1e-5`, teacher CE
  `0.45`, and anchor KL `6.0`.
- The run was numerically stable and stayed close to the rescue BC anchor:
  update success ranged from `0.875` to `1.0`, and anchor KL stayed around
  `1e-5..7e-5`. The temporary checkpoint is
  `/private/tmp/ecml_ppo_rescue_candidate_seed2401_u6.pt`.
- Adding this checkpoint as a fifth Sequence-Gate candidate did not create new
  accepted wins on the targeted validation set
  `207,589,1079,1185,1438,1509,1740,1929,2046,1442,1463,1671,1814,660,946,1160,335,529`.
  The result was identical to the current default Sequence-Gate: reward
  wins/losses/ties `9/0/9`, success wins/losses/ties `3/0/15`, mean reward
  delta `+0.060039`, and mean success delta `+0.027778` versus guarded rerank.
- Conclusion: the PPO continuation is stable but too conservative to promote.
  The next RL step should change the training signal or data distribution, for
  example by mining raw policy differences for new rescue prefixes or by
  training directly on counterfactual rescue actions instead of only staying
  close to the existing rescue BC policy.

Follow-up diff-prefix mining for the same Rescue-PPO checkpoint:
- `tools/mine_diff_prefix_dataset.py` mined the checkpoint as a raw candidate
  on targeted positive and hard-negative seeds
  `207,589,1079,1185,1438,1509,1740,1929,2046,1442,1463,1671,1814,660,946,1160,335,529`
  with prefix lengths `1,2,3`. This produced 54 sequence rows: `20 good`,
  `24 neutral`, and `10 bad`.
- Prefix outcomes were useful but mixed. Prefix `1` had reward
  wins/losses/ties `5/1/12` and success `1/0/17`; prefix `2` had reward
  `7/4/7` and success `3/0/15`; prefix `3` had reward `8/5/5` and success
  `4/0/14`. The new candidate therefore adds dense training signal, but it is
  still unsafe as a raw policy.
- Almost all deviations were `MOVE_FORWARD -> MOVE_RIGHT`; good rows also
  included a few `MOVE_LEFT -> MOVE_FORWARD` corrections. Important hard
  negatives include seed `1814` (`-0.075758` reward), seed `946`
  (`-0.115741` reward), seed `660` (`-0.066348` reward), and the
  prefix-length trap on seed `1463` where prefix `2` is bad but prefix `3`
  becomes good.
- Offline scoring with the current packaged Sequence-Success model and the
  current online thresholds (`min_utility=999`, `max_bad_probability=0.0025`,
  `min_success_probability=0.75`, `bad_std_coef=2`,
  `success_std_coef=0`) accepted `6/0/0` good/neutral/bad rows from this new
  CSV. Accepted seeds were `207`, `1079`, and `1509`; no bad rows were
  accepted. A looser old offline Success threshold `0.02` would accept one bad
  row, seed `1814` prefix `3`, so the current high Success threshold is doing
  real safety work.
- Retraining the Sequence-Success/Risk ensemble by simply adding these 54 rows
  to the current training pool was not safe. On the disjoint
  success-diversity `130-249` validation CSV, the retrained model accepted
  `2/0/8` good/neutral/bad at the current online thresholds, whereas the
  currently packaged model accepted `0/0/0`. Increasing bad weights and bad
  uncertainty did not remove this leak.
- Conclusion: keep `/private/tmp/ecml_diff_prefix_rescue_ppo_seed2401_targeted.csv`
  as diagnostic/training data, but do not promote the new PPO checkpoint or a
  retrained sequence model from this batch. The useful next RL direction is a
  candidate generator that creates genuinely new success-positive motifs, not
  another conservative fine-tune of Rescue-BC or a naive append-only scorer
  retrain.

Second post-check PPO candidate-generation test:
- Trained `/private/tmp/ecml_ppo_successdiv_rescue_mix_seed2501_u6.pt` from
  `submission/models/ecml_ppo_trajectory_successdiv_seed610_u4.pt` instead of
  Rescue-BC. The curated training seeds mixed known success-positive motifs
  (`119,184,207,280,529,589`), newer Rescue-BC wins
  (`1017,1040,1079,1095,1185,1438,1509,1740,1929,2046`), and hard negatives
  (`198,205,211,264,335,578,631,660,905,946,1442,1463,1671,1814`).
  Hyperparameters were still conservative: 6 PPO updates x 4 episodes, low
  learning rate `1.2e-5`, teacher CE `0.30`, anchor KL `4.5`, rollout
  temperature `1.08`, conflict/slack shaping, and stronger terminal team
  success/failure shaping.
- The run stayed numerically controlled: update success ranged from `0.833333`
  to `0.958333`, and anchor KL stayed around `7e-6..3.7e-5`. This is stable
  enough for candidate generation, but not a raw-policy deployment signal.
- Raw checkpoint screening on 33 known positive/hard-negative seeds versus
  guarded rerank was mixed but more diverse than the previous Rescue-PPO:
  reward wins/losses/ties `10/6/17`, success wins/losses/ties `5/1/27`, mean
  deltas `+0.019923` reward and `+0.015152` success. It recovered positives
  on seeds `120`, `184`, `258`, `335`, and `392`, but also regressed reward on
  `119`, `311`, `452`, `477`, `660`, and `905`; seed `905` was a success
  regression. The raw checkpoint is therefore unsafe.
- Adding this checkpoint as a fifth Sequence-Gate candidate produced exactly
  the same online result as the current default on the same 33-seed suite:
  both had reward wins/losses/ties `9/0/24`, success wins/losses/ties
  `3/0/30`, and changed exactly seeds
  `207,589,1079,1185,1438,1509,1740,1929,2046`. The new checkpoint adds no
  online coverage under the current Sequence-Success/Risk gate and guards.
- Conclusion: the diversified PPO run confirms that candidate generation can
  create more raw positive motifs, but the current online gate already filters
  it down to the existing accepted set. More short PPO variants are likely low
  value unless paired with a better learned risk/scoring model or a targeted
  objective that directly optimizes for currently blocked, verified-safe
  motifs such as seed `120`/`184`-style rescues without opening `905`-style
  failures.

Success-regression risk head prototype:
- The raw `ecml_ppo_successdiv_rescue_mix_seed2501_u6.pt` changed 16 seeds on
  the known positive/hard-negative suite, so
  `tools/mine_diff_prefix_dataset.py` mined these seeds with prefix lengths
  `1,2,3,5`:
  `119,120,184,207,280,589,1509,2046,258,311,335,392,452,477,660,905`.
  The resulting CSV
  `/private/tmp/ecml_diff_prefix_successdiv_rescue_mix2501_known_changed.csv`
  has 64 rows: `42 good`, `3 neutral`, and `19 bad`; 20 rows are
  Success-positive and 4 rows are Success-negative.
- This dataset exposes the scoring problem cleanly. Useful new positives
  include seed `120` (`+0.167380` reward, `+0.166667` Success), seed `184`
  (`+0.028539`, `+0.166667`), and reward-only positives `258/335/392`.
  Hard negatives include immediate reward losses on `311/477/660`,
  prefix-length traps on `452`, and seed `905`, where prefix `1` is
  reward-positive but Success-negative and longer prefixes remain
  Success-negative.
- `tools/evaluate_counterfactual_value_ensemble.py`,
  `tools/score_counterfactual_value_ensemble.py`, and
  `submission.sequence_success_policy.MyPolicy` now support an optional
  Success-regression head (`success_delta < 0`) with a separate UCB threshold.
  This is disabled for existing checkpoints/defaults unless a model was
  exported with `include_success_regression_head=True` and
  `ECML_SEQUENCE_MAX_SUCCESS_REGRESSION_PROBABILITY` is set.
- Training the current sequence pool with this extra head, without adding the
  new mix2501 rows to training, and validating on the mix2501 prefix CSV gave
  a clean split result: at `min_utility=999`,
  `max_bad_probability=0.0025`, `min_success_probability=0.75`, and
  `max_success_regression_probability=0.02..0.5`, aggregate validation over
  split seeds `1,3,5` accepted `35/0/0` good/neutral/bad and
  `35` Success-positive rows, with zero Success-negative leakage. The exported
  model `/private/tmp/ecml_success_sequence_with_success_regression_head.pt`
  scored the same mix2501 CSV at `12/0/0`, accepting only seeds `119`, `120`,
  and `589`, and blocking the previously accepted bad seed `905` prefix `2`.
- Extra offline scoring with that exported model stayed clean on checked
  datasets: Rescue-PPO2401 `6/0/0`, Rescue-BC1010 `6/0/0`, terminal
  `490-609` `4/0/0`, seed901 fast suite `0/0/0`, hitemp `130-249` `0/0/0`,
  and successdiv `130-249` `0/0/0`.
- Online integration is not yet promotable. With the new model and new PPO
  candidate but the existing safety guards, a compact online screen
  `119,120,184,207,280,589,905,311,452,477,660` stayed safe but unchanged
  versus current default: only seeds `207` and `589` changed. Trace on seed
  `120` showed the blocked positive comes from the older
  `MOVE_FORWARD -> MOVE_LEFT` slack guard before the scorer can accept it.
- Disabling that left-slack guard (`ECML_SEQUENCE_LEFT_MAX_SLACK=1000000`)
  opened seed `120`, but also reopened the known seed `578` Success regression
  (`-0.063615` reward, `-0.166667` Success). Therefore the new head improves
  offline risk modeling but is not yet sufficient to replace the slack guard.
  Keep the optional model/head support, but do not change the packaged default
  model or default guard thresholds yet.

Detour value guard follow-up:
- Added configurable online guards in `submission.sequence_success_policy`:
  `ECML_SEQUENCE_LEFT_DETOUR_MIN_VALUE_LCB` for `MOVE_FORWARD -> MOVE_LEFT`
  detours, and a low-conflict right-detour guard controlled by
  `ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MAX_CELLS`,
  `ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MIN_SLACK`, and
  `ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MIN_VALUE_LCB`. These guards were
  first added as disabled knobs, then promoted after the holdout checks below.
- Trace diagnosis with the success-regression model showed that seed `120`
  and the known bad seed `578` both pass the Success/Risk heads as
  `MOVE_FORWARD -> MOVE_LEFT`, but differ strongly in value:
  seed `120` has `value_lcb=-0.035726`, while seed `578` has
  `value_lcb=-0.702727`. A left-detour value guard at `-0.4` keeps `120` and
  blocks `578`.
- Opening the left slack guard with only that left value guard produced a
  positive but unsafe 33-seed screen: reward `10/3/20`, success `4/1/28`.
  It recovered `120` and `392`, but reopened reward regressions on `1814`,
  `1442`, and `1463`, including a Success loss on `1442`.
- The regression traces were all low-conflict `MOVE_FORWARD -> MOVE_RIGHT`
  detours from `ecml_rescue_bc_1010_v1.pt`. A stricter low-conflict right
  guard with `cells<=1`, `slack<140`, and `value_lcb<0` neutralized those
  regressions while keeping useful right-detours such as `207`, `1438`, and
  `1509`.
- Best tested variant so far keeps the current packaged sequence model, adds
  the mix2501 PPO checkpoint as a fifth candidate, opens the left slack guard,
  and enables the two new guards:
  `ECML_SEQUENCE_LEFT_MAX_SLACK=1000000`,
  `ECML_SEQUENCE_LEFT_DETOUR_MIN_VALUE_LCB=-0.4`,
  `ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MAX_CELLS=1`,
  `ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MIN_SLACK=140`, and
  `ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MIN_VALUE_LCB=0.0`.
  On the 33 known positive/hard-negative seeds it achieved reward `8/0/25`,
  success `4/0/29`, with reward-delta sum `+1.081390`. Current default on the
  same suite is reward `9/0/24`, success `3/0/30`, reward-delta sum
  `+1.080710`; the new variant trades some reward-only coverage for one more
  Success rescue.
- Fresh holdout `2210-2309` for this guarded-open variant was clean and sparse:
  reward `1/0/99`, success `0/0/100`, changing only seed `2211`
  (`+0.103135` reward, unchanged Success). A second disjoint holdout
  `2310-2409` was fully neutral: reward `0/0/100`, success `0/0/100`.
- No-mix ablation showed the fifth candidate is necessary for the extra seed
  `120` Success rescue: with default candidates only and the same guards,
  known-33 dropped to reward `7/0/26`, success `3/0/30`.
- Promotion: packaged
  `submission/models/ecml_ppo_successdiv_rescue_mix_seed2501_u6.pt`, added it
  to the default candidate list, opened the Sequence left-slack default, and
  set the new guard defaults to the validated values above. This changes the
  default submission from the previous sparse rescue set to the guarded-open
  variant; the edge is still small, so future work should keep validating
  disjoint windows before adding more candidate policies.

Pair-rescue opportunity screen after guarded-open promotion:
- Scoreboard on fresh seeds `2410-2459` compared guarded rerank, promoted
  sequence default, and raw `mix2501` as a single-checkpoint
  `RerankPolicy`. Promoted sequence was fully neutral versus rerank:
  reward `0/0/50`, success `0/0/50`. Raw `mix2501` was mixed:
  reward `1/1/48`, success `1/0/49`, mean deltas `-0.000754` reward and
  `+0.003333` Success.
- The useful raw opportunity was seed `2457`: raw `mix2501` improved
  `0.633271/0.5 -> 0.666667/0.666667`. The matching hard negative was seed
  `2446`: raw `mix2501` regressed reward
  `0.616325/0.5 -> 0.545222/0.5`.
- Prefix mining with lengths `1,2,3,5` showed the important credit-assignment
  pattern. Seed `2457` is not a good one-step action: prefix 1 is neutral, but
  prefix 2 and longer are good (`+0.033396` reward, `+0.166667` Success). The
  first two events are early `MOVE_FORWARD -> MOVE_RIGHT` deviations at
  times `46` and `48`. Seed `2446` is bad from prefix 1 onward and consists
  of repeated `MOVE_FORWARD -> MOVE_LEFT` same-edge detours.
- Current sequence model scoring confirms this is not a threshold tweak:
  `2457` prefix 2 has `success_lcb=0.128794`, `bad_ucb=0.069158`, and
  `value_lcb=-1.038975`, so it is far outside the deployed acceptance region.
  The model has not learned this pair-rescue motif.
- Implication: the next high-value RL/MARL direction is to mine more
  neutral-then-good multi-event prefixes and train a dedicated pair/prefix
  rescue recognizer, or to redesign online gating so it can evaluate short
  candidate plans before accepting the first neutral-looking action. Greedy
  one-event acceptance cannot recover these rescues safely.

Fresh pair-prefix follow-up:
- A raw `mix2501` versus guarded rerank screen on seeds `2460-2559` produced a
  mixed but useful mining set: reward `2/2/96`, success `0/1/99`, mean reward
  delta `+0.000235`, mean Success delta `-0.001667`. Raw positives were
  reward-only seeds `2497` (`+0.066047`) and `2532` (`+0.088095`); raw
  negatives were seed `2496` (`-0.063371` reward, `-0.166667` Success) and
  seed `2520` (`-0.067313` reward).
- Prefix mining on those changed seeds confirmed a second neutral-then-good
  sequence: seed `2497` prefix 1 was neutral, but prefix 2 became reward-good
  after `MOVE_FORWARD -> MOVE_LEFT` followed by `STOP_MOVING -> MOVE_LEFT`.
  Seed `2532` was already reward-good at prefix 1. Seeds `2496` and `2520`
  were bad from prefix 1 onward or after the same two forced events.
- Scoring those 16 fresh prefix rows with the packaged
  `ecml_success_only_sequence_with_rescue.pt` accepted only one row, and it was
  the Success-negative bad seed `2496` prefix 1. The accepted row had
  `success_probability_lcb=0.772876` and `bad_probability_ucb=0.000740`
  despite true `success_delta=-0.166667`. The good reward-only rows were
  rejected, which is expected for the current Success-only acceptance path, but
  the bad acceptance shows the old checkpoint lacks a useful Success-regression
  risk head for these pair-prefix motifs.
- Training the value/risk ensemble with a Success-regression head on the old
  prefix pool plus `2446/2457`, then validating on `2496/2497/2520/2532`,
  accepted nothing at the deployed conservative thresholds
  (`min_utility=999`, `max_bad_probability<=0.05`,
  `min_success_probability>=0.5`). The reverse holdout, training with
  `2496/2497/2520/2532` and validating on `2446/2457`, also accepted nothing
  at those conservative thresholds.
- Lowering the pair Success threshold can recover `2457`: on the
  `2446/2457` holdout, thresholds around `success_lcb>=0.15`,
  `bad_ucb<=0.02`, and Success-regression UCB <= `0.03..0.1` accepted one
  Success-positive good row and no bad rows. However, the same low-threshold
  area leaked bad rows under five-seed cross-validation on the combined
  206-row prefix pool; even `success_lcb>=0.5` and `bad_ucb<=0.01` accepted
  `8` good and `4` bad rows. Do not promote a lower threshold without a much
  larger, pair-specific dataset and a cleaner risk head.

Broader raw-mix screen for additional rescue data:
- Raw `mix2501` versus guarded rerank on fresh seeds `2560-2759` was net
  positive but mixed: reward `13/4/183`, Success `1/1/198`, mean reward delta
  `+0.003875`, mean Success delta `0`. The one new Success-positive seed was
  `2686` (`0.572531/0.666667 -> 0.739198/0.833333`), while the hard
  Success-negative seed was `2573`
  (`0.841486/0.833333 -> 0.674819/0.666667`).
- Prefix mining on the 17 changed seeds produced 68 rows. Prefix `3` had the
  best Success safety in this small set: reward `12/4/1`, Success `1/0/16`.
  Prefix `5` maximized reward (`13/4/0`) but reintroduced the Success loss
  (`1/1/15`). The new Success seed `2686` is not a pair rescue; prefix `1`
  already applies the only needed `MOVE_FORWARD -> MOVE_LEFT` action and is
  Success-positive. Several reward-only wins are true sequence effects:
  `2572` is bad at prefix `2` but good at prefix `3`, `2631` is bad at prefix
  `1` but good at prefix `2`, `2709` is neutral at prefix `1` and good at
  prefix `2`, and `2737` is only good at prefix `5`.
- The packaged `ecml_success_only_sequence_with_rescue.pt` does not recognize
  the new rescue. On the 68 rows it accepted only one neutral row
  (`2665` prefix `2`) and accepted no good rows. For `2686`, all prefixes had
  `success_lcb=0.000622`, `bad_ucb=0.978459`, and `value_lcb=-0.589401`.
- Adding these 68 rows to the value/risk ensemble pool and retraining with a
  Success-regression head did not produce a deployable model. Five-seed
  cross-validation at conservative thresholds still leaked bad rows and
  accepted mostly reward-only rows through the Success head; the best sorted
  strict setting shown (`max_bad_probability=0.0025`,
  `min_success_probability>=0.6`) accepted `5` good, `1` neutral, and `5` bad
  rows, with `0` accepted Success-positive rows. Stronger Success/Risk
  weighting reduced but did not remove leakage (`4/1/1` good/neutral/bad at
  very strict `max_bad_probability=0.0005`, `min_success_probability>=0.85`),
  again with `0` accepted Success-positive rows.
- Conclusion: the current value/risk ensemble is not the right final shape for
  new Success rescues. It confuses reward-good rows with Success-rescue rows
  and does not generalize to `2686`/`2457` safely. The next modeling step
  should be a dedicated Success-rescue classifier or pair/sequence planner
  trained with Success-positive labels, Success-negative labels, and
  reward-only positives kept as separate "value-only" examples instead of
  sharing the same gate acceptance path.
- Added `tools/evaluate_success_rescue_classifier.py` as an offline prototype
  for that separation. It trains two independent bootstrap MLP ensembles:
  one binary classifier for `success_delta > 0` and one unsafe classifier for
  `success_delta < 0`, reward-negative rows, or additional failed agents.
  Acceptance is `success_lcb >= threshold` and `unsafe_ucb <= threshold`.
- On the same 274-row combined pool, this separated classifier improved the
  safety side but not the Success-recall side. With normal thresholds
  (`min_success_probability=0.5..0.95`,
  `max_unsafe_probability=0.001..0.005`) five-seed cross-validation accepted
  zero bad rows in the strict region, but accepted only reward-positive rows
  and `0` Success-positive rows. Lowering the Success threshold as far as
  `0.001..0.3` recovered at most `1-2` Success-positive rows, but immediately
  introduced bad and/or Success-negative leakage. Audit examples show the
  failure modes: `2457` prefix `2` can get high Success score but high Unsafe
  score, while `2686` gets low Unsafe score but near-zero Success score.
- This narrows the next step: the issue is no longer just gate thresholding.
  We need better Success-rescue representation or data construction, likely
  by training on planned sequence/state context where reward-only positives
  are explicitly excluded from the Success head, and by mining more
  Success-positive patterns from additional candidate policies.

Multi-candidate raw screen for Success-diversity:
- A fresh scoreboard on seeds `2760-2839` compared guarded rerank against all
  packaged raw checkpoints. Summary versus rerank:
  `terminal` reward `5/0/75`, Success `0/0/80`, mean reward delta
  `+0.006494`; `successdiv610` reward `1/0/79`, Success `0/0/80`;
  `mix2501` reward `1/2/77`, Success `0/0/80`; `targeted901` reward
  `1/2/77`, Success `1/1/78`; `rescuebc` reward `13/14/53`, Success
  `5/5/70`. The best data miner is therefore `rescuebc`, not because it is
  deployable, but because it exposes both Success rescues and matching hard
  negatives.
- Changed seed sets:
  `rescuebc` changed 27 seeds, with Success-positive
  `2766,2780,2781,2802,2805` and Success-negative
  `2770,2775,2790,2799,2823`. `targeted901` changed only
  `2784,2786,2797`, with `2797` Success-positive and `2786`
  Success-negative.
- Prefix mining for `targeted901` is simple and useful as a unit test:
  `2797` is Success-positive from prefix `1`; `2786` is Success-negative from
  prefix `1`; `2784` is reward-bad from prefix `1`.
- Prefix mining for `rescuebc` produced 108 rows: 37 good, 44 neutral, 27 bad,
  including 10 Success-positive and 10 Success-negative rows. Prefix `1` had
  reward `2/4/21` and Success `1/1/25`; prefix `2` reward `8/5/14`,
  Success `1/2/24`; prefix `3` reward `12/8/7`, Success `4/4/19`; prefix
  `5` reward `14/11/2`, Success `4/3/20`.
- The mined sequences show why one-step gating is structurally weak here:
  `2780` is Success-negative for prefixes `1..3` but Success-positive at
  prefix `5`; `2781` is Success-positive from prefix `1` but reward-negative;
  `2795` is Success-positive at prefix `3` but reward-bad at prefix `5`;
  `2802` and `2805` are neutral for prefixes `1/2` and Success-positive at
  prefix `3`; `2790` becomes Success-negative at prefix `3`; `2823` is a
  final Success-negative raw candidate whose prefix `5` is only reward-good
  and Success-neutral. These are plan-level labels, not isolated action labels.
- The packaged Sequence model still fails on this data. On the RescueBC prefix
  rows it accepted `4` rows at deployed thresholds, but only
  `1/2/1` good/neutral/bad and `0` Success-positive rows; the accepted bad was
  `2795` prefix `5`. On `targeted901` it accepted nothing. The dedicated
  Success/Unsafe classifier with these new rows added increased the pool to
  394 rows (69 Success-positive, 108 unsafe), but still was not deployable:
  strict thresholds accepted only reward-positive rows, while looser
  thresholds recovered at most a few Success-positive rows together with bad
  or Success-negative leakage.
- Current conclusion: we now have the right data shape for a next model, but
  not the right model interface. The next high-value implementation is to add
  candidate/source and plan-prefix context as explicit features and evaluate a
  short-sequence planner/classifier that scores full candidate prefixes, not a
  greedy accepted-event stream.
- Candidate-source feature support was added to the diff-prefix mining and
  online sequence-scoring path. New mined rows now include stable one-hot
  features derived from the checkpoint stem, for example
  `candidate_source_ecml_ppo_trajectory_targeted_seed901_u4` and aggregated
  `event_candidate_source_*` features. The online Sequence policy sets the
  same features from its candidate checkpoint path before scoring. This is
  backwards compatible with the current packaged model because unknown feature
  columns are ignored; it only affects future models trained with these new
  columns. Smoke mining on `targeted901` seed `2797` produced the expected
  source columns, and a one-episode online smoke on seed `2797` still ran.
- Re-mining the `rescuebc` and `targeted901` datasets with source features did
  not by itself fix the Success/Unsafe classifier. The combined CV pool still
  had 394 rows, now with 737 feature columns. With source features included,
  strict thresholds accepted only reward-good rows and no Success-positive
  rows; excluding `candidate_source` reproduced the previous non-source
  behavior, where the soft region could recover a few Success-positive rows
  only together with bad/Success-negative leakage. Training only on the new
  source-featured `rescuebc+targeted901` rows was worse: strict thresholds
  accepted nothing, and softer thresholds accepted only Success-negative bad
  rows.
- `tools/evaluate_success_rescue_classifier.py` now has
  `--reward-negative-unsafe-mode non_success`. This avoids labeling
  Success-positive but reward-negative rows as unsafe, which is important for
  cases like `2781`. The corrected label mode reduced the combined pool's
  unsafe labels from 108 to 95, but still was not deployable: at strict
  thresholds it accepted only reward-good plus reward-bad rows and `0`
  Success-positive rows; at softer thresholds it recovered at most one
  Success-positive row while also accepting a Success-negative row and several
  bad rows. The issue is therefore not just source identity or this unsafe
  label definition; the next model needs richer plan-state representation or
  a different sequence objective.
- Added `tools/evaluate_sequence_rescue_planner.py` as the first offline
  short-sequence prototype. It reads the JSON prefix rows, keeps the ordered
  `event_details`, encodes them with a small GRU, and trains separate
  Success-positive and Unsafe heads. By default it drops the flattened
  top-level `event_*` aggregate features so we can test whether ordered event
  context alone carries signal. It also reports a best-by-seed planner metric:
  for each validation seed it selects at most one prefix by
  `success_lcb - unsafe_weight * unsafe_ucb`, which is closer to the intended
  rescue decision than accepting every row above a threshold.
- Pure ordered-event input is not enough yet. On the 394-row combined pool
  (`--reward-negative-unsafe-mode non_success`, 5 split seeds, 5-member
  ensemble, 350 epochs), threshold acceptance found `0` Success-positive
  rows and leaked bad rows. A separate score audit showed the Success head can
  assign high scores to some positive rows, but the Unsafe head is poorly
  calibrated and blocks them; relaxing Unsafe thresholds recovers positives
  only together with too much bad/Success-negative leakage.
- Adding the flattened aggregate features back (`--no-drop-aggregate-event-features`)
  improves the planner-style signal but still is not deployment-safe. The
  best-by-seed planner metric produced positive net Success in several
  regions, for example `unsafe_weight=0.25`, `score_threshold=0.0` selected
  `18/15/9` good/neutral/bad rows with `7` Success-positive and `1`
  Success-negative, reward delta `+0.801671`, and Success delta `+1.0`.
  Stricter no-Success-negative regions exist, for example
  `unsafe_weight=0.1`, `score_threshold=0.25` selected `11/4/7` with `3`
  Success-positive, `0` Success-negative, reward delta `+0.154851`, and
  Success delta `+0.5`, but still leaked too many bad rows for online use.
- Current conclusion: the direction is useful for diagnosis but not yet a
  winning submission component. The important insight is that per-seed
  prefix selection is much better aligned with the problem than greedy
  row-thresholding, but the labels and features still do not separate
  Success rescues from reward-bad traps reliably. Next high-value step is to
  move the objective from binary row classification toward seed-level
  candidate ranking or imitation of the best counterfactual prefix, with
  evaluation on held-out fresh seeds before any online integration.
- Added `tools/evaluate_sequence_rescue_ranker.py` for that next objective.
  It reuses the sequence/static feature pipeline, but trains a RankNet-style
  pairwise objective within each seed group: prefixes with higher target value
  should score above weaker prefixes from the same seed. The target value is
  Success-first but still reward-aware:
  `success_weight * success_delta + reward_weight * reward_delta -
  reward_loss_penalty * max(-reward_delta, 0) -
  success_loss_penalty * max(-success_delta, 0) -
  failure_penalty * max(failed_delta, 0)`. A small pointwise MSE term keeps
  scores roughly calibrated so a threshold can choose "no rescue".
- On the 394-row combined pool, the ranker is the first offline model that
  clearly improves over the binary planner/classifier direction. With
  `reward_loss_penalty=3.0`, `pointwise_weight=0.25`, five split seeds, and a
  5-member ensemble, the high-threshold region selected useful rescues with
  no Success-negative leakage: at `score_threshold=1.25` it selected
  `6/0/3` good/neutral/bad rows, with `5` Success-positive, `0`
  Success-negative, reward delta `+0.455534`, Success delta `+0.833333`, and
  failed-agent delta `-5`; at `score_threshold=1.5` it selected `5/0/2`,
  with `4` Success-positive, `0` Success-negative, reward delta `+0.271882`,
  and Success delta `+0.666667`.
- The remaining blocker is still bad-leak calibration. The top selected
  audit rows include reward-negative traps such as seed `2446`, which can
  score above true rescues in some validation splits despite the reward-loss
  penalty. Increasing `pointwise_weight` to `1.0` reduced some high-threshold
  leakage but also reduced recall too much (`score_threshold=1.25` only
  selected `2/0/1`, with `2` Success-positive and `0` Success-negative).
  This suggests the rank objective is better aligned, but the model still
  needs either more hard negatives around these trap patterns, an explicit
  no-op/baseline candidate in each seed group, or a second calibrated safety
  head before online integration.
- `tools/evaluate_sequence_rescue_ranker.py` now supports
  `--add-baseline-candidate`, which adds one synthetic no-op candidate per
  seed group with zero deltas and empty `event_details`. The evaluation does
  not count selected baseline rows as accepted rescues, and the printed
  metrics include `selected_baseline`/`selected_nonbaseline` so we can see
  whether the model chooses "do nothing" instead of a risky rescue.
- The first no-op CV did not solve the trap issue. With default ranker
  weights plus synthetic baseline candidates, the model selected baseline for
  25 validation seed-groups across the five splits, but still ranked several
  bad rows above baseline. At `score_threshold=1.5`, it selected `5/0/3`
  good/neutral/bad rows with `4` Success-positive and `1` Success-negative;
  at `score_threshold=2.0`, it selected `1/0/1`, still leaking a bad seed
  (`2446`).
- A stronger no-op variant (`reward_loss_penalty=8.0`,
  `pair_weight_scale=4.0`, `max_pair_weight=12.0`,
  `pointwise_weight=0.5`) reduced leakage but lost too much useful recall.
  At `score_threshold=1.25`, it selected `5/0/2` with `4` Success-positive
  and `0` Success-negative, reward delta `+0.334147`, Success delta
  `+0.666667`; at `score_threshold=1.5`, only `2/0/1` remained. This is
  safer but weaker than the previous non-baseline ranker.
- Current conclusion: explicit no-op is the right interface, but a synthetic
  baseline row is not enough to calibrate scores. The next useful work should
  mine and overweight hard-negative families around the observed trap seeds
  (`2446`, `660`, `2573`, `905`, `946`) and/or combine the ranker with a
  separate calibrated safety head. This is a signal that we should not deploy
  the ranker yet; it is a better offline learning objective, not a finished
  online controller.
- Mined targeted hard-negative families for trap seeds
  `2446,660,2573,905,946` with prefix lengths `1..5` and the trajectory
  conflict observation builder. Three candidate sources were used:
  `mix2501`, `rescuebc`, and `targeted901`. The first attempt without the
  trajectory conflict obs failed with a `41` vs `64` feature-shape mismatch,
  confirming again that these checkpoints must be mined/evaluated with
  `submission.my_observation_builder.MyTrajectoryConflictObservationBuilder`.
- The targeted trap datasets are useful and compact:
  `mix2501` produced 20 rows over seeds `660,905,2446,2573`, with `19` bad,
  `1` neutral, `0` Success-positive, `7` Success-negative, and reward
  `4/15` positive/negative; `rescuebc` produced 25 rows over all five seeds,
  with `3` good, `14` neutral, `8` bad, and no Success changes;
  `targeted901` produced 15 rows over `660,905,2573`, with `4` good,
  `10` neutral, `1` bad, and `1` Success-negative. This is the right kind of
  data: dense around known traps rather than broad random mining.
- Adding these hard-negative families to the no-op ranker pool improved the
  useful high-threshold region materially. With default ranker weights,
  no-op candidates, and the added trap rows, `score_threshold=1.25` selected
  `8/1/3` good/neutral/bad rows, `7` Success-positive, `0`
  Success-negative, reward delta `+0.466069`, Success delta `+1.166667`,
  and failed-agent delta `-7`. This is better than the previous no-op CV at
  the same threshold (`8/1/6`, `5` Success-positive, `1` Success-negative,
  reward `+0.326446`, Success `+0.666667`).
- The remaining high-score bad rows are now concentrated and interpretable:
  mainly seed `2446` prefixes `2/5`, seed `2573` prefix `1`, and seed `660`
  prefix `5`. The separate Success/Unsafe classifier did not veto them:
  pure reward-negative rows received very low `unsafe_probability_ucb`
  despite being labeled unsafe, so combining ranker plus current unsafe head
  does not fix this failure mode.
- Added optional negative-score margin terms to
  `tools/evaluate_sequence_rescue_ranker.py`:
  `--negative-margin-weight`, `--negative-target-threshold`, and
  `--negative-score-margin`. The idea is to force rows with negative target
  value below an absolute score margin, not just below better candidates.
  Ablations did not solve the issue: margin weight `2.0` was nearly
  unchanged, while margin weight `10.0` reduced some leakage but also reduced
  recall and worsened the useful `score_threshold=1.25` region (`6/1/4`,
  `5` Success-positive, `0` Success-negative, reward `+0.437928`, Success
  `+0.833333`).
- Current conclusion: targeted hard-negative mining is the right direction
  and already improved the offline ranker, but this model still cannot
  reliably identify small reward-only traps. The next step should not be more
  threshold tuning; it should be either (1) a dedicated reward-loss/risk head
  trained specifically on small reward-negative traps, or (2) mining more
  local variants around `2446/2573/660` until the ranker sees enough nearby
  positives/neutral no-ops to learn the boundary.
- Added `tools/evaluate_reward_risk_head.py` as a dedicated reward-loss/risk
  veto prototype. It trains a single bootstrap MLP ensemble whose label is
  risk-focused rather than Success-focused: negative utility target,
  Success-regression, additional failed agents, or reward loss in
  `non_success` mode. The tool can also join a ranker audit CSV and evaluate
  "ranker selects, risk head vetoes" with matching CV split seeds. Duplicate
  joined rows are handled conservatively by using the highest
  `risk_probability_ucb` for a key.
- On the hard-negative-augmented pool, the standalone risk head is very
  conservative at strict thresholds and has low precision, but that is
  acceptable for veto use. With `risk_positive_weight=40`, 5 split seeds, and
  a 5-member ensemble, `max_risk_probability=0.0005` reached risk recall
  `0.883721` while over-vetoing many non-risk rows.
- As a ranker veto, this is the first offline configuration that removes the
  known high-score reward traps cleanly. Using the hard-negative no-op ranker
  audit, `ranker_score_threshold=1.25` and
  `veto_max_risk_probability=0.0005` changed the selected set from the raw
  ranker's `8/1/3` good/neutral/bad (`7` Success-positive, `0`
  Success-negative, reward `+0.466069`, Success `+1.166667`) to `5/1/0`
  good/neutral/bad (`4` Success-positive, `0` Success-negative, reward
  `+0.499341`, Success `+0.666667`, failed-agent delta `-4`). The vetoed rows
  include the main bad traps `2446` prefix `2`, `2446` prefix `5`, and
  `2573` prefix `1`; it also vetoes some true rescues (`2802`, `2457`,
  `184`), so recall is still the tradeoff.
- Current conclusion: the best current architecture is a two-stage offline
  planner: seed-level ranker for rescue value, then a strict reward-risk veto.
  This is meaningfully better aligned than the earlier gate/classifier
  approach. It is still not ready for submission because the validation pool
  is small and partly built from known traps; next step is fresh held-out
  validation/mining specifically for this two-stage design before any online
  integration.
- Fresh held-out validation on seeds `2840..2879` was mined with the trajectory
  conflict observation builder and the same three candidate families. The
  candidate pool is broad enough to be informative but still not a true
  competition-distribution test because the candidate generators and scenario
  settings are the same as before. Summary: `mix2501` produced 120 rows
  (`20` good, `91` neutral, `9` bad, Success `+10/-0`), `rescuebc` produced
  200 rows (`29` good, `144` neutral, `27` bad, Success `+22/-9`), and
  `targeted901` produced 65 rows (`5` good, `56` neutral, `4` bad, no Success
  changes). Combined validation pool: 385 rows, `54` good, `291` neutral,
  `40` bad, Success `+32/-9`, reward `+49/-45`.
- The hard-negative-trained sequence ranker did not hold up on this fresh
  held-out block. Conservative thresholds avoided Success-negative leaks but
  failed to find useful rescues: at `score_threshold=1.25` it selected
  `0/18/4` good/neutral/bad, reward delta `-0.111330`, Success delta `0.0`;
  at `1.5` it selected `0/5/1`, reward delta `-0.012397`, Success delta
  `0.0`. Lowering to `0.75` recovered some value (`11/42/13`,
  reward `+0.801694`, Success `+2.500000`, failed-agent delta `-15`) but
  with too many bad leaks for a safe online controller.
- The standalone reward-risk head is useful as a broad risk detector on this
  held-out block, but not as a rescue selector. At
  `max_risk_probability=0.0005` it kept `48/279/16` good/neutral/bad with
  reward `+7.389159`, Success `+11.666667`, and no Success-negative accepted,
  while still accepting 16 bad rows. This says the risk signal has coverage,
  but it is too imprecise to replace ranking.
- Combining the held-out ranker with the strict risk veto did not rescue the
  two-stage architecture. Because the ranker placed no good rows above the
  conservative thresholds, the veto mostly filtered neutral/bad candidates:
  `ranker_score_threshold=1.25` and `veto_max_risk_probability=0.0005`
  accepted only `0/0/1`, reward `-0.074140`, Success `0.0`; at
  `ranker_score_threshold=1.0` and the same veto it accepted `0/2/2`, reward
  `-0.148280`, Success `0.0`. Relaxing the veto recovers at most the raw
  ranker's weak held-out behavior.
- Current conclusion after fresh held-out: the bottleneck is no longer the
  veto; it is ranking/generalization. The previous trap-seed improvement was
  real on known hard negatives, but it overstates deployment readiness. The
  next high-value work should shift from threshold tuning to changing the
  learning problem: train on seed-level decisions with explicit baseline/no-op
  competition, add richer conflict-horizon/global occupancy features, and
  evaluate on unseen seed blocks before accepting any online integration.
- Added `tools/evaluate_group_rescue_ranker.py`, a listwise seed-level
  selector. It trains each seed group as a candidate list with an explicit
  synthetic baseline/no-op candidate, then evaluates by selecting the best
  non-baseline candidate only if its conservative score margin beats the
  baseline by a configurable threshold. This removes the previous dependency
  on an absolute utility score and makes the offline objective closer to the
  deployment decision: "intervene or do nothing".
- On the same fresh `2840..2879` held-out block, the group ranker with all
  features did not solve the problem. At margin `0.0` it selected `8/24/12`
  good/neutral/bad, with reward `-0.043356`, Success `+1.333333`, and two
  Success-negative leaks. Higher margins reduced recall but still left
  bad rows; lower margins increased reward but also leaks. Conclusion:
  listwise training alone is not enough.
- Removing candidate-source features helped generalization slightly. With
  `--exclude-feature-regex candidate_source` and
  `--exclude-event-feature-regex candidate_source`, high margins
  `1.0..1.5` selected `2/7-11/5` good/neutral/bad, no Success-negative
  leaks, reward `+0.072384`, Success `+0.666667`, failed-agent delta `-4`.
  This suggests source tags were contributing to overfit, but removing them
  does not create enough signal by itself.
- Combining the source-ablated group ranker with the existing strict
  reward-risk veto is the safest fresh-heldout configuration so far, but it
  has low recall. With ranker margin thresholds `0.75..1.5` and
  `veto_max_risk_probability=0.0005`, it accepted `1` good, `0` bad,
  `3-5` neutral rows depending on margin, reward `+0.431373`, Success
  `+0.5`, failed-agent delta `-3`, and no Success-negative leaks. This is a
  useful conservative rescue filter, not yet a high-impact winning policy.
- Current conclusion after the group-ranker experiment: the right direction
  is to keep the conservative "intervene only when strongly justified" stack
  as a safety baseline, but the main research work should move to richer
  conflict-horizon/global occupancy features and/or actual policy training
  that sees those features online. The offline candidate ranker can now find
  one clean held-out rescue, but it is too low-recall to close the gap to a
  winning competition solution by itself.
- Added an opt-in action-conflict observation builder:
  `submission.my_observation_builder.MyActionConflictObservationBuilder`.
  It keeps the existing 64-dimensional trajectory-conflict observation intact
  and appends 15 per-action features for `LEFT/FORWARD/RIGHT`: local validity,
  target-distance delta, route-prefix conflict count, head-on edge conflicts,
  and opposing-direction intersections. The resulting feature size is `79`
  (`84` values including the appended 5-action mask). Existing checkpoints and
  builders are unchanged.
- Wired the new observation into the trainable paths with
  `--use-action-conflict-obs`: `tools/train_masked_ppo.py`,
  `tools/train_behavior_clone.py`, and `tools/train_rescue_behavior_clone.py`.
  This gives us a cleaner path toward learned action selection: first warmstart
  a 79-dim policy by BC, then run PPO on the same online features. The features
  are intentionally not a hard rule or gate; they expose action-specific
  medium-term conflict consequences to the policy.
- Smoke tests passed. A reset on seed `2840`, `scene_5`, 6 agents produced
  feature length `79` and total observation length `84` for all agents. A
  one-update PPO smoke run completed with
  `obs_builder=MyActionConflictObservationBuilder`, `obs_size=79`, and saved
  `/private/tmp/ecml_action_conflict_smoke.pt`. A one-episode BC smoke run also
  completed and saved `/private/tmp/ecml_action_conflict_bc_smoke.pt`; the
  collection had 3213 valid teacher samples and 155 teacher/reference
  disagreements.
- Current conclusion after adding action-conflict obs: this is the right
  infrastructure step toward less heuristic policy learning, but it is not yet
  evidence of a stronger policy. The next test should be a controlled
  BC-warmstart plus PPO run on unseen seeds, compared against the current
  trajectory-conflict PPO/rescue stack and starterkit baseline.
- Controlled action-conflict BC warmstart:
  `/private/tmp/ecml_action_conflict_bc_8ep.pt` trained 8 episodes on seeds
  `3200..3207` with `MyActionConflictObservationBuilder`, class-balanced CE,
  and `submission.rerank_policy.MyPolicy` as teacher. Collection had 22587
  valid teacher samples, only 3 teacher/reference disagreements, and training
  accuracy around `0.997`. Held-out scoreboard on `2840..2859` versus current
  default was negative: current `0.897936 / 0.925000`, action-BC
  `0.855028 / 0.908333`, reward delta `-0.042909`, Success delta
  `-0.016667`, reward W/L/T `2/9/9`, Success W/L/T `1/2/17`. Conclusion:
  pure imitation of the current teacher does not exploit the new features.
- Conservative PPO fine-tune from that BC checkpoint
  `/private/tmp/ecml_action_conflict_ppo_3200_u4.pt` was numerically stable
  but did not change the deterministic policy on the held-out screen.
  Hyperparameters were deliberately conservative: 4 updates x 3 complete
  episodes, `lr=1e-5`, `teacher_ce=0.35`, `anchor_kl=6.0`, small slack/global
  shaping and conflict-priority penalty. Anchor KL stayed tiny
  (`~1e-6..2.6e-5`), rollout Success ranged `0.777778..1.0`, and the external
  `2840..2859` scoreboard was identical to action-BC. Conclusion: this PPO
  setup is stable but too anchored to create useful new actions.
- Added `tools/convert_diff_prefix_to_rescue_events.py` to convert positive
  diff-prefix JSON rows into event-level labels usable by
  `tools/train_rescue_behavior_clone.py`. This is an approximation: prefix
  success may require a sequence of actions, but the converter emits each
  positive prefix event as an individual supervised rescue label. Converting
  the current positive/diagnostic JSON pool produced 191 event labels over
  53 seeds in `/private/tmp/ecml_diff_prefix_positive_rescue_events.csv`.
- Action-conflict Rescue-BC from converted prefix labels:
  `/private/tmp/ecml_action_conflict_rescue_bc_v1.pt` trained from the 191
  converted labels plus anchor seeds `3200..3215`. Collection replay hit
  `167` labels, had `0` misses, `24` invalid labels, and `18` baseline
  mismatches; it collected 46912 anchor samples. Held-out `2840..2859`
  remained unsafe as a raw policy: `0.856323 / 0.900000`, reward delta
  `-0.041613`, Success delta `-0.025000`, reward W/L/T `3/6/11`, Success
  W/L/T `1/2/17` versus current default. It found useful wins on seeds
  `2842` (`+0.112309` reward) and `2854` (`+0.166667` Success), but leaked
  large losses on `2847`, `2845`, `2852`, `2855`, `2846`, and `2859`.
- Current conclusion after the first action-conflict learning cycle: the new
  observation is technically sound and raw learned policies can generate new
  positive actions, but raw BC/PPO is still not safe. The best next use is as
  candidate-generation data for the offline sequence/risk stack: mine
  first-diff/prefix outcomes from `ecml_action_conflict_rescue_bc_v1.pt`,
  add its wins and hard negatives to the ranker/risk datasets, and only deploy
  through a strict scorer/veto.
- Mined prefix outcomes from the raw action-conflict Rescue-BC checkpoint on
  the changed held-out seeds `2842,2845,2846,2847,2850,2852,2854,2855,2859`.
  Output:
  `/private/tmp/ecml_diff_prefix_action_rescue_bc_v1_2840_2859_changed.{csv,json}`.
  This produced a compact mixed sequence dataset with clear positives and hard
  negatives:
  - prefix `1`: reward `1/2/6`, Success `2/0/7`, mean delta
    `-0.011289/+0.037037`
  - prefix `2`: reward `0/4/5`, Success `1/2/6`, mean delta
    `-0.043375/-0.018519`
  - prefix `3`: reward `2/2/5`, Success `1/0/8`, mean delta
    `-0.031793/+0.018519`
  - prefix `4`: reward `1/4/4`, Success `1/0/8`, mean delta
    `-0.056872/+0.018519`
  - prefix `5`: reward `3/3/3`, Success `1/0/8`, mean delta
    `-0.022509/+0.018519`
- Seed-level examples are useful for the next scorer iteration. `2842` only
  becomes reward-positive at prefix `5`; early prefixes are neutral. `2850`
  has a genuine prefix-1 Success win. `2854` has repeated Success-positive
  prefixes despite some reward loss. Hard negatives include `2855` (all
  prefixes reward-negative while Success is unchanged), `2852` prefix `2`
  Success regression and prefixes `4/5` reward losses, and `2847` prefix `2`
  Success regression. This is exactly the signal the current offline
  ranker/risk stack needs: positive rescues plus nearby traps from the same
  action-conflict generator.
- Fresh scorer A/B on a disjoint held-out slice `2860..2879` was used to
  avoid evaluating on the same seeds that produced the action-conflict rescue
  prefix rows. The filtered validation pool had 200 rows: 23 good, 154
  neutral, 23 bad; Success `+14/-4`, reward `+23/-23`.
- Naively appending
  `/private/tmp/ecml_diff_prefix_action_rescue_bc_v1_2840_2859_changed.json`
  to the group-ranker training pool was negative. With candidate-source
  features excluded and the same `2860..2879` validation slice, high ranker
  margins selected no useful rescues: margin `1.5` selected `0/5/4`
  good/neutral/bad, reward `-0.416667`; margin `1.25` selected `0/5/6`,
  reward `-0.564947`; margin `1.0` selected `0/6/7`, reward `-0.669113`.
  Lowering to `0.75` recovered two good rows but still leaked eight bad rows.
  Conclusion: action-conflict rescue rows are not safe ranker training data in
  their current small, biased form.
- The same action-conflict rows are safer as risk-head data if the baseline
  source-ablated group ranker stays unchanged. With matching split seeds,
  `ranker_score_threshold=1.0..1.5` and
  `veto_max_risk_probability=0.0005` still accepted exactly the same clean
  rescue as the previous baseline stack: `1/0/0` good/neutral/bad, Success
  `+0.5`, reward `+0.431373`, failed-agent delta `-3`. At margin `0.75` the
  action-augmented risk head leaked one bad row, so the strict high-margin
  regime remains the only acceptable setting.
- Standalone, the action-augmented risk head became more permissive on the
  fresh slice: at `max_risk_probability=0.0005` it accepted
  `70/247/8` good/neutral/bad across the five validation splits versus the
  previous baseline risk head's `41/165/7`, with risk recall dropping slightly
  from `0.939130` to `0.930435`. This is not a deployment improvement; it is
  only evidence that the extra rows do not break the strict ranker+veto stack
  when the ranker is kept fixed.
- Current decision: do not train the ranker directly on the first
  action-conflict rescue batch. Keep those rows as diagnostics and possible
  risk-head/hard-negative material. The next step toward a higher-recall,
  less heuristic solution should be to enlarge the action-conflict data on
  disjoint seeds and train a source-specific action-conflict scorer or policy,
  instead of mixing this small generator-specific batch into the shared
  ranker.
- Mined a second disjoint action-conflict Rescue-BC prefix block on seeds
  `2880..2919` with prefix lengths `1..5`. Output:
  `/private/tmp/ecml_diff_prefix_action_rescue_bc_v1_2880_2919_changed.{csv,json}`.
  The 200 rows contain 25 good, 162 neutral, and 13 bad rows. Prefix `5` is
  the best raw slice by reward (`7/3/30` reward win/loss/tie), but still has
  Success `+1/-3/36`, so this is more diagnostic/training material rather
  than deployable evidence. Positive seeds include `2883`, `2892`, `2913`,
  `2916`, and `2918`; clear Success-regression hard negatives include
  `2882`, `2889`, `2890`, `2891`, and `2908`.
- Retraining the source-ablated group ranker with both action-conflict
  batches improved the raw row-level table relative to the first naive
  action-ranker, but it is still not safe. On the unchanged `2860..2879`
  validation slice, margin `1.25` selected `1/2/4` good/neutral/bad with
  reward `+0.044733`, while margin `1.0` selected `2/2/5` and reward
  `+0.371938`. The strict source-aware Risk head can veto this back to clean
  row-level metrics at margin `1.0`, risk `0.0005`, but row-level counts are
  misleading because the same seed/event can appear once per split.
- Added unique-candidate accounting to
  `tools/evaluate_reward_risk_head.py` for ranker+veto reports. The printed
  and JSON veto metrics now include `unique_ranker_selected` and
  `unique_accepted_*` fields keyed by seed, prefix, forced event sequence, and
  outcome deltas. This prevents cross-split duplicates from being read as
  multiple rescues.
- After the unique accounting fix, the apparent action-ranker recall gain
  disappears. The best clean setting for the two-action-batch ranker plus the
  source-aware Risk head is still one unique held-out rescue: margin `1.0`,
  risk `0.0005` gives `unique_accepted_good=1`, `unique_accepted_bad=0`,
  unique reward `+0.431373`, unique Success `+0.5`. The row-level report showed
  two accepted good rows only because seed `2872` was accepted in two split
  models. Current conclusion: the path is producing useful candidate data, but
  the shared offline ranker still does not generalize enough. Next work should
  either train a source-specific action-conflict scorer/policy or change the
  evaluation/training protocol to one deployment model per held-out block,
  not cross-split row aggregation.
- Added `tools/summarize_ranker_veto_deployment.py` to make that deployment
  interpretation explicit. It reads a group-ranker audit and a risk-head audit,
  then reports per-split model metrics plus consensus metrics over unique
  seed/prefix/event candidates. This avoids treating five training seeds as
  five deployment rescues.
- Deployment-style summary on `2860..2879`, margin `1.0`, risk `0.0005`:
  the baseline source-ablated ranker plus source-aware Risk head accepts the
  clean `2872` rescue in one model split. The two-action-batch ranker plus the
  same Risk head accepts the same `2872` rescue in two model splits, and the
  `consensus_min_splits=2` metric therefore keeps one unique good candidate
  with reward `+0.431373`, Success `+0.5`, and no bad candidates. This is a
  stability improvement for the known rescue, not a recall improvement.
- Current decision after the deployment-style check: action-conflict mining is
  giving us real signal, but the current shared ranker mostly learns to
  rediscover one easy rescue. To move toward a winning RL/MARL solution, the
  next high-value work should stop optimizing this offline shared ranker and
  instead train/evaluate a source-specific action-conflict policy/scorer with a
  deployment-style validation protocol from the start.
- Started that source-specific check with a fresh action-conflict validation
  block on seeds `2920..2959`, using the same
  `/private/tmp/ecml_action_conflict_rescue_bc_v1.pt` candidate and prefix
  lengths `1..5`. Output:
  `/private/tmp/ecml_diff_prefix_action_rescue_bc_v1_2920_2959_changed.{csv,json}`.
  This block is more positive than `2880..2919`: prefix `1` has reward
  `5/0/35` and Success `+2/-0/38`, prefix `2` has reward `6/0/34`, and
  prefix `5` is high-variance with reward `8/7/25` and Success `+3/-3/34`.
  It contains the right validation structure for source-specific learning:
  robust positives (`2931`, `2935`, `2936`, `2940`, `2941`, `2949`) and
  nearby traps (`2924`, `2926`, `2928`, `2937`, `2939`, `2954`, `2955`).
- Action-only group-ranker experiment: train on the two previous
  action-conflict blocks (`2840..2859`, `2880..2919`) and validate on
  `2920..2959`. Raw ranker at margin `0` selected `2` good, `1` neutral, `0`
  bad rows, reward `+0.149737`, Success `+0.333333`; all higher margins were
  empty. Adding an action-only Risk head kept the result clean but did not
  improve recall. Deployment summary with margin `0`, risk `0.001` found one
  stable unique good candidate under `consensus_min_splits=2`: seed `2936`,
  prefix `1`, event `MOVE_FORWARD->MOVE_RIGHT`, reward `+0.074868`, Success
  `+0.166667`, failed-agent delta `-1`. It also accepted one neutral candidate
  under `consensus_min_splits=1`.
- Current conclusion after the first source-specific scorer: the action
  conflict source is learnable, but the current offline scorer recognizes only
  one of many obvious positives in a favorable validation block. This is not a
  winning path by itself. The next useful change should move closer to online
  RL/policy learning with these features: either use the mined positives as
  targeted BC/RL curriculum for the action-conflict policy, or train a
  lightweight online value/rescue head inside the policy instead of selecting
  offline prefix rows after the fact.
- Targeted action-conflict Rescue-BC curriculum, fair split:
  converted positive prefix rows from `2840..2859` and `2880..2919` into
  53 event labels over 11 seeds, then trained
  `/private/tmp/ecml_action_conflict_rescue_bc_v2_train2840_2919.pt` from
  `/private/tmp/ecml_action_conflict_rescue_bc_v1.pt` with low LR
  (`1e-5`) and anchor seeds `3200..3215`. Collection hit 48 labels, missed
  none, and had 5 invalid labels. Raw held-out comparison on `2920..2959`
  was worse than v1: v2 reward delta mean `-0.018101` and Success delta
  `+0.016667` versus v1 reward delta mean `-0.015690` and Success delta
  `+0.020833`. Positive-only curriculum fixed seed `2925` but damaged
  `2936` and `2958`; net effect was negative.
- Added negative-aware curriculum support. `tools/convert_diff_prefix_to_rescue_events.py`
  now supports `--include-negative-baseline`, which emits harmful prefix
  events as `event_kind=negative_baseline` baseline-action labels while
  skipping negative events that also appear in a positive prefix. `tools/train_rescue_behavior_clone.py`
  now supports `--include-avoidance-events` and `--avoidance-weight`, so those
  rows can train explicit "do not take the candidate action here" labels.
- Negative-aware v3 curriculum on the same fair split produced 93 labels
  (53 positive rescue, 40 negative baseline) over 21 seeds. Training
  `/private/tmp/ecml_action_conflict_rescue_avoidance_bc_v3_train2840_2919.pt`
  hit 73 labels, including 25 avoidance hits, with 20 invalid labels. Raw
  held-out comparison on `2920..2959` was still not deployable and did not beat
  v1: reward delta mean `-0.017357`, Success delta `+0.020833`, reward
  wins/losses/ties `9/16/15`, Success `5/3/32`. It preserved v1's Success
  gains but reduced a reward-positive seed (`2945`) and did not eliminate the
  major traps (`2924`, `2922`, `2944`, `2937`, `2954`).
- Current conclusion after targeted BC curricula: event-level BC is too blunt
  for this problem. It can nudge local actions, but the good/bad distinction
  depends on rollout context and prefix length. The next RL step should not be
  another plain BC pass. It should either optimize an online objective with
  counterfactual/RL feedback, or add a learned online value/risk head that
  scores the candidate action before the ActorCritic commits to it.
- Added `tools/evaluate_action_event_head.py` as the first online-action-head
  prototype. It flattens prefix-diff JSON rows into individual candidate-action
  events, trains two bootstrap MLP ensembles (`good` and `risk`), and evaluates
  thresholded acceptance on unique seed/time/agent/action candidates. This is
  closer to the deployment interface than the previous prefix-row rankers
  because it asks: "should this candidate action be allowed now?"
- First held-out event-head run trained on the fair action-conflict blocks
  `2840..2859` and `2880..2919`, then validated on `2920..2959` with
  `--reward-loss-mode non_success`, five split seeds, five ensemble members,
  and thresholds over `min_good_probability` and `max_risk_probability`.
  The data contained 734 train events (`120` good, `530` neutral, `84` bad)
  and 600 validation events (`99` good, `447` neutral, `54` bad) over 133
  numeric features.
- The event head finds signal but is not deployable yet. Example aggregate
  settings:
  `min_good=0.9,max_risk=0.5` accepted 7 unique good, 50 unique neutral, and
  10 unique bad candidates, with unique reward `-0.306849` and Success
  `+0.666667`; `min_good=0.8,max_risk=0.5` accepted 9/62/12 unique
  good/neutral/bad with reward `-0.265936` and Success `+0.666667`;
  `min_good=0.7,max_risk=0.5` accepted 13/74/13 with reward `-0.057172` and
  Success `+1.000000`. The positive Success signal is real, but the bad-action
  leakage is too high for an online guard.
- Current conclusion after the event-head prototype: the right interface is an
  online action-value/risk decision, but supervised labels from prefix
  counterfactuals are still too coarse. The next high-value step should be
  rollout-aware learning: either train this action head with an online RL-style
  objective/TD target, or use it as a critic/risk auxiliary during PPO-style
  training instead of deploying a thresholded event classifier directly.
- Added the first rollout-aware action-conflict PPO hook to
  `tools/train_masked_ppo.py`. With `--use-action-conflict-obs`, PPO can now
  apply opt-in dense penalties to the actually selected action's per-action
  conflict features:
  `--action-conflict-penalty-coef`,
  `--action-head-on-penalty-coef`, and
  `--action-opposing-penalty-coef`. Defaults are `0`, so existing runs are
  unchanged. The trainer also logs selected-action diagnostics:
  `action_conflict_risk`, `action_conflict_mean`, `action_conflict_risky`,
  `action_head_on`, and `action_opposing`.
- Smoke validation:
  direct `--help` now works without manually setting `PYTHONPATH=.`; one
  16-step PPO smoke run with action-conflict penalties saved
  `/private/tmp/ecml_action_conflict_penalty_smoke.pt`; passing an
  action-conflict penalty without the 79-feature observation correctly raises
  a validation error. A synthetic feature check confirmed that selected
  `LEFT`/`FORWARD` action risks are converted into negative reward penalties.
- Current assessment: this is still reward shaping, not a winning policy by
  itself, but it is the lowest-risk bridge from the failed offline event-head
  classifier to actual online RL. The next experiment should train a 79-feature
  PPO candidate on curated hard seeds with these selected-action penalties,
  then compare both final score and the new conflict-action metrics against
  the unpenalized action-conflict PPO baseline.
- First selected-action-penalty PPO candidate:
  `/private/tmp/ecml_action_conflict_penalty_ppo_seed3600_u6.pt` was trained
  from `/private/tmp/ecml_action_conflict_rescue_bc_v1.pt` on known
  `2840..2919` positive/hard-negative action-conflict seeds plus a few anchor
  seeds. It used 6 updates x 4 complete episodes, `lr=1e-5`, teacher CE
  `0.25`, anchor KL `4.0`, rollout temperature `1.08`, terminal team
  success/failure shaping, and action penalties
  `0.02/0.08/0.04` for conflict/head-on/opposing. Training stayed controlled:
  update Success ended at `0.958333`, anchor KL stayed below `0.0006`, and
  `action_conflict_risky` fell from `0.156` in update 1 to `0.0864` in update
  6.
- Held-out raw-policy scoreboard on `2920..2959` against the current packaged
  Sequence-Gate default: current `0.922640 / 0.925000`; raw Action-BC v1
  `0.906950 / 0.945833` (`reward_delta=-0.015690`, Success `+0.020833`);
  penalty-PPO3600 `0.907912 / 0.945833`
  (`reward_delta=-0.014728`, Success `+0.020833`). PPO3600 is a small reward
  improvement over raw Action-BC with unchanged Success, but still not safe
  enough to deploy raw.
- Seed-level diagnosis: PPO3600 improves over Action-BC on `2944` and removes
  BC changes on `2925`; it also removes the BC win on `2958` and worsens
  `2936`. Against current, the best PPO3600 gains are `2935`, `2930`,
  `2934`, `2941`, `2940`, `2945`, and `2949`, but the largest regressions
  remain `2924`, `2922`, `2955`, `2936`, `2937`, `2956`, and `2944`.
- Current assessment after PPO3600: selected-action conflict shaping is useful
  training instrumentation and nudges the raw policy in the intended
  direction, but it does not solve the main bottleneck. The winning path still
  needs either a stronger learned action-value/risk model or an online
  sequence gate that can accept PPO3600's safe wins while blocking the same
  recurring traps.
- Sequence-Gate integration screen for PPO3600: because PPO3600 is a
  79-feature checkpoint, the online gate must run with
  `MyActionConflictObservationBuilder` if this checkpoint is included as a
  candidate. On `2920..2959`, Sequence-Gate with Action-Conflict obs but the
  default candidate list was exactly identical to the packaged current score:
  `0.922640 / 0.925000`. Adding PPO3600 as a sixth candidate raised the score
  to `0.927412 / 0.925000`, with reward wins on seeds `2941` and `2944`, no
  reward losses, and no Success changes.
- Fresh follow-up block `2960..2999`: Sequence-Gate with Action-Conflict obs
  and default candidates scored `0.894669 / 0.908333`; adding PPO3600 scored
  `0.896933 / 0.908333`, again with no regressions. The only changed seed was
  `2964` (`+0.090580` reward, Success unchanged).
- Current assessment after 80 gated validation seeds (`2920..2999`): PPO3600
  is not useful as a raw policy, but it is useful as an additional candidate
  under the existing strict Sequence-Success/Risk gate: 3 reward wins, 0 reward
  losses, 0 Success changes across these two disjoint 40-seed blocks. Next
  promotion step should be broader stress validation before copying the
  checkpoint into `submission/models/` and adding it to the default candidate
  list.
- Broader gated stress validation added two further fresh 40-seed blocks.
  On `3000..3039`, Sequence-Gate with Action-Conflict obs and default
  candidates scored `0.928094 / 0.925000`; adding PPO3600 was exactly
  unchanged. On `3040..3079`, default scored `0.857512 / 0.900000`; adding
  PPO3600 was again unchanged. Aggregated over `2920..3079` (160 episodes),
  PPO3600 under the gate improves reward mean from `0.900729` to `0.902488`
  with identical Success `0.914583`: reward wins/losses/ties `3/0/157`,
  Success `0/0/160`. Changed seeds are only `2941`, `2944`, and `2964`.
- Current promotion decision: PPO3600 has passed the first broad gated
  regression screen. Promotion requires two code/config changes: copy the
  checkpoint into `submission/models/`, add it to the default sequence
  candidate list, and switch the Docker `OBS_BUILDER` from the 64-feature
  trajectory builder to the 79-feature Action-Conflict builder. Existing
  64-feature checkpoints remain compatible because their `ActorCritic`
  instances read only their configured prefix of the longer observation and
  still use the appended 5-action mask.
- PPO3800 follow-up: a second action-conflict PPO checkpoint
  `/private/tmp/ecml_action_conflict_successdiv_penalty_ppo_seed3800_u10.pt`
  was trained from the success-diversity/rescue mix checkpoint. Raw PPO3800 is
  still not deployable on `3080..3119` (`0.886727 / 0.925000` versus promoted
  Sequence default `0.902456 / 0.933333`), but it contains real rescue
  opportunities. Prefix mining on six focus seeds showed `3107` needs a
  two-event PPO prefix for full rescue, `3088` is rescued by the first PPO3800
  right-detour, while `3101` and `3108` contain clear bad prefixes.
- A broad gate relaxation was unsafe. Lowering the Sequence Success threshold
  and right-detour value guard globally opened `3088`
  (`+0.139924` reward, `+0.166667` Success) and was clean on `3080..3119`,
  but failed on the independent `3120..3159` block with reward wins/losses
  `2/3/35` and Success wins/losses `0/2/38`. The regressions came from older
  candidates also benefiting from the global relaxation, not from PPO3800
  itself.
- The promoted-safe PPO3800 variant is therefore candidate-specific: PPO3800
  is added as an extra default candidate, but the relaxed thresholds apply only
  to that checkpoint and only after the normal Sequence gate has already
  accepted one event. The extra candidate is capped at two total accepted
  events per episode. Validation:
  - `3080..3119`: `0.902456 / 0.933333` to `0.905954 / 0.937500`,
    reward and Success wins/losses/ties `1/0/39`.
  - `3120..3159`: unchanged `0.932636 / 0.920833`, wins/losses/ties
    `0/0/40`.
  - `3160..3199`: unchanged `0.906829 / 0.908333`, wins/losses/ties
    `0/0/40`.
- Assessment after PPO3800 candidate-specific gating: this is a safe RL
  deployment improvement, but still not the "large jump." It confirms the
  route to better RL performance: train policies that create more low-risk
  first-extra actions like `3088`, and separately build a sequence-level
  planner for multi-event rescues like `3107`.
- Added `tools/mine_rolling_sequence_dataset.py` as the first explicit
  rolling-horizon sequence candidate generator. Unlike policy-diff mining, it
  does not require a second policy to propose deviations. It scans a baseline
  rollout for critical decisions, scores legal alternative actions by local
  conflict/risk improvement, rejects known unsafe transitions such as
  `MOVE_RIGHT->MOVE_FORWARD`, selects top-scoring events, replays short
  chronological prefixes, and writes sequence rows with final outcome labels.
- First smoke on `3088,3107,3101`: the initial implementation exposed why this
  must stay offline for now. On already-solved `3088`, Prefix 2/3 can regress;
  on `3101`, early alternatives can create a Success loss. On `3107`, however,
  the planner found a useful two-event reward improvement and rediscovered the
  same first key action as PPO3800 (`121:a2 MOVE_FORWARD->MOVE_RIGHT`), showing
  that this is a real multi-step candidate-generation path.
- Failure-focused rolling-sequence mining on ten current failed/partial seeds
  from `3080..3119`
  (`3081,3083,3096,3106,3107,3109,3110,3112,3113,3116`) produced:
  - Prefix 1: reward wins/losses/ties `2/2/6`, Success `1/1/8`,
    mean delta `-0.003817 / 0.000000`.
  - Prefix 2: reward `3/2/5`, Success `1/0/9`, mean delta
    `+0.010310 / +0.016667`.
  - Prefix 3: reward `3/1/6`, Success `2/0/8`, mean delta
    `+0.021346 / +0.033333`.
  Strong positives include seed `3081` (`+0.122288` reward,
  `+0.166667` Success; first event `188:a3 MOVE_LEFT->MOVE_FORWARD`),
  seed `3107` (`+0.072971` reward; first event
  `121:a2 MOVE_FORWARD->MOVE_RIGHT`), and seed `3112` (`+0.038369`
  reward). Hard negatives include seed `3096` Prefix 1/2 and seed `3113`
  Prefix 1, which are exactly the kind of rows needed for a learned
  value/risk scorer.
- Current assessment after the rolling-sequence prototype: the candidate
  generator is promising and more strategic than single-action PPO gating, but
  it is not safe as a hand-written online policy. The next high-value step is
  to scale this mining over broader failure windows and train a sequence
  value/risk model that accepts the `3081/3107/3112`-style positives while
  rejecting `3096/3113`-style traps.
- Scaled rolling-sequence mining to the current failed/partial seed pool from
  `3080..3199`: 19 train-window seeds from `3080..3119` and 36 holdout-window
  seeds from `3120..3199`, each with prefix lengths `1..5`. The holdout raw
  prefix means improved with longer prefixes, but still had meaningful reward
  regressions:
  - Prefix 1: mean delta `-0.007301 / +0.013889`, reward wins/losses/ties
    `4/7/25`, Success `4/1/31`.
  - Prefix 3: mean delta `+0.022928 / +0.027778`, reward `13/6/17`,
    Success `7/1/28`.
  - Prefix 5: mean delta `+0.028771 / +0.041667`, reward `13/9/14`,
    Success `9/0/27`.
  This confirms that the generator finds real rescues, but longer prefixes are
  not automatically safe.
- Oracle check with no-op/baseline available on all 55 mined seeds shows the
  current sequence-generator opportunity size: a perfect selector would accept
  25 seeds and leave 30 seeds unchanged, for about `+2.192` reward delta and
  `+2.000` Success delta. This is large enough to justify more learned
  sequence selection work, but it is not yet a solved RL policy.
- The first listwise group-ranker evaluation exposed a numeric issue in the
  shared sequence feature pipeline: finite sentinel values such as
  `priority_effective_slack=1e9` could appear in validation but not training
  splits, producing huge out-of-distribution standardized values and absurd
  scores (one bad `3096` candidate received a score near `593000`). The
  sequence feature encoder now clips numeric static and event features to
  `[-1000, 1000]` before standardization. This is an offline scorer/training
  robustness fix; it does not change the packaged policy behavior.
- With clipped features, five-split CV over all 55 mined seed groups gives a
  promising but still conservative sequence-selector signal. Full static+event
  features, no uncertainty penalty, margin `-0.25`: accepted `7` good, `11`
  neutral, `0` bad, with `0` Success-negative leaks and aggregate deltas
  `+0.996` reward / `+0.833` Success over the validation folds. Event-only
  scoring is slightly weaker but stable at margin `0.1/0.25`: accepted `8`
  good, `7` neutral, `0` bad, aggregate `+0.816` reward / `+0.667` Success.
- The strict out-of-window test remains weak when training only on the 19
  `3080..3119` seeds and validating on the 36 `3120..3199` seeds. Both the
  full-feature and event-only rankers leak bad candidates. Conclusion: the
  rolling-sequence ranker is a useful learning direction and a better target
  than more hard-coded gate relaxations, but it is not deployment-ready. The
  next large-step RL plan should mine substantially more seed groups, then
  train a sequence-value model or distill the best prefixes into PPO/BC so the
  policy learns the rescues directly instead of relying on a brittle online
  rescue gate.
- Added `--best-prefix-per-seed` to
  `tools/convert_diff_prefix_to_rescue_events.py` so rolling-sequence rows can
  be converted into cleaner BC labels: only the highest-utility positive prefix
  per seed is emitted, rather than every positive intermediate prefix. On the
  55 mined seeds from `3080..3199`, this produced 51 event labels from 25
  positive seeds. The label set is mixed but meaningful: 24 Success-rescue
  events and 27 reward-only rescue events, with actions split across
  `MOVE_LEFT`, `MOVE_FORWARD`, and `MOVE_RIGHT`.
- First rolling-sequence Oracle-BC checkpoint:
  `/private/tmp/ecml_rolling_oracle_bc_v1.pt`. Training used the 79-feature
  Action-Conflict observation, the current `submission.sequence_success_policy`
  as rollout baseline, all 55 mined seeds as low-weight anchors, and the 51
  best-prefix Oracle events as high-weight BC targets. Collection was clean:
  `rescue_hits=51`, `rescue_misses=0`, `rescue_invalid=0`,
  `baseline_mismatches=0`, `anchor_samples=13301`. This is an important
  reproducibility check: the mined Oracle events are reachable from the current
  baseline rollout.
- Raw Oracle-BC is not deployable. On the same 55-seed screen versus current
  Sequence default with Action-Conflict obs, it scored:
  current `0.847169 / 0.830303`, raw BC `0.816776 / 0.830303`;
  reward wins/losses/ties `17/31/7`, Success `14/13/28`. It contains strong
  new rescues, e.g. `3180` (`+0.225254` reward, `+0.333333` Success), `3188`,
  `3194`, `3196`, but also severe regressions such as `3143`, `3092`, `3093`,
  `3193`, and `3146`. The diagnosis is clear: global BC from sparse rescue
  labels learns useful rescue behavior but changes too many unlabelled
  decisions.
- First PPO/BC polish from the raw BC checkpoint:
  `/private/tmp/ecml_rolling_oracle_bc_ppo_v1.pt`. It used 8 PPO updates x 4
  complete episodes, Action-Conflict penalties, terminal team success/failure
  shaping, Teacher CE to the current Sequence policy (`0.8`), and a small KL
  anchor to the BC init (`0.5`). The training remained numerically controlled,
  but did not fix the deployment issue. On the same 55 seeds it scored:
  current `0.847169 / 0.830303`, PPO/BC `0.818167 / 0.830303`;
  reward `19/30/6`, Success `17/14/24`. PPO recovered or improved a few BC
  wins, but preserved too many bad drifts, including a catastrophic `3093`
  Success loss.
- Current PPO/BC conclusion: the data path is now valid and the rescue labels
  are real, but "BC checkpoint first, PPO polish second" is not the right
  architecture yet. The next better variant should keep the current safe policy
  as initialization and add the Oracle rescue labels as an auxiliary BC loss
  during PPO, with explicit negative-baseline labels from harmful prefixes and
  stronger no-op/teacher anchoring. That should let PPO learn rescue actions
  only where the online return supports them, instead of deploying a globally
  drifted BC policy.
- `tools/train_masked_ppo.py` now supports an auxiliary BC loss through
  `--aux-bc-csv` and `--aux-bc-coef`. The implementation reuses the
  event-to-observation collection path from `train_rescue_behavior_clone.py`,
  then mixes a weighted masked cross-entropy loss into each PPO minibatch. It
  supports positive rescue rows and `event_kind=negative_baseline` avoidance
  rows, plus optional low-weight baseline-action anchor samples. The intent is
  to keep PPO online and reward-driven while giving it direct gradients on the
  rare rescue/avoidance states.
- First auxiliary-BC PPO run from the stable actor init:
  `/private/tmp/ecml_aux_bc_currentinit_ppo_v1.pt`. Training used
  `submission/checkpoint.pt` with 79-feature Action-Conflict obs, 8 PPO updates
  x 4 complete episodes, Teacher CE to `submission.sequence_success_policy`
  (`1.2`), KL to the stable actor init (`1.0`), Action-Conflict penalties,
  terminal team shaping, and Aux-BC coefficient `0.45`.
  The Aux-BC CSV combined the 51 best-prefix positive labels with 42 harmful
  prefix negative-baseline labels. Collection hit `82` labels
  (`51` positive plus `31` negative-baseline after mask filtering), skipped
  `11` invalid labels, and had `0` baseline mismatches.
- On the same 55-seed `3080..3199` screen, Aux-BC PPO is a clear improvement
  over raw BC and BC-init PPO, but still not safe enough to promote raw:
  current Sequence default `0.847169 / 0.830303`, Aux-BC PPO
  `0.844261 / 0.854545`; reward wins/losses/ties `21/21/13`, Success
  `16/7/32`. Strong wins include `3196`, `3180`, `3188`, `3107`, `3194`,
  and `3116`. Remaining severe regressions are `3093`, `3143`, `3146`,
  `3109`, `3168`, and `3083` by Success, plus several reward-only losses.
- Current decision: this is the first PPO/BC variant moving in the intended
  direction. It confirms the better architecture: stable-policy init +
  auxiliary rescue/avoidance BC + online PPO. It should not be deployed raw
  yet. The next v2 should overweight the seven Success-loss seeds as
  negative-baseline/teacher anchors, cache Aux-BC samples for faster sweeps,
  and test a lower Aux-BC coefficient or stronger KL/Teacher CE before any
  gated integration.
- `tools/train_masked_ppo.py` now supports `--aux-bc-cache` and
  `--aux-bc-refresh-cache`. Collected Aux-BC observations/actions/weights are
  saved as a torch payload and can be reused without replaying the environment.
  The broad cache `/private/tmp/ecml_rolling_sequence_aux_bc_cache.pt` reloads
  in a few seconds and reproduces the v1 collection stats:
  `samples=13380`, `rescue_hits=82`, `avoidance_hits=31`,
  `rescue_invalid=11`, `baseline_mismatches=0`, `anchor_samples=13298`.
- Focused v2 cache:
  `/private/tmp/ecml_rolling_sequence_aux_bc_v2_focus_cache.pt` kept the same
  82 valid rescue/avoidance labels but added dense baseline anchors on the
  v1 Success-loss seeds `3083,3093,3109,3143,3146,3168,3174`
  (`anchor_samples=27099`). The v2 PPO run used this cache, lower Aux-BC
  coefficient `0.22`, stronger Teacher CE/KL (`2.0/2.0`), lower learning rate
  `7e-6`, and a focused 32-episode schedule mixing loss seeds with strong
  rescue seeds.
- v2 was not an improvement. On the same 55-seed `3080..3199` screen:
  current Sequence default `0.847169 / 0.830303`, v2
  `0.825744 / 0.830303`; reward `19/20/16`, Success `10/7/38`.
  It preserved some wins (`3180`, `3107`, `3196`) but made the key safety
  failures worse or unchanged: `3093`, `3109`, `3143`, `3146`, `3168`,
  plus a new Success loss on `3134`. Conclusion: dense anchors on the loss
  seeds are not sufficient. The next useful improvement should mine exact
  negative action labels from the candidate-vs-current policy diffs on these
  failure seeds, instead of trying to correct them with generic baseline
  anchors.
- Added `tools/convert_policy_diffs_to_negative_events.py` to turn
  baseline-vs-candidate action-diff rows into Aux-BC-compatible
  `event_kind=negative_baseline` rows. This makes exact "candidate chose X,
  current safe policy chose Y" corrections reproducible from
  `tools/analyze_policy_action_diffs.py`. On the v2 regression seeds
  `3083,3093,3109,3134,3143,3146,3168`, the converter produced 47 negative
  labels. The candidate action was mostly `MOVE_FORWARD` (`33/47`), while the
  safe forced labels were balanced between `MOVE_LEFT` (`18`), `MOVE_RIGHT`
  (`17`), and `MOVE_FORWARD` (`12`).
- Aux-BC PPO v3 used those exact policy-diff negative labels plus the previous
  rolling-sequence rescue/avoidance labels. Cache:
  `/private/tmp/ecml_aux_bc_v3_policy_diff_cache.pt` with `17483` samples,
  `100` rescue hits, `49` avoidance hits, `34` invalid labels, `0` baseline
  mismatches, and `17383` low-weight anchors. Training started again from the
  stable actor init, used Teacher CE/KL `2.0/2.0`, LR `8e-6`, and Aux-BC
  coefficient `0.30`. Checkpoint:
  `/private/tmp/ecml_aux_bc_currentinit_ppo_v3.pt`.
- v3 is not the next deployable checkpoint. On the same 55-seed screen:
  current Sequence default `0.847169 / 0.830303`, v3
  `0.838133 / 0.845455`; reward `18/23/14`, Success `10/5/40`. It reduced
  Success losses compared with v1 (`5` vs `7`) and preserved strong rescue
  wins on `3196`, `3180`, `3107`, `3194`, `3116`, and `3188`, but it still
  failed hard on the exact regression seeds we tried to repair:
  `3143`, `3093`, `3146`, `3168`, and `3134`, with large reward-only losses
  on `3109`, `3108`, `3154`, `3115`, and `3126`.
- Current decision: exact pointwise negative policy-diff labels help reduce
  some Success losses, but they do not solve the causal sequence failures.
  The best current RL checkpoint remains v1 for Success lift, but it is still
  not raw-deployable. The next high-value direction should stop adding more
  point labels and instead learn a sequence-level value/risk model or selector:
  evaluate candidate action prefixes over short horizons, learn which prefixes
  improve final team outcome, then distill only those causally validated
  prefixes into PPO/BC or use the model as a learned online reranker.
- Started the next sequence-selector OOD loop on a fresh seed window. Current
  Sequence default on `3200..3239` scored `0.914404 / 0.916667`; 15 partial
  seeds were selected for rolling-prefix mining:
  `3200,3203,3209,3210,3211,3218,3219,3222,3224,3229,3230,3234,3235,3237,3238`.
  Rolling prefixes `1..5` produced 75 rows with useful label diversity:
  `26` good, `34` neutral, `15` bad; Success deltas included eight positive
  rows (`4` at `+0.166667`, `4` at `+0.5`) and one negative row
  (`-0.166667`). The raw oracle signal is again real but unsafe as a rule:
  prefix 2 had mean delta `+0.020280 / +0.044444`, while prefix 5 still had
  one Success loss despite positive mean delta.
- OOD Group-Ranker validation was run with train rows from the old 55 mined
  groups (`3080..3199`) and validation only on the fresh `3200..3239` partial
  block. The previous full-feature target was safe but too conservative:
  with uncertainty penalties and margin `0.25/0.5`, it accepted one good
  reward-only row and five neutral rows, with `0` bad and `0` Success-negative
  leaks, but no Success rescue (`+0.089147 / +0.000000` aggregate reward /
  Success over three training seeds).
- Event-only scoring generalized better on the same OOD block. With the
  original target and margin `-0.25`, it accepted three good reward-only rows
  and four neutral rows with `0` bad and `0` Success-negative leaks
  (`+0.267442 / +0.000000`).
- A success-heavy event-only target is the strongest sequence-selector result
  so far. Configuration: `success_weight=12`, `reward_loss_penalty=1.5`,
  `success_loss_penalty=16`, `failure_penalty=6`, uncertainty coefficients
  `0.5/0.5`, margin `-1.0`. Across split seeds `1,3,5`, it accepted
  `7` good and `8` neutral rows, `0` bad rows, `0` Success-negative leaks,
  and aggregate deltas `+1.283928 / +1.500000`. The key Success rescue
  `3218` prefix 2 (`+0.299376` reward, `+0.5` Success) was accepted in all
  three split models. The consensus accepted rows were stable for
  `3200` prefix 1 (neutral), `3218` prefix 2 (strong Success rescue),
  `3229` prefix 1 (neutral), and `3237` prefix 1 (reward rescue).
- Current decision after the OOD smoke: this supports the architecture shift.
  Sequence-level learned selection is more promising than more pointwise PPO
  labels. It is still not deployable from one 15-seed OOD block; next we need
  either more independent OOD blocks with the success-heavy event-only target,
  or an exported conservative sequence selector that can be audited online
  before any PPO distillation.
- Extended the OOD sequence-selector audit to `3240..3319`. Current Sequence
  default scored `0.909811 / 0.922917` over 80 seeds; 29 partial/failed seeds
  were mined with rolling prefixes `1..5`, producing 145 sequence rows:
  `58` good, `67` neutral, `20` bad. The block is a stronger stress test than
  `3200..3239`: it contains large rescues such as `3297` prefix 1
  (`+0.467613` reward, `+0.5` Success), `3300` prefix 1 (`+0.333333 /
  +0.333333`), and `3311` prefix 1 (`+0.122129 / +0.166667`), but also hard
  Success-negative traps such as `3265` prefix 5 and `3302` prefixes 1..5.
- Raw rolling-prefix means on `3240..3319` again show real opportunity but
  unsafe direct deployment. Prefix 1 averaged `+0.016628 / +0.051724` with
  Success wins/losses/ties `7/1/21`; prefix 2 averaged `+0.015569 /
  +0.063218` with Success `9/1/19`; longer prefixes kept positive Success
  means but accumulated more Success losses, reaching `8/4/17` at prefix 5.
- Success-heavy event-only Group-Ranker, trained only on the old 55 mined
  groups and validated on `3240..3319`, is safe only at conservative margins.
  At margin `-0.25` it accepted `1` good and `7` neutral rows with `0` bad and
  `0` Success-negative leaks, aggregate `+0.052101 / +0.166667`. At margin
  `-1.0` it recovered more Success (`4` good, `14` neutral) but leaked one bad
  reward-only candidate, aggregate `-0.060374 / +0.666667`. This is not
  deployable without a veto.
- Adding the previous OOD block (`3200..3239`) to training made the same
  event-only selector safer but too conservative on `3240..3319`: at margin
  `-1.0` it accepted `1` good and `8` neutral rows with `0` bad and
  `0` Success-negative leaks, aggregate `+0.052101 / +0.166667`; stricter
  margins accepted only neutral rows. More OOD data therefore helps safety,
  but does not by itself unlock recall.
- Split-consensus is currently the safest learned-selection rule. With the old
  55-group training set and margin `-1.0`, candidates accepted by all three
  split models were clean on both OOD blocks: on `3200..3239`, 4 unique
  candidates, `0` bad, `0` Success-negative, `+0.388524 / +0.500000`; on
  `3240..3319`, 4 unique candidates, `0` bad, `0` Success-negative,
  `+0.052101 / +0.166667`. This is promising but still low recall.
- A standalone event-level Good/Risk head trained on `3080..3239` and
  validated on `3240..3319` is not safe as an acceptor: even strict settings
  leaked multiple bad and Success-negative unique events. Conclusion: the next
  high-value engineering step is not another PPO/BC run and not just lower
  ranker margins. We need a prefix-level risk/veto model, preferably trained
  listwise at the same granularity as the Group-Ranker, so that we can keep
  the high-recall `-1.0` ranker while rejecting traps like `3265` and `3302`.
- Reused the existing prefix-level sequence Success/Unsafe model in
  `tools/evaluate_sequence_rescue_planner.py` as the first real prefix-risk
  audit. Training on `3080..3239` and validating on `3240..3319`, with
  `reward_negative_unsafe_mode=non_success`, produced safe threshold metrics
  at low recall and a stronger planner-style score. The raw planner metric at
  `unsafe_weight=0.5`, `score_threshold=0.25` accepted `4` good and `10`
  neutral rows across split models, `0` bad, `0` Success-negative, aggregate
  `+0.407831 / +0.500000`.
- Added `tools/summarize_prefix_planner_deployment.py` to avoid over-counting
  split-model rows or multiple prefix lengths for the same seed. It scores
  prefix candidates per split, requires split consensus, and then keeps at
  most one prefix per seed. Under this deployment-style metric on
  `3240..3319`, the best safe prefix-planner settings reached:
  `10` unique seeds accepted, `3` good and `7` neutral, `0` bad,
  `0` Success-negative, aggregate `+0.285702 / +0.333333`
  (`unsafe_weight=0.25`, `score_threshold=0.2`, `consensus_min_splits=1`).
  A more conservative consensus-2 setting accepted `7` seeds, `2` good and
  `5` neutral, also safe, aggregate `+0.167666 / +0.333333`.
- The ranker+prefix-veto combination is safe but lower recall on this block.
  `tools/summarize_ranker_veto_deployment.py` now accepts both
  `risk_probability_ucb` and prefix-model `unsafe_probability_ucb` columns.
  With Group-Ranker margin `-1.0` plus prefix unsafe veto, the deployment-style
  unique-candidate metric on `3240..3319` kept `0` bad and `0`
  Success-negative rows but only one useful Success rescue (`+0.052101 /
  +0.166667`). Current decision: the prefix-level planner/risk model is the
  better next online candidate than the old Group-Ranker gate. Next steps are
  to validate this deployment-style prefix planner on at least one more OOD
  block and then export a conservative online prefix selector.
- Validated the prefix planner on a second independent OOD block,
  `3320..3399`. The current Sequence default scored `0.906810 / 0.912500`
  over 80 seeds; 31 partial seeds were mined with rolling prefixes `1..5`,
  producing 155 rows: `59` good, `80` neutral, `16` bad. The block contains
  both useful rescues and real traps: `3386` prefixes rescue a `0.5 / 0.5`
  baseline to `1.0 / 1.0`, while `3390` and `3399` include Success-negative
  prefixes and `3324` includes reward-only bad prefixes.
- The previous prefix planner settings did not remain safe under the earlier
  loose deployment threshold grid. With thresholds up to `0.5`, every
  deployment-style configuration leaked at least one bad candidate on
  `3320..3399`. Raising the planner threshold and requiring full split
  consensus recovered safe behavior: `unsafe_weight=0.25`,
  `score_threshold=0.9`, `consensus_min_splits=3` accepted 3 unique seeds,
  `1` good and `2` neutral, `0` bad, `0` Success-negative, aggregate
  `+0.500000 / +0.500000`. The stricter `score_threshold=0.95` accepted only
  the strong `3386` rescue, still `0` bad and `0` Success-negative, aggregate
  `+0.500000 / +0.500000`.
- Added an optional reward-risk head to
  `tools/evaluate_sequence_rescue_planner.py` and corresponding
  `planner_reward_risk_weight` support in
  `tools/summarize_prefix_planner_deployment.py`. This head learns
  reward-negative prefix risk separately from the existing Success and Unsafe
  heads. On this OOD block it did not yet improve the best safe recall over
  the high-threshold consensus rule, so the current evidence says the main
  deployable improvement is stricter calibrated selection, not the new head.
  The next experiment should use the reward-risk head as a recall recovery
  tool: keep the safe high-threshold consensus baseline, then search for lower
  thresholds that stay clean only when reward-risk is weighted.
- Added `tools/compare_prefix_planner_deployment.py` to compare shared
  deployment configs across OOD blocks. The low-threshold overlap between
  `3240..3319` and `3320..3399` has no safe common config: every shared config
  up to `score_threshold=0.5` leaked at least one bad candidate, despite much
  better combined recall (`+1.333333` Success at the top but with `1` bad
  leak). The high-threshold overlap has 10 safe common configs. The safest
  useful family requires `consensus_min_splits=3` and `score_threshold>=0.9`;
  for example `unsafe_weight=0.25`, `score_threshold=0.9` accepted 3 unique
  candidates across the two blocks, `1` good and `2` neutral, `0` bad,
  `0` Success-negative, aggregate `+0.500000 / +0.500000`. The purer
  deployment candidate is `score_threshold=0.95`, which accepts only the
  strong `3386` rescue and no neutral interventions, with the same aggregate
  lift.
- Current decision: this is safe but not yet a winning selector. We should not
  export this as the final solution and declare victory; it is a conservative
  fallback. The next high-value work is recall recovery without bad leaks:
  improve the learned prefix objective from "Success-positive minus risk" to a
  calibrated utility/risk objective, or train a listwise selector that directly
  optimizes "best safe prefix per seed" instead of scoring rows independently.
- Revisited the existing listwise tool `tools/evaluate_group_rescue_ranker.py`
  with a success-heavy utility target and explicit baseline/no-op candidate.
  This is closer to the desired selector than the rowwise prefix planner: for
  each seed group it learns which prefix, if any, should beat no-op. Training
  protocol stayed OOD: `3240..3319` was validated with training rows through
  `3200..3239`, and `3320..3399` was validated with training rows through
  `3240..3319`.
- On `3240..3319`, the listwise ranker remained conservative but safe:
  deployment-style consensus/no-veto at margin `-1.0`, `consensus_min_splits=1`
  accepted 10 unique candidates, `1` good and `9` neutral, `0` bad,
  `0` Success-negative, aggregate `+0.052101 / +0.166667`. This confirms
  the hard part of this block is recall, not veto calibration.
- On `3320..3399`, the same listwise setup generalized much better. With
  deployment-style consensus/no-veto at margin `0.1` or `0.2` and
  `consensus_min_splits=1`, it accepted 17 unique candidates, `6` good and
  `11` neutral, `0` bad, `0` Success-negative, aggregate
  `+0.709765 / +1.000000`. A light prefix-risk veto at max risk `1.0` did not
  change the top result.
- Added `tools/compare_ranker_deployment.py` to compare ranker deployment
  summaries across OOD blocks. The best shared safe listwise config across
  `3240..3319` and `3320..3399` is margin `0.1` or `0.2`, max risk `1.0`
  or effectively no veto, `consensus_min_splits=1`: 18 unique candidates,
  `6` good and `12` neutral, `0` bad, `0` Success-negative, aggregate
  `+0.709765 / +1.000000`. This doubles the safe Success lift of the
  high-threshold prefix-planner fallback (`+0.500000 / +0.500000`).
- Updated decision: the listwise Group-Ranker is now the better next online
  candidate than the rowwise prefix planner. It is still not a complete
  winning solution because one OOD block contributes almost all of the lift,
  but it is the strongest learned selector so far under zero-leak constraints.
  Next step should be an online export of the listwise selector behind a
  conservative fallback, followed by distilling its accepted prefixes into the
  RL/BC policy.
- Exported the listwise selector for online use. `tools/evaluate_group_rescue_ranker.py`
  can now write a submission checkpoint with static/event feature schema,
  normalization statistics, and the full `SequenceRescueNet` ensemble via
  `--export-model`. The exported checkpoint is
  `submission/models/ecml_group_listwise_successheavy.pt`: 15 ensemble members,
  `8` static features, `164` event features, `max_events=5`, trained on all
  currently mined rolling-prefix rows through `3320..3399` (`780` rows,
  `130` seed groups).
- `submission.sequence_success_policy.SequenceSuccessPolicy` now loads this
  checkpoint by default when present and uses it as the primary selector in
  `ECML_SEQUENCE_SELECTOR_MODE=listwise`. The runtime scorer constructs the
  same prefix row used during mining from already accepted events plus the
  current candidate diff, scores it against a synthetic no-op baseline, and
  accepts only when the listwise margin clears
  `ECML_SEQUENCE_LISTWISE_MARGIN_THRESHOLD` (default `0.2`). The old
  Value/Risk sequence scorer remains available as fallback by changing
  `ECML_SEQUENCE_SELECTOR_MODE`.
- Runtime smoke test with the online listwise selector on seeds `3320..3324`
  completed without errors: mean `reward=0.896815`, mean `success=0.933333`
  over 5 episodes. This is only a load/runtime sanity check, not a statistically
  meaningful benchmark.
