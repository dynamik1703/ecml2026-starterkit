# Sequence Dispatch BC/RL Plan

## Goal

Move from local action RL toward an LLM-style high-level sequence model:

1. A strong OR/MAPF/SIPP teacher produces safe dispatch decisions.
2. A supervised sequence model imitates global priority/timing decisions.
3. The trained model is used only as a priority bias inside DLA/SIPP.
4. RL fine-tuning later operates on priorities/locks, not raw Flatland actions.

## V1 Implemented

- `submission/sequence_dispatcher.py`
  - Candidate token feature extraction.
  - JSONL sequence example construction.
  - Small Transformer encoder for candidate priority scoring.
  - Optional `SequenceDispatchRanker` runtime wrapper.
- `tools/collect_sequence_dispatch_dataset.py`
  - Rolls out an existing MAPF/SIPP policy.
  - Logs one variable-length example per planning step.
  - Target is the teacher ordering from the current safe `_priority_key`.
- `tools/train_sequence_dispatcher.py`
  - Trains a Transformer with listwise cross-entropy over candidate order.
  - Splits by seed to avoid rollout leakage.
  - Exports a `.pt` checkpoint compatible with the runtime wrapper.
- `submission/rl_mapf_sipp_policy.py`
  - Supports optional sequence dispatch scores.
  - Default is disabled via `ECML_SEQUENCE_DISPATCH_ENABLED=0`.

## Data Schema

Each JSONL row is one MAPF/SIPP planning step:

- `handles`: agent handles for the candidate tokens.
- `features`: shape `[num_candidates, feature_dim]`.
- `teacher_ranks`: rank per token, aligned with `handles`.
- `teacher_scores`: normalized teacher priority score per token.
- `teacher_order`: handles sorted by the teacher priority.
- Metadata: `seed`, `step`, `scene`, `num_agents`, `max_steps`.

## Training Command

```bash
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/ecml_mpl \
  .venv/bin/python tools/collect_sequence_dispatch_dataset.py \
  --policy submission.rl_mapf_sipp_policy.RLMAPFSIPPPolicy \
  --scene scene_4 --num-agents 80 --episodes 6 --seed 7410 \
  --output-jsonl /private/tmp/ecml_sequence_scene4.jsonl

PYTHONPATH=. .venv/bin/python tools/train_sequence_dispatcher.py \
  --input-jsonl /private/tmp/ecml_sequence_scene4.jsonl \
  --output /private/tmp/ecml_sequence_dispatcher_scene4.pt \
  --epochs 15 --batch-size 64 --hidden-size 96
```

## Deployment Gate

Use only after local matrix validation:

```bash
ECML_SEQUENCE_DISPATCH_ENABLED=1
ECML_SEQUENCE_DISPATCH_MODEL=submission/models/ecml_sequence_dispatch_transformer_v1.pt
ECML_SEQUENCE_DISPATCH_SCENES=scene_4
ECML_SEQUENCE_DISPATCH_MIN_AGENTS=70
ECML_SEQUENCE_DISPATCH_WEIGHT=0.5
```

## Next Steps

1. Train scene-specific sequence models for `scene_3` and `scene_4`.
2. Compare against the current pairwise `scene_4` ranker on seeds `7410-7415`.
3. Add DAgger:
   - roll out sequence model,
   - collect failure/low-completion states,
   - relabel with MAPF/SIPP/lock teacher,
   - retrain.
4. Add RL fine-tuning:
   - action space: priority offsets or conflict-group hold/release,
   - low-level executor: unchanged DLA/SIPP,
   - KL regularization to BC model,
   - reward dominated by completion and deadlock avoidance.

## Non-goals For V1

- No raw `LEFT/FORWARD/STOP` action learning.
- No replacement of DLA/SIPP safety.
- No default activation in Docker/submission until local gates are passed.
