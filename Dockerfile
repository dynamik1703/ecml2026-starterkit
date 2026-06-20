# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=v4.2.5
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}


COPY submission/ submission/
COPY tools/ tools/

ENV POLICY=submission.risk_veto_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyActionConflictObservationBuilder
ENV ECML_RISK_VETO_CANDIDATE_CHECKPOINT=submission/models/ecml_ppo_actionobs_prefixrelax_ft_v1.pt
ENV ECML_RISK_VETO_RISK_CHECKPOINT=submission/models/ecml_risk_head_v4_mc6270_6280_s123_val4.pt
ENV ECML_RISK_VETO_REWARD_RISK_CHECKPOINT=submission/models/ecml_reward_risk_head_v4_lowreward09_mc6270_6280_s123_val4.pt
ENV ECML_RISK_VETO_VALUE_CHECKPOINT=submission/models/ecml_action_value_head_v4_mc6270_6280_s123_val4.pt
ENV ECML_RISK_VETO_PREFIX_RELAX_ENABLED=1
ENV ECML_RISK_VETO_MAX_RIGHT_DISTANCE_DELTA=150
ENV ECML_RISK_VETO_STOP_LEFT_MIN_SLACK_FOR_UNCONFLICTED=80
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/xdg-cache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

# install requirements in env activated in entrypoint
RUN bash entrypoint_generic.sh python -m pip install -r submission/requirements.txt
