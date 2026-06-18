# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=v4.2.5
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}


COPY submission/ submission/
COPY tools/ tools/

ENV POLICY=submission.rerank_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/xdg-cache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

# install requirements in env activated in entrypoint
RUN bash entrypoint_generic.sh python -m pip install -r submission/requirements.txt
