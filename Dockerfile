# syntax=docker/dockerfile:1.7
# jhu_ta1 evaluation image: the environment the JHU DKPS cards run in.
#
# Kitware evaluates a card by turning its `kwdagger:` block into a DAG and
# running every node of that DAG as one `docker run` of this image, with the
# checkout bind-mounted at its own absolute path and the node's cwd there. The
# copy of this repo baked below supplies the installed environment; the mount
# supplies the code that actually runs. See docs/containerized_evaluation.md.
#
#   docker build -t jhu-magnet-dkps-gpu .
#
# MAGNET_REF is the aiq-magnet commit Kitware evaluates against. It is
# published on AIQ-Kitware/aiq-magnet; until it lands on main,
# `--build-arg MAGNET_REF=main` builds against the public main instead. The
# materialize_lite node in the pipelines needs
# magnet.backends.helm.cli.materialize_helm_suite, which is newer than main.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG MAGNET_REF=0ce80c623a15516f719d06d95c84118d3d71de0f

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        git \
        jq \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade uv

WORKDIR /opt/src

# MAGNET first, so the heavy layer is stable across edits to this repo. The
# [optional] extra (gcsfs) is what lets the materialize node download HELM
# lite runs from the public bucket when no precomputed root is given.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "aiq-magnet[optional] @ git+https://github.com/AIQ-Kitware/aiq-magnet@${MAGNET_REF}"

# Runtime dependencies, mirroring pyproject.toml minus aiq-magnet. dkps is the
# DKPS library; sentence-transformers and einops back the local nomic embedder
# used by the text-embedding datasets.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        'dkps @ git+https://github.com/jataware/dkps@magnet' \
        'httpx>=0.28.1,<0.29' \
        'sentence-transformers>=3.1' \
        'einops>=0.8'

# This repo, without dependencies so it uses the magnet pinned above rather
# than re-resolving pyproject's git requirement.
COPY . /opt/src/jhu-magnet-dkps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e /opt/src/jhu-magnet-dkps

WORKDIR /opt/src/jhu-magnet-dkps
CMD ["bash"]
