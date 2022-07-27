# End of parser directives (must precede any comments)
# =====================================================================
# Sale prediction Docker Image @ Vize
# -----------------------------------
#
# Build Image:
# ------------
#
#     >>> docker build -t training/tests:v1 -f Dockerfile .
#
# Run Image:
# ----------
#     --rm ~ cleanup on container or daemon exit; -p (--publish) exposes port; -it ~ stdin open and pseudo-tty
#     >>> docker run -it --rm --gpus all --shm-size=1g -p 9001:8080 -v "$(pwd)/data":/app/data {REPOSITORY}:{TAG} python3.8 master.py
#     >>> docker run -it --rm --gpus all --shm-size=1g -p 9001:8080 -v "$(pwd)/data":/app/data {REPOSITORY}:{TAG} python3.8 inference.py
#
# =====================================================================

FROM nvidia/cuda:11.6.0-base-ubuntu20.04
LABEL authors="Biano AI <ai-research@biano.com>"

# ----------------------------------------------------------------------------------------------------
# 1. System Settings
# ----------------------------------------------------------------------------------------------------
ARG PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-EeuxoC", "pipefail", "-c"]
# ---------------------------------------------------------------------
# 2. Workdir
# ---------------------------------------------------------------------
RUN mkdir -p /app
WORKDIR /app

# ---------------------------------------------------------------------
# 3. Requirements
# ---------------------------------------------------------------------

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN python3 -m pip install --upgrade pip
COPY ./requirements/training.txt /requirements/
RUN python3 -m pip install torch --pre --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3 -m pip install --isolated --no-input --compile --exists-action=a --disable-pip-version-check --no-cache-dir -r /requirements/training.txt \
    && rm -rf /requirements/training.txt

# ---------------------------------------------------------------------
# 4. App sources
# ---------------------------------------------------------------------

COPY ./queries /app/queries
COPY ./src /app/src
COPY ./biano-1152.json /app/biano-1152.json
COPY preprocess.py /app/preprocess.py
COPY train.py /app/train.py
COPY validate.py /app/validate.py
COPY artificial_validation.py /app/artificial_validation.py
COPY src/DAQ.py /app/DAQ.py
