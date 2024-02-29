FROM hub.cerit.io/xrosinec/python:3.11-datascience

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libgomp1

RUN pip install --upgrade pip
RUN pip install --no-cache-dir biopython torch nvidia-cudnn-cu11 && rm -rf ~/.cache

COPY . /scripts
WORKDIR /scripts

COPY --chown=1001:1001 / /
RUN chmod 777 /scripts/3d-af-surfer/tmp

USER 1001

