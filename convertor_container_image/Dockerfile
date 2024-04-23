FROM hub.cerit.io/xrosinec/python:3.11-datascience
USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1001:1001 / /scripts
WORKDIR /scripts
USER 1001
