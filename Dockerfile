FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt update -q \
 && apt install -y -qq curl wget vim \
 && apt clean -q

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY deepeval_evals/ ./deepeval_evals/
COPY evals-cymraeg/ ./evals-cymraeg/
