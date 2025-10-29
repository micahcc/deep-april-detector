FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y \
    vim \
    curl \
    git-all \
    git

RUN conda install -y \
    opencv \
    pillow \
    tqdm \
    ipdb
