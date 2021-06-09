ARG BASE_CONTAINER=pytorch/pytorch

FROM $BASE_CONTAINER

LABEL maintainer="Unsupervised CFL <rene.glitza@rub.de>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    wget \
    gdebi \
    htop \
    curl \
    jq \
    net-tools \
    git \
    sudo

# Add user here if you want to save as non-root using nvidia-docker
# ARG USER=user
# ARG USER=root
# ARG USER_ID=1000
# ARG GROUP_ID=1000
# RUN addgroup --gid $GROUP_ID $USER
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
# RUN adduser $USER sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY torch.yml .
RUN conda env update --name base --file torch.yml

# USER $USER

ADD . /workspace

EXPOSE 6007
WORKDIR /workspace

# Commands to run TensorBoard (not used) and ucfl
# CMD tensorboard --logdir=/workspace/runs --host "0.0.0.0" --port 6007 &
# CMD cd /workspace && python3 unsupervised_clustered_federated_learning.py

# For building and running:
# docker build . -t jearde/ucfl:latest
# docker run -it -d --name ucfl --user $(id -u):$(id -g) -v $(pwd):/workspace --ipc=host --net=host jearde/ucfl:latest
# docker exec -it ucfl bash