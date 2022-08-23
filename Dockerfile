FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y \
        tzdata git wget cmake make zip curl unzip\
        pkg-config build-essential \
        libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev \ 
        libxmlsec1-dev libffi-dev liblzma-dev \
        autoconf libtool flex bison \
        libgl1-mesa-glx rsync libedit-dev htop \
    && apt-get clean

ENV HOME="/root"
WORKDIR ${HOME}
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

ENV PYTHON_VERSION=3.9.6
WORKDIR /home
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

RUN pip3 install virtualenv jupyterlab==3.3.2
CMD python3 -m jupyter lab --allow-root --ip='*' --NotebookApp.token='' --NotebookApp.password=''