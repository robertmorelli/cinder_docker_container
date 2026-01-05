FROM --platform=linux/amd64 ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
    build-essential \
    ccache \
    chezmoi \
    curl \
    cmake \
    gdb \
    git \
    lcov \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    lzma \
    lzma-dev \
    tk-dev \
    uuid-dev \
    xvfb \
    zlib1g-dev \
    python3 \
    wget \
    software-properties-common \
    gnupg

# copy the source into the container
COPY --chmod=0755 cinder/ /cinder/
COPY --chmod=0755 cinderx/ /cinderx/
COPY --chmod=0755 static-python-perf/ /root/static-python-perf/

# setup jit compiled python
WORKDIR /cinder
RUN ./configure --with-pydebug --enable-optimizations && \
    make -j4 && \
    make pythoninfo

# setup cinderx for __static__ and shiz
RUN ln -sf /cinder/python /usr/bin/python3 && \
    cp /cinder/pyconfig.h /cinder/Include/pyconfig.h && \
    cd /cinderx && \
    python3 -m ensurepip && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 setup.py install

WORKDIR /root

# setup helix
RUN curl -LO https://github.com/helix-editor/helix/releases/download/25.01/helix-25.01-x86_64-linux.tar.xz && \
    tar xf helix-25.01-x86_64-linux.tar.xz && \
    mv helix-25.01-x86_64-linux /opt/helix && \
    ln -s /opt/helix/hx /usr/local/bin/hx && \
    rm helix-25.01-x86_64-linux.tar.xz

# setup code
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && \
    tar -xf vscode_cli.tar.gz && \
    mv code /usr/local/bin/code && \
    rm vscode_cli.tar.gz

# setup dotfiles
ARG DOTFILES_REPO="robertmorelli/dotfiles"
RUN chezmoi init --apply "$DOTFILES_REPO"

# set python path
ENV PATH="/cinder:${PATH}"

CMD ["/bin/bash"]
