FROM --platform=linux/amd64 ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
    build-essential \
    ccache \
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

COPY --chmod=0755 cinder/ /cinder/

# setup jit compiled python
WORKDIR /cinder
RUN ./configure --with-pydebug --enable-optimizations && \
    make -j4 && \
    make pythoninfo

# setup cinderx for __static__ and shiz
RUN ln -sf /cinder/python /usr/bin/python3 && \
    cp /cinder/pyconfig.h /cinder/Include/pyconfig.h && \
    cd / && \
    git clone https://github.com/facebookincubator/cinderx.git && \
    cd /cinderx && \
    python3 -m ensurepip && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 setup.py install

WORKDIR /root

# clone benchmarks
RUN git clone https://github.com/utahplt/static-python-perf.git

# setup helix
RUN curl -LO https://github.com/helix-editor/helix/releases/download/25.01/helix-25.01-x86_64-linux.tar.xz && \
    tar xf helix-25.01-x86_64-linux.tar.xz && \
    mv helix-25.01-x86_64-linux /opt/helix && \
    ln -s /opt/helix/hx /usr/local/bin/hx && \
    rm helix-25.01-x86_64-linux.tar.xz && \
    git clone https://github.com/robertmorelli/helix-setup.git && \
    cd helix-setup && \
    chmod +x install.sh && \
    bash ./install.sh

# Set Python path
ENV PATH="/cinder:${PATH}"

CMD ["/bin/bash"]
