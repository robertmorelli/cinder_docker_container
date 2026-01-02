FROM --platform=linux/amd64 ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    curl \
    libbz2-1.0 \
    libffi8 \
    libgdbm6 \
    liblzma5 \
    libncurses6 \
    libreadline8 \
    libsqlite3-0 \
    libssl3 \
    libuuid1 \
    zlib1g \
    git \
    build-essential \
    g++ \
    ccache \
    gdb \
    lcov \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    lzma \
    tk-dev \
    uuid-dev \
    zlib1g-dev \
    xz-utils

COPY --chmod=0755 cinder/ /cinder/src/
# build (out-of-tree)
WORKDIR /cinder/build
RUN /cinder/src/configure --prefix=/cinder && \
    make clean && \
    make -j8 && \
    make install

# install tools
WORKDIR /root
RUN git clone https://github.com/utahplt/static-python-perf.git
RUN curl -LO https://github.com/helix-editor/helix/releases/download/25.01/helix-25.01-x86_64-linux.tar.xz && \
    tar xf helix-25.01-x86_64-linux.tar.xz && \
    mv helix-25.01-x86_64-linux /opt/helix && \
    ln -s /opt/helix/hx /usr/local/bin/hx && \
    rm helix-25.01-x86_64-linux.tar.xz

RUN git clone https://github.com/robertmorelli/helix-setup.git && \
    cd helix-setup && \
    chmod +x install.sh && \
    bash ./install.sh

ENV PATH="/cinder/bin:$PATH"
CMD ["/bin/bash"]
