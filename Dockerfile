FROM --platform=linux/amd64 ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

# DEP STUFF
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
    python3-pip \
    wget \
    software-properties-common \
    gnupg

# copy the source into the container
COPY --chmod=0755 cinder/ /cinder/
COPY --chmod=0755 static-python-perf/ /root/static-python-perf/

# CINDER STUFF
WORKDIR /cinder
RUN ./configure --enable-optimizations CFLAGS="-Wno-error=maybe-uninitialized" && make -j4
ENV PATH="/cinder:${PATH}"
COPY de_typer.py /cinder/Tools/benchmarks/de_typer.py

# DEV STUFF

WORKDIR /root

# helix
RUN curl -LO https://github.com/helix-editor/helix/releases/download/25.01/helix-25.01-x86_64-linux.tar.xz && \
    tar xf helix-25.01-x86_64-linux.tar.xz && \
    mv helix-25.01-x86_64-linux /opt/helix && \
    ln -s /opt/helix/hx /usr/local/bin/hx && \
    rm helix-25.01-x86_64-linux.tar.xz

# vs code
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && \
    tar -xf vscode_cli.tar.gz && \
    mv code /usr/local/bin/code && \
    rm vscode_cli.tar.gz

# zig 0.15
RUN curl -LO https://ziglang.org/download/0.15.2/zig-x86_64-linux-0.15.2.tar.xz && \
    tar xf zig-x86_64-linux-0.15.2.tar.xz && \
    mv zig-x86_64-linux-0.15.2 /opt/zig && \
    ln -s /opt/zig/zig /usr/local/bin/zig && \
    rm zig-x86_64-linux-0.15.2.tar.xz

# poop
RUN git clone https://github.com/andrewrk/poop.git && cd poop && zig build
ENV PATH="/root/poop/zig-out/bin:${PATH}"

# dotfiles
RUN sh -c "$(curl -fsLS get.chezmoi.io)" -- -b /usr/local/bin
ARG DOTFILES_REPO="robertmorelli/dotfiles"
RUN chezmoi init --apply "$DOTFILES_REPO"

# ENTRY
CMD ["/bin/bash"]
