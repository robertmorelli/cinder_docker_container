FROM --platform=linux/amd64 fedora:40 AS build

# DEP STUFF - Fedora packages for building Python and cinderx
RUN dnf install -y \
    gcc \
    gcc-c++ \
    make \
    ccache \
    curl \
    cmake \
    gdb \
    git \
    lcov \
    bzip2-devel \
    libffi-devel \
    gdbm-devel \
    xz-devel \
    ncurses-devel \
    readline-devel \
    sqlite-devel \
    openssl-devel \
    tk-devel \
    libuuid-devel \
    xorg-x11-server-Xvfb \
    zlib-devel \
    python3 \
    python3-pip \
    wget \
    which \
    expat-devel \
    && dnf clean all

# copy the source into the container
COPY --chmod=0755 cinder/ /cinder/
COPY --chmod=0755 static-python-perf/ /root/static-python-perf/

# CINDER STUFF - Build Python
WORKDIR /cinder
RUN ./configure \
    CFLAGS="-Wno-error -Wno-error=strict-prototypes" \
    CXXFLAGS="-Wno-error" \
    && make -j$(nproc) CFLAGS="-Wno-error -Wno-error=strict-prototypes"

# Build cinderx extension using the official build script
WORKDIR /cinder/cinderx
RUN ./build.sh --build-root /cinder --python-bin /cinder/python --output-dir /cinder

# Verify the extension was built
RUN ls -la /cinder/*.so || echo "Extension files:" && find /cinder -name "_cinderx*.so" -o -name "_static*.so" 2>/dev/null | head -20

# Set up environment - use the in-tree python binary
ENV PATH="/cinder:${PATH}"
ENV PYTHONPATH="/cinder"

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
