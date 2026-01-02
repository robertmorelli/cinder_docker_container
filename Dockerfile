FROM --platform=linux/amd64 ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies based on Cinder's official .github/workflows/posix-deps-apt.sh
# python3 is required for building the JIT (Tools/jit/build.py)
# clang-19 and llvm-19 are required for JIT compilation
RUN apt-get update && apt-get install -yq \
    build-essential \
    ccache \
    gdb \
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

# Install LLVM 19 / Clang 19 for JIT support
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19 && \
    rm llvm.sh

# Copy Cinder source code
COPY --chmod=0755 cinder/ /cinder/

# Build Cinder based on official build.yml workflow
# NOTE: --enable-experimental-jit causes build failures with clang-19
# The JIT infrastructure is experimental and has issues on linux/amd64
# All -X jit flags are accepted even without --enable-experimental-jit
WORKDIR /cinder
RUN ./configure --with-pydebug && \
    make -j4 && \
    make pythoninfo

# Install CinderX for Static Python support (build from source)
# The PyPI package only supports Python 3.12, but we can build from source for 3.14
# Create symlink so CMake finds our Python 3.14 and copy pyconfig.h where CMake expects it
RUN apt-get install -yq git cmake && \
    ln -sf /cinder/python /usr/bin/python3 && \
    cp /cinder/pyconfig.h /cinder/Include/pyconfig.h && \
    cd / && \
    git clone https://github.com/facebookincubator/cinderx.git && \
    cd /cinderx && \
    python3 -m ensurepip && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 setup.py install


WORKDIR /root
RUN apt-get update && apt-get install -yq curl
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

# Set Python path
ENV PATH="/cinder:${PATH}"

CMD ["/bin/bash"]
