# Stage 1: Build Environment
FROM --platform=linux/amd64 ubuntu:24.04 AS build

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    git \
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

# Build arguments for make parallelism
ARG make_jobs
ARG make_verbose=1
ENV MAKE_JOBS="${make_jobs}" MAKE_VERBOSE="${make_verbose}"

WORKDIR /cinder/build

# Copy source code
COPY --chmod=0755 cinder/ /cinder/src/

# Clean the source directory to ensure out-of-tree build works
# This removes any build artifacts that may have been copied from the host
RUN cd /cinder/src && (make -C /cinder/src distclean 2>/dev/null || true)

# fix cinder
# RUN sed -i "s/libraries=\['dl'\]/libraries=['dl', 'z']/" /cinder/src/cinderx/setup.py

# Configure and build Cinder Python
RUN \
    /cinder/src/configure --prefix=/cinder --enable-cinderx-module && \
    make -j${MAKE_JOBS:-$(nproc)} VERBOSE=$MAKE_VERBOSE


# Stage 2: Install
FROM --platform=linux/amd64 build AS install

WORKDIR /cinder/build

# Install Python to /cinder
RUN sed -i '325a\            libraries=["z"],' /cinder/src/cinderx/setup.py && make install

# Stage 3: Runtime
FROM --platform=linux/amd64 ubuntu:24.04 AS runtime

# Install only runtime dependencies (shared libraries needed by Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    git


RUN apt-get update && apt-get install -y \
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

# Copy the installed Cinder Python from the install stage
COPY --from=install /cinder /cinder

# Add Python to PATH
ENV PATH="/cinder/bin:$PATH"

WORKDIR /workspace

CMD ["/bin/bash"]

