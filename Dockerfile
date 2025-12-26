# build for x86
FROM --platform=linux/amd64 ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

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

COPY --chmod=0755 cinder/ /cinder/src/

RUN cd /cinder/src && (make -C /cinder/src distclean 2>/dev/null || true)

# its enable-cinderx-module instead of enable-optimizations now idk why
run \
    /cinder/src/configure --prefix=/cinder --enable-cinderx-module && \
    make clean && \
    make -j8 VERBOSE=$MAKE_VERBOSE


# build for x86
FROM --platform=linux/amd64 build AS install

WORKDIR /cinder/build
RUN make install

# build for x86
FROM --platform=linux/amd64 ubuntu:24.04 AS runtime

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

# copy artifact
COPY --from=install /cinder /cinder
ENV PATH="/cinder/bin:$PATH"

WORKDIR /cinder/src
CMD ["/bin/bash"]

