# Cinder Docker Container Fix Summary

## Original Problems

### 1. Missing cinderx Extension Build
The original Dockerfile only built the base Cinder Python interpreter but **did not build the cinderx extension module**. The cinderx module provides:
- JIT compiler (`-X jit`)
- Shadow frame support (`-X jit-shadow-frame`)
- Strict module loader (`-X install-strict-loader`)
- Static Python support (`__static__` imports)

Without cinderx, the `-X jit` flags would fail.

### 2. Ubuntu 24.04 Compiler Incompatibility
Ubuntu 24.04 ships with GCC 14, which has stricter warnings that are treated as errors by default in Cinder's build. Specifically:
- `-Werror=strict-prototypes` failed on old-style C function declarations like `_PyCode_AllocMutable()` without a prototype

### 3. Incorrect CFLAGS
The original `-Wno-error=maybe-uninitialized` flag was insufficient to suppress all the warnings-as-errors from newer GCC versions.

## Fixes Applied

### 1. Switched to Fedora 40
Better compiler compatibility and cleaner package management for the required build dependencies.

### 2. Added Proper Compiler Flags
```dockerfile
CFLAGS="-Wno-error -Wno-error=strict-prototypes"
CXXFLAGS="-Wno-error"
```
Also passed `CFLAGS` to `make` to ensure flags propagate correctly.

### 3. Added cinderx Build Step
Used the official `build.sh` script to build the cinderx extension:
```dockerfile
WORKDIR /cinder/cinderx
RUN ./build.sh --build-root /cinder --python-bin /cinder/python --output-dir /cinder
```

### 4. Set Up Correct Environment
```dockerfile
ENV PATH="/cinder:${PATH}"
ENV PYTHONPATH="/cinder"
```

## Result
The command now works:
```bash
cd ~/static-python-perf/Benchmark/deltablue/advanced && python -X install-strict-loader -X jit -X jit-shadow-frame main.py
```
