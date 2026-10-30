# Multi-stage Dockerfile for Cross-Language HPR-QP Solver Suite
# Stage 1: Build C, C++, and Rust targets
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libomp-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY c/ ./c/
COPY cpp/ ./cpp/
COPY rust/ ./rust/

# Build C targets
WORKDIR /build/c
RUN gcc -O3 -Wall -std=c99 -fopenmp -shared -fPIC benchmark_fast_c.c -o libhpr_fast.so || true
RUN gcc -O3 -Wall -std=c99 hpr_qp.c benchmark_cvxqp1_l.c -lm -o benchmark_cvxqp1_l_c || true

# Build C++ targets
WORKDIR /build/cpp
RUN g++ -O3 -std=c++20 -fopenmp benchmark_cvxqp1_l.cpp hpr_qp.cpp -o benchmark_cvxqp1_l_cpp || true

# Build Rust targets
WORKDIR /build/rust
RUN cargo build --release

# Stage 2: Final runtime container with Python and precompiled binaries
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ocl-icd-libopencl1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python code and project metadata
COPY pyproject.toml ./
COPY hpr_qp.py transformer_scratch.py cli.py visualize_convergence.py ./
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY assets/ ./assets/

# Install Python requirements
RUN pip3 install --no-cache-dir torch numpy matplotlib pytest

# Copy compiled binaries from builder stage
COPY --from=builder /build/c/benchmark_cvxqp1_l_c /usr/local/bin/
COPY --from=builder /build/cpp/benchmark_cvxqp1_l_cpp /usr/local/bin/
COPY --from=builder /build/rust/target/release/benchmark_cvxqp1_l /usr/local/bin/benchmark_cvxqp1_l_rust

ENTRYPOINT ["python3", "cli.py"]
CMD ["--problem", "hs21", "--backend", "cpu"]
