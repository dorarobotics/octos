# ============================================================
# Stage 1: Build the crew binary
# ============================================================
FROM rust:1.85-alpine AS builder

RUN apk add --no-cache musl-dev git pkgconfig

WORKDIR /src

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
COPY crates/crew-core/Cargo.toml crates/crew-core/Cargo.toml
COPY crates/crew-llm/Cargo.toml crates/crew-llm/Cargo.toml
COPY crates/crew-memory/Cargo.toml crates/crew-memory/Cargo.toml
COPY crates/crew-agent/Cargo.toml crates/crew-agent/Cargo.toml
COPY crates/crew-bus/Cargo.toml crates/crew-bus/Cargo.toml
COPY crates/crew-cli/Cargo.toml crates/crew-cli/Cargo.toml

# Create stub lib.rs files for dependency caching
RUN mkdir -p crates/crew-core/src && echo "" > crates/crew-core/src/lib.rs && \
    mkdir -p crates/crew-llm/src && echo "" > crates/crew-llm/src/lib.rs && \
    mkdir -p crates/crew-memory/src && echo "" > crates/crew-memory/src/lib.rs && \
    mkdir -p crates/crew-agent/src && echo "" > crates/crew-agent/src/lib.rs && \
    mkdir -p crates/crew-bus/src && echo "" > crates/crew-bus/src/lib.rs && \
    mkdir -p crates/crew-cli/src && echo "fn main() {}" > crates/crew-cli/src/main.rs

RUN cargo build --release --bin crew 2>/dev/null || true

# Copy full source and build
COPY . .
RUN touch crates/*/src/*.rs && \
    cargo build --release --bin crew \
      --features telegram,discord,slack,whatsapp,feishu,email

# ============================================================
# Stage 2: Minimal runtime image
# ============================================================
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata \
    # Runtime deps for skills (pptx, mofa-pptx, browser)
    nodejs npm ffmpeg chromium \
    # LibreOffice + Poppler for office document conversion and visual QA
    libreoffice poppler-utils \
    # GCC for soffice sandbox shim (compiled on first use if needed)
    gcc musl-dev

# Install Node.js skill dependencies
RUN npm install -g pptxgenjs react-icons react react-dom sharp

# Copy binary
COPY --from=builder /src/target/release/crew /usr/local/bin/crew

# Copy builtin skills
COPY --from=builder /src/crates/crew-agent/skills /opt/crew/skills

# Create workspace
RUN mkdir -p /root/.crew/skills && \
    cp -r /opt/crew/skills/* /root/.crew/skills/ 2>/dev/null || true

ENTRYPOINT ["crew"]
CMD ["gateway"]
