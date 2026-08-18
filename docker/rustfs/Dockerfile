# Dockerfile for RustFS
# Lightweight high-performance file and artifact storage service

FROM rust:1.77-bookworm as builder

WORKDIR /build

# Initialize Rust project
RUN cargo init --name rustfs

WORKDIR /build/rustfs

# Add dependencies to Cargo.toml
RUN cat >> Cargo.toml << 'EOF'
actix-web = "4.4"
actix-files = "0.6"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
EOF

# Copy main.rs with actual artifact serving capabilities
RUN cat > src/main.rs << 'EOF'
use actix_files::Files;
use actix_web::{web, App, HttpServer, HttpResponse, Responder, middleware};
use std::fs;
use std::path::Path;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let bind_addr = std::env::var("RUSTFS_LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    
    let data_dir = std::env::var("RUSTFS_DATA_DIR")
        .unwrap_or_else(|_| "/data".to_string());
    
    // Ensure data directory exists
    if !Path::new(&data_dir).exists() {
        fs::create_dir_all(&data_dir)?;
    }

    println!("Starting RustFS on {}", bind_addr);
    println!("Data directory: {}", data_dir);
    
    let serve_dir = data_dir.clone();
    
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health_check))
            .route("/status", web::get().to(status))
            .service(Files::new("/artifacts", &serve_dir).show_files_listing())
    })
    .bind(&bind_addr)?
    .run()
    .await
}

async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({ 
        "status": "ok" 
    }))
}

async fn status() -> impl Responder {
    let data_dir = std::env::var("RUSTFS_DATA_DIR")
        .unwrap_or_else(|_| "/data".to_string());
    HttpResponse::Ok().json(serde_json::json!({ 
        "service": "rustfs",
        "status": "running",
        "data_dir": data_dir
    }))
}
EOF

# Build the binary in release mode
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/rustfs/target/release/rustfs /usr/local/bin/rustfs

# Create storage directory
RUN mkdir -p /data

WORKDIR /data

# Expose port
EXPOSE 8080

# Environment variables
ENV RUSTFS_LISTEN_ADDR=0.0.0.0:8080 \
    RUSTFS_DATA_DIR=/data

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${RUSTFS_PORT:-8080}/health || exit 1

ENTRYPOINT ["rustfs"]
