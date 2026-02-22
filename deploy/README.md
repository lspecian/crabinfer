# CrabInfer Server Deployment

## Docker

```bash
# Build
docker build -t crabinfer-server .

# Run (mount your model directory)
docker run -p 8080:8080 -v /path/to/models:/models crabinfer-server \
  --model /models/your-model.gguf
```

## Systemd

```bash
# 1. Create service user
sudo useradd -r -s /usr/sbin/nologin crabinfer
sudo mkdir -p /var/lib/crabinfer/models
sudo chown crabinfer:crabinfer /var/lib/crabinfer

# 2. Copy binary
sudo cp target/release/crabinfer-server /usr/local/bin/

# 3. Install service
sudo cp deploy/crabinfer-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crabinfer-server

# 4. Check status
sudo systemctl status crabinfer-server
journalctl -u crabinfer-server -f
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/v1/messages` | Anthropic-compatible messages |
