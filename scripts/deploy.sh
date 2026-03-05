#!/usr/bin/env bash
# Deploy crew + app-skill binaries to the Cloud Mac Mini.
# Usage: ./scripts/deploy.sh
set -euo pipefail

REMOTE="cloud@69.194.3.128"
REMOTE_PW="zjsgf128"
SCP="sshpass -p $REMOTE_PW scp -o PubkeyAuthentication=no"
SSH="sshpass -p $REMOTE_PW ssh -o PubkeyAuthentication=no $REMOTE"
REMOTE_BIN="/Users/cloud/.cargo/bin"
PLIST="io.ominix.crew-serve"

BINARIES=(crew news_fetch deep-search deep_crawl send_email account_manager asr)

echo "==> Building release binaries..."
cargo build --release -p crew-cli --features telegram,whatsapp,feishu,twilio,api
cargo build --release -p news_fetch -p deep-search -p deep-crawl -p send-email -p account-manager -p asr

# Build ominix-api if source is available
OMINIX_DIR="${OMINIX_DIR:-$HOME/home/OminiX-MLX}"
if [ -d "$OMINIX_DIR" ]; then
    echo "==> Building ominix-api..."
    (cd "$OMINIX_DIR" && cargo build --release -p ominix-api --features asr,tts)
    codesign -s - "$OMINIX_DIR/target/release/ominix-api" 2>/dev/null || true
fi

echo "==> Signing binaries locally..."
for bin in "${BINARIES[@]}"; do
    codesign -s - "target/release/$bin" 2>/dev/null || true
done

echo "==> Uploading binaries to remote..."
for bin in "${BINARIES[@]}"; do
    echo "    $bin"
    $SCP "target/release/$bin" "$REMOTE:/tmp/${bin}.new"
done

# Upload ominix-api if built
if [ -d "$OMINIX_DIR" ] && [ -f "$OMINIX_DIR/target/release/ominix-api" ]; then
    echo "    ominix-api"
    $SCP "$OMINIX_DIR/target/release/ominix-api" "$REMOTE:/tmp/ominix-api.new"
    if [ -f "$OMINIX_DIR/target/release/mlx.metallib" ]; then
        echo "    mlx.metallib"
        $SCP "$OMINIX_DIR/target/release/mlx.metallib" "$REMOTE:/tmp/mlx.metallib.new"
    fi
fi

echo "==> Stopping launchd service..."
$SSH "launchctl unload ~/Library/LaunchAgents/${PLIST}.plist 2>/dev/null || true"
sleep 1
$SSH "pkill -f 'crew serve' 2>/dev/null || true; pkill -f 'crew gateway' 2>/dev/null || true"
sleep 1

echo "==> Replacing binaries on remote..."
for bin in "${BINARIES[@]}"; do
    $SSH "mv /tmp/${bin}.new ${REMOTE_BIN}/${bin} && codesign --force -s - ${REMOTE_BIN}/${bin}"
done

# Replace ominix-api if uploaded
if $SSH "[ -f /tmp/ominix-api.new ]" 2>/dev/null; then
    echo "==> Replacing ominix-api on remote..."
    $SSH "launchctl unload ~/Library/LaunchAgents/io.ominix.ominix-api.plist 2>/dev/null || true; sleep 1"
    $SSH "mv /tmp/ominix-api.new ${REMOTE_BIN}/ominix-api && codesign --force -s - ${REMOTE_BIN}/ominix-api"
    if $SSH "[ -f /tmp/mlx.metallib.new ]" 2>/dev/null; then
        $SSH "mv /tmp/mlx.metallib.new ${REMOTE_BIN}/mlx.metallib"
    fi
    # Create launchd plist for ominix-api if it doesn't exist
    $SSH "[ -f ~/Library/LaunchAgents/io.ominix.ominix-api.plist ] || cat > /tmp/ominix-api.plist << 'PEOF'
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>Label</key>
    <string>io.ominix.ominix-api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/cloud/.local/bin/ominix-api</string>
        <string>--port</string>
        <string>8080</string>
        <string>--models-dir</string>
        <string>/Users/cloud/.ominix/models</string>
        <string>--asr-model</string>
        <string>qwen3-asr-1.7b</string>
        <string>--tts-model</string>
        <string>Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit</string>
    </array>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/cloud/.ominix/api.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/cloud/.ominix/api.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/cloud/.local/bin:/Users/cloud/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
PEOF
    mv /tmp/ominix-api.plist ~/Library/LaunchAgents/io.ominix.ominix-api.plist"
    $SSH "launchctl load ~/Library/LaunchAgents/io.ominix.ominix-api.plist"
    echo "    ominix-api service started"
fi

echo "==> Ensuring ffmpeg is installed (required for OGG/Opus ASR)..."
$SSH "/opt/homebrew/bin/brew list ffmpeg &>/dev/null || /opt/homebrew/bin/brew install ffmpeg" || echo "  WARN: could not install ffmpeg"

echo "==> Cleaning stale skill dirs (bootstrap recreates them)..."
for skill in news deep-search deep-crawl send-email account-manager asr; do
    $SSH "rm -rf /Users/cloud/.crew/skills/${skill}" 2>/dev/null || true
done

# Check ASR/TTS models (try both ~/.ominix and ~/.OminiX)
echo "==> Checking voice models..."
$SSH "ls -d ~/.ominix/models/Qwen3-ASR* ~/.OminiX/models/Qwen3-ASR* 2>/dev/null | head -1 && echo '  ASR model: OK' || echo '  WARN: ASR model missing'"
$SSH "ls -d ~/.ominix/models/Qwen3-TTS* ~/.OminiX/models/Qwen3-TTS* 2>/dev/null | head -1 && echo '  TTS model: OK' || echo '  WARN: TTS model missing'"

echo "==> Starting launchd service..."
$SSH "launchctl load ~/Library/LaunchAgents/${PLIST}.plist"

echo "==> Done! Verifying..."
sleep 2
$SSH "launchctl list | grep crew || echo 'WARNING: service not found'"
echo "Deploy complete."
