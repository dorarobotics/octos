---
name: asr
description: OminiX ASR (speech-to-text), TTS (text-to-speech), and podcast generation via Qwen3 models on Apple Silicon. Triggers: voice, transcribe audio, text to speech, podcast, 语音识别, 语音合成, speak this, read aloud, 播客.
version: 1.0.0
author: hagency
always: false
---

# OminiX ASR / TTS / Podcast

On-device speech-to-text, text-to-speech, and multi-speaker podcast generation using OminiX Qwen3 ASR/TTS models on Apple Silicon.

## Configuration

The asr skill auto-discovers the ominix-api server URL via (in priority order):
1. `OMINIX_API_URL` environment variable
2. Discovery file `~/.ominix/api_url` (written by ominix-api on startup)
3. Default: `http://localhost:8080`

## Tools

### voice_transcribe

Transcribe an audio file to text. Supports WAV, OGG, MP3, FLAC, M4A.

```json
{"audio_path": "/tmp/voice.ogg", "language": "Chinese"}
```

**Parameters:**
- `audio_path` (required): Absolute path to the audio file
- `language` (optional, default "Chinese"): "Chinese", "English", "Japanese", "Korean", "Cantonese", etc.

### voice_synthesize

Generate speech audio from text. Produces a WAV file.

```json
{"text": "Hello world", "language": "english", "speaker": "vivian"}
```

**Parameters:**
- `text` (required): Text to synthesize
- `output_path` (optional): Where to save WAV. Default: `/tmp/crew_tts_<timestamp>.wav`
- `language` (optional, default "chinese"): "chinese", "english", "japanese", "korean"
- `speaker` (optional, default "vivian"): Voice preset name

**Available speakers:** vivian, serena, ryan, aiden, eric, dylan (English/Chinese), uncle_fu (Chinese), ono_anna (Japanese), sohee (Korean)

### generate_podcast

Generate multi-speaker podcast from a dialogue script.

```json
{
  "script": [
    {"speaker": "Host", "voice": "vivian", "text": "Welcome to today's episode..."},
    {"speaker": "Guest", "voice": "ryan", "text": "Thanks for having me..."}
  ],
  "output_path": "/tmp/podcast.wav",
  "language": "english"
}
```

All segments are synthesized individually and concatenated into a single WAV file (24kHz 16-bit mono). If ffmpeg is available, an MP3 version is also produced.
