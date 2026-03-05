//! ASR/TTS/Podcast skill binary via ominix-api.
//!
//! Protocol: `./main <tool_name>` with JSON on stdin, JSON on stdout.
//! Requires OMINIX_API_URL environment variable (default: http://localhost:8080).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

// ── Input types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TranscribeInput {
    audio_path: String,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Deserialize)]
struct SynthesizeInput {
    text: String,
    #[serde(default)]
    output_path: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    speaker: Option<String>,
}

#[derive(Deserialize)]
struct PodcastInput {
    script: Vec<PodcastLine>,
    #[serde(default)]
    output_path: Option<String>,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Deserialize)]
struct PodcastLine {
    speaker: String,
    voice: String,
    text: String,
}

#[derive(Deserialize)]
struct VoiceCloneInput {
    text: String,
    reference_audio: String,
    #[serde(default)]
    output_path: Option<String>,
    #[serde(default)]
    language: Option<String>,
}

// ── Helpers ──────────────────────────────────────────────────────────

fn api_base_url() -> String {
    std::env::var("OMINIX_API_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string())
        .trim_end_matches('/')
        .to_string()
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(5))
        .build()
        .expect("failed to build HTTP client")
}

fn check_health(client: &reqwest::blocking::Client, base_url: &str) -> Result<(), String> {
    match client
        .get(format!("{base_url}/health"))
        .timeout(Duration::from_secs(5))
        .send()
    {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(format!(
            "ominix-api returned HTTP {} — is it running on {base_url}?",
            resp.status()
        )),
        Err(e) => Err(format!(
            "Cannot reach ominix-api at {base_url}: {e}. \
             Start it with: ominix-api --port 8081"
        )),
    }
}

fn fail(msg: &str) -> ! {
    let out = json!({"output": msg, "success": false});
    println!("{out}");
    std::process::exit(1);
}

fn succeed(msg: &str) -> ! {
    let out = json!({"output": msg, "success": true});
    println!("{out}");
    std::process::exit(0);
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let end: String = s.chars().take(max).collect();
        format!("{end}...")
    }
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── voice_transcribe ─────────────────────────────────────────────────

fn handle_transcribe(input_json: &str) {
    let input: TranscribeInput = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => fail(&format!("Invalid input: {e}")),
    };

    let path = Path::new(&input.audio_path);
    if !path.exists() {
        fail(&format!("Audio file not found: {}", input.audio_path));
    }
    if !path.is_file() {
        fail(&format!("Not a file: {}", input.audio_path));
    }
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() == 0 {
            fail("Audio file is empty (0 bytes)");
        }
        if meta.len() > 100_000_000 {
            fail("Audio file too large (>100MB)");
        }
    }

    let base_url = api_base_url();
    let client = http_client();
    if let Err(e) = check_health(&client, &base_url) {
        fail(&e);
    }

    let language = input.language.unwrap_or_else(|| "Chinese".to_string());

    let body = json!({
        "file_path": input.audio_path,
        "language": language,
        "response_format": "verbose_json"
    });

    let resp = match client
        .post(format!("{base_url}/v1/audio/transcriptions"))
        .json(&body)
        .send()
    {
        Ok(r) => r,
        Err(e) => fail(&format!("ASR request failed: {e}")),
    };

    let status = resp.status();
    let resp_text = resp.text().unwrap_or_default();

    if !status.is_success() {
        fail(&format!(
            "ASR error (HTTP {status}): {}",
            truncate(&resp_text, 200)
        ));
    }

    let result: serde_json::Value = match serde_json::from_str(&resp_text) {
        Ok(v) => v,
        Err(e) => fail(&format!("Failed to parse ASR response: {e}")),
    };

    let text = result["text"].as_str().unwrap_or("").trim();
    if text.is_empty() {
        fail("ASR returned empty transcription (silence or unsupported format)");
    }

    let mut output = text.to_string();
    if let Some(duration) = result["duration"].as_f64() {
        output = format!("{text}\n\n[Audio duration: {duration:.1}s]");
    }

    succeed(&output);
}

// ── voice_synthesize ─────────────────────────────────────────────────

fn synthesize_segment(
    client: &reqwest::blocking::Client,
    base_url: &str,
    text: &str,
    voice: &str,
    language: &str,
    output_path: &Path,
) -> Result<(usize, f64), String> {
    let body = json!({
        "input": text,
        "voice": voice,
        "language": language
    });

    let resp = client
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body)
        .send()
        .map_err(|e| format!("TTS request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let resp_text = resp.text().unwrap_or_default();
        return Err(format!(
            "TTS error (HTTP {status}): {}",
            truncate(&resp_text, 200)
        ));
    }

    let wav_bytes = resp
        .bytes()
        .map_err(|e| format!("Failed to read TTS response: {e}"))?;

    if wav_bytes.len() < 44 {
        return Err("TTS returned invalid WAV data (too small)".to_string());
    }

    std::fs::write(output_path, &wav_bytes)
        .map_err(|e| format!("Failed to write {}: {e}", output_path.display()))?;

    // 24kHz 16-bit mono = 48000 bytes/sec
    let duration_secs = wav_bytes.len().saturating_sub(44) as f64 / 48000.0;
    Ok((wav_bytes.len(), duration_secs))
}

fn handle_synthesize(input_json: &str) {
    let input: SynthesizeInput = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => fail(&format!("Invalid input: {e}")),
    };

    if input.text.trim().is_empty() {
        fail("'text' must not be empty");
    }

    let base_url = api_base_url();
    let client = http_client();
    if let Err(e) = check_health(&client, &base_url) {
        fail(&e);
    }

    let output_path = input
        .output_path
        .unwrap_or_else(|| format!("/tmp/crew_tts_{}.wav", timestamp()));

    if let Some(parent) = Path::new(&output_path).parent() {
        if !parent.exists() {
            fail(&format!(
                "Output directory does not exist: {}",
                parent.display()
            ));
        }
    }

    let language = input.language.unwrap_or_else(|| "chinese".to_string());
    let speaker = input.speaker.unwrap_or_else(|| "vivian".to_string());

    match synthesize_segment(&client, &base_url, &input.text, &speaker, &language, Path::new(&output_path)) {
        Ok((size, duration_secs)) => {
            succeed(&format!(
                "Generated audio: {output_path} ({duration_secs:.1}s, {size} bytes). Use send_file to deliver it to the user."
            ));
        }
        Err(e) => fail(&e),
    }
}

// ── generate_podcast ─────────────────────────────────────────────────

/// Build a 44-byte WAV header for 24kHz 16-bit mono PCM.
fn wav_header(sample_rate: u32, channels: u16, bits_per_sample: u16, data_len: u32) -> [u8; 44] {
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let file_size = 36 + data_len;

    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&file_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes()); // subchunk1 size
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM format
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_len.to_le_bytes());
    h
}

/// Concatenate multiple WAV files (must all be 24kHz 16-bit mono) into one.
fn concat_wav(segments: &[PathBuf], output: &Path) -> Result<(), String> {
    let mut pcm = Vec::new();
    for seg in segments {
        let data =
            std::fs::read(seg).map_err(|e| format!("Failed to read {}: {e}", seg.display()))?;
        // Skip the 44-byte WAV header, append raw PCM
        if data.len() > 44 {
            pcm.extend_from_slice(&data[44..]);
        }
    }

    let header = wav_header(24000, 1, 16, pcm.len() as u32);
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(&pcm);
    std::fs::write(output, out)
        .map_err(|e| format!("Failed to write {}: {e}", output.display()))?;
    Ok(())
}

fn handle_podcast(input_json: &str) {
    let input: PodcastInput = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => fail(&format!("Invalid input: {e}")),
    };

    if input.script.is_empty() {
        fail("'script' must not be empty");
    }

    let base_url = api_base_url();
    let client = http_client();
    if let Err(e) = check_health(&client, &base_url) {
        fail(&e);
    }

    let output_path = input
        .output_path
        .unwrap_or_else(|| format!("/tmp/crew_podcast_{}.wav", timestamp()));

    let language = input.language.unwrap_or_else(|| "chinese".to_string());

    let mut segments: Vec<PathBuf> = Vec::new();
    let mut total_duration = 0.0f64;

    for (i, line) in input.script.iter().enumerate() {
        if line.text.trim().is_empty() {
            continue;
        }

        let seg_path = PathBuf::from(format!("/tmp/crew_podcast_seg_{}_{}.wav", timestamp(), i));

        match synthesize_segment(&client, &base_url, &line.text, &line.voice, &language, &seg_path)
        {
            Ok((_size, duration)) => {
                total_duration += duration;
                segments.push(seg_path);
            }
            Err(e) => {
                // Clean up segments on failure
                for seg in &segments {
                    let _ = std::fs::remove_file(seg);
                }
                fail(&format!(
                    "Failed to synthesize line {} ({} / {}): {e}",
                    i + 1,
                    line.speaker,
                    line.voice
                ));
            }
        }
    }

    if segments.is_empty() {
        fail("No audio segments generated (all lines were empty)");
    }

    // Concatenate all segments
    if let Err(e) = concat_wav(&segments, Path::new(&output_path)) {
        for seg in &segments {
            let _ = std::fs::remove_file(seg);
        }
        fail(&format!("Failed to concatenate segments: {e}"));
    }

    // Clean up individual segments
    for seg in &segments {
        let _ = std::fs::remove_file(seg);
    }

    let segment_count = segments.len();

    // Optionally convert to MP3 via ffmpeg
    let mp3_path = output_path.replace(".wav", ".mp3");
    let mp3_note = if std::process::Command::new("ffmpeg")
        .args([
            "-i",
            &output_path,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-y",
            &mp3_path,
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        format!("\nMP3 version: {mp3_path}")
    } else {
        String::new()
    };

    let minutes = (total_duration / 60.0).floor() as u32;
    let seconds = (total_duration % 60.0).round() as u32;

    succeed(&format!(
        "Podcast generated: {output_path} ({minutes}m {seconds}s, {segment_count} segments){mp3_note}\n\nUse send_file to deliver it to the user."
    ));
}

// ── voice_clone_synthesize ────────────────────────────────────────────

fn handle_voice_clone(input_json: &str) {
    let input: VoiceCloneInput = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => fail(&format!("Invalid input: {e}")),
    };

    if input.text.trim().is_empty() {
        fail("'text' must not be empty");
    }

    let ref_path = Path::new(&input.reference_audio);
    if !ref_path.exists() {
        fail(&format!("Reference audio not found: {}", input.reference_audio));
    }
    if !ref_path.is_file() {
        fail(&format!("Not a file: {}", input.reference_audio));
    }
    if let Ok(meta) = std::fs::metadata(ref_path) {
        if meta.len() == 0 {
            fail("Reference audio file is empty (0 bytes)");
        }
        if meta.len() > 50_000_000 {
            fail("Reference audio too large (>50MB)");
        }
    }

    let base_url = api_base_url();
    let client = http_client();
    if let Err(e) = check_health(&client, &base_url) {
        fail(&e);
    }

    let output_path = input
        .output_path
        .unwrap_or_else(|| format!("/tmp/crew_voice_clone_{}.wav", timestamp()));

    if let Some(parent) = Path::new(&output_path).parent() {
        if !parent.exists() {
            fail(&format!(
                "Output directory does not exist: {}",
                parent.display()
            ));
        }
    }

    let language = input.language.unwrap_or_else(|| "chinese".to_string());

    let body = json!({
        "input": input.text,
        "reference_audio": input.reference_audio,
        "language": language
    });

    let resp = match client
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body)
        .send()
    {
        Ok(r) => r,
        Err(e) => fail(&format!("Voice clone TTS request failed: {e}")),
    };

    let status = resp.status();
    if !status.is_success() {
        let resp_text = resp.text().unwrap_or_default();
        fail(&format!(
            "Voice clone error (HTTP {status}): {}",
            truncate(&resp_text, 200)
        ));
    }

    let wav_bytes = match resp.bytes() {
        Ok(b) => b,
        Err(e) => fail(&format!("Failed to read TTS response: {e}")),
    };

    if wav_bytes.len() < 44 {
        fail("Voice clone returned invalid WAV data (too small)");
    }

    if let Err(e) = std::fs::write(&output_path, &wav_bytes) {
        fail(&format!("Failed to write {output_path}: {e}"));
    }

    let duration_secs = wav_bytes.len().saturating_sub(44) as f64 / 48000.0;

    succeed(&format!(
        "Voice clone audio generated: {output_path} ({duration_secs:.1}s, {} bytes). Use send_file to deliver it to the user.",
        wav_bytes.len()
    ));
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tool_name = args.get(1).map(|s| s.as_str()).unwrap_or("unknown");

    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
        fail(&format!("Failed to read stdin: {e}"));
    }

    match tool_name {
        "voice_transcribe" => handle_transcribe(&buf),
        "voice_synthesize" => handle_synthesize(&buf),
        "generate_podcast" => handle_podcast(&buf),
        "voice_clone_synthesize" => handle_voice_clone(&buf),
        _ => fail(&format!(
            "Unknown tool '{tool_name}'. Expected: voice_transcribe, voice_synthesize, generate_podcast, voice_clone_synthesize"
        )),
    }
}
