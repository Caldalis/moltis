//! VoxCPM TTS provider, served through a vLLM-Omni speech endpoint.
//!
//! [VoxCPM](https://github.com/OpenBMB/VoxCPM) is a tokenizer-free TTS model
//! released under Apache-2.0. It is served here through
//! [vLLM-Omni](https://github.com/vllm-project/vllm-omni), which exposes an
//! OpenAI-compatible speech API extended with VoxCPM-specific fields:
//!
//! ```bash
//! vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000
//! ```
//!
//! Two endpoints are used:
//!
//! - `POST {endpoint}/audio/speech` — synthesis
//! - `GET  {endpoint}/audio/voices` — the speakers the server actually has
//!
//! Unlike the OpenAI provider, `voice` is never defaulted: VoxCPM rejects
//! speaker names it does not know, and omitting the field selects zero-shot
//! synthesis. Voice personas are applied as a VoxCPM *voice design* prefix on
//! the input text, because the VoxCPM2 serving adapter reads the description
//! from the text rather than from a separate `instructions` field.

use {
    crate::{
        config::VoxCpmTtsConfig,
        tts::{AudioFormat, AudioOutput, SynthesizeRequest, TtsProvider, Voice},
    },
    anyhow::{Result, anyhow},
    async_trait::async_trait,
    bytes::Bytes,
    reqwest::Client,
    serde::{Deserialize, Serialize},
    std::{borrow::Cow, time::Duration},
};

/// Default vLLM-Omni OpenAI-compatible base URL.
pub const DEFAULT_ENDPOINT: &str = "http://localhost:8000/v1";

/// Synthesis can be slow on CPU-bound or cold servers.
const SYNTHESIZE_TIMEOUT: Duration = Duration::from_secs(120);

/// Listing voices is a cheap metadata call.
const VOICES_TIMEOUT: Duration = Duration::from_secs(5);

/// VoxCPM TTS provider backed by a vLLM-Omni server.
#[derive(Clone, Debug)]
pub struct VoxCpmTts {
    client: Client,
    endpoint: String,
    model: Option<String>,
    voice: Option<String>,
    voice_design: bool,
}

impl VoxCpmTts {
    /// Create a new VoxCPM provider from config.
    #[must_use]
    pub fn new(config: &VoxCpmTtsConfig) -> Self {
        Self {
            client: Client::new(),
            endpoint: normalize_endpoint(&config.endpoint),
            model: non_empty(config.model.as_deref()),
            voice: non_empty(config.voice.as_deref()),
            voice_design: config.voice_design,
        }
    }

    /// Map a moltis audio format onto a vLLM-Omni `response_format`.
    ///
    /// vLLM-Omni supports `wav`, `mp3`, `flac`, `pcm` and `opus`. Formats it
    /// does not serve fall back to the closest one it does, and the format
    /// actually returned is reported back so callers never mislabel the bytes.
    fn negotiate_format(requested: AudioFormat) -> (&'static str, AudioFormat) {
        match requested {
            AudioFormat::Mp3 | AudioFormat::Aac => ("mp3", AudioFormat::Mp3),
            AudioFormat::Opus | AudioFormat::Webm => ("opus", AudioFormat::Opus),
            AudioFormat::Pcm => ("pcm", AudioFormat::Pcm),
            AudioFormat::Wav => ("wav", AudioFormat::Wav),
        }
    }
}

/// Trim a trailing slash so `{endpoint}/audio/speech` never doubles up.
fn normalize_endpoint(endpoint: &str) -> String {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        DEFAULT_ENDPOINT.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Treat blank config strings as unset.
fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
}

/// Flatten a persona instruction block into a single parenthesised clause.
///
/// `VoicePersonaPrompt::render` produces newline-separated `Key: value` lines.
/// VoxCPM reads a voice description from a `(...)` prefix on the text, so the
/// block is collapsed onto one line and its own parentheses are dropped to
/// keep the delimiter unambiguous.
fn flatten_voice_design(instructions: &str) -> String {
    let mut out = String::with_capacity(instructions.len());
    let mut pending_space = false;

    for ch in instructions.chars() {
        match ch {
            '(' | ')' => {},
            '\n' | '\r' => {
                if !out.is_empty() {
                    // Newline-separated fields read better as a comma list.
                    if !out.ends_with(',') {
                        out.push(',');
                    }
                    pending_space = true;
                }
            },
            c if c.is_whitespace() => {
                if !out.is_empty() {
                    pending_space = true;
                }
            },
            c => {
                if pending_space {
                    out.push(' ');
                    pending_space = false;
                }
                out.push(c);
            },
        }
    }

    while out.ends_with(',') {
        out.pop();
    }
    out
}

/// Prefix `text` with a VoxCPM voice-design clause built from `instructions`.
///
/// Returns the text unchanged when voice design is disabled, when there are no
/// instructions, or when the text already carries its own `(...)` prefix.
fn apply_voice_design<'a>(
    text: &'a str,
    instructions: Option<&str>,
    enabled: bool,
) -> Cow<'a, str> {
    if !enabled {
        return Cow::Borrowed(text);
    }
    let Some(instructions) = instructions else {
        return Cow::Borrowed(text);
    };
    if text.trim_start().starts_with('(') {
        // Caller (or a `[[tts:...]]` directive) already supplied a design prefix.
        return Cow::Borrowed(text);
    }
    let design = flatten_voice_design(instructions);
    if design.is_empty() {
        return Cow::Borrowed(text);
    }
    Cow::Owned(format!("({design}){text}"))
}

#[async_trait]
impl TtsProvider for VoxCpmTts {
    fn id(&self) -> &'static str {
        "voxcpm"
    }

    fn name(&self) -> &'static str {
        "VoxCPM"
    }

    fn is_configured(&self) -> bool {
        // Mirrors the other local providers: a non-default endpoint or an
        // explicit model counts as deliberate configuration. Whether the
        // server is actually up is probed separately by the gateway.
        self.endpoint != DEFAULT_ENDPOINT || self.model.is_some()
    }

    async fn voices(&self) -> Result<Vec<Voice>> {
        let url = format!("{}/audio/voices", self.endpoint);
        let resp = self
            .client
            .get(&url)
            .timeout(VOICES_TIMEOUT)
            .send()
            .await
            .map_err(|e| {
                anyhow!(
                    "failed to reach VoxCPM server at {}: {e}. Start it with: vllm serve openbmb/VoxCPM2 --omni --port 8000",
                    self.endpoint
                )
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("VoxCPM voices request failed: {status} - {body}"));
        }

        let listing: VoicesResponse = resp
            .json()
            .await
            .map_err(|e| anyhow!("failed to parse VoxCPM voices response: {e}"))?;

        Ok(listing.into_voices())
    }

    async fn synthesize(&self, request: SynthesizeRequest) -> Result<AudioOutput> {
        let (wire_format, actual_format) = Self::negotiate_format(request.output_format);
        let input = apply_voice_design(
            &request.text,
            request.instructions.as_deref(),
            self.voice_design,
        );

        // `voice` is deliberately not defaulted: VoxCPM validates the name
        // against its registered speakers and rejects unknown ones, while
        // omitting it selects zero-shot synthesis.
        let voice = request.voice_id.as_deref().or(self.voice.as_deref());

        let body = SpeechRequest {
            model: request.model.as_deref().or(self.model.as_deref()),
            input: input.as_ref(),
            voice,
            response_format: wire_format,
            speed: request.speed,
        };

        let url = format!("{}/audio/speech", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .timeout(SYNTHESIZE_TIMEOUT)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                anyhow!(
                    "failed to reach VoxCPM server at {}: {e}. Start it with: vllm serve openbmb/VoxCPM2 --omni --port 8000",
                    self.endpoint
                )
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("VoxCPM synthesis failed: {status} - {body}"));
        }

        let data = resp.bytes().await?;
        if data.is_empty() {
            return Err(anyhow!("VoxCPM returned an empty audio response"));
        }

        Ok(AudioOutput {
            data: Bytes::from(data.to_vec()),
            format: actual_format,
            duration_ms: None,
        })
    }
}

// ── Wire types ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct SpeechRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<&'a str>,
    response_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct VoicesResponse {
    #[serde(default)]
    voices: Vec<String>,
    #[serde(default)]
    uploaded_voices: Vec<UploadedVoice>,
}

#[derive(Debug, Deserialize)]
struct UploadedVoice {
    name: String,
    #[serde(default)]
    speaker_description: Option<String>,
    #[serde(default)]
    ref_text: Option<String>,
}

impl VoicesResponse {
    /// Merge the preset and uploaded listings, preferring uploaded metadata.
    fn into_voices(self) -> Vec<Voice> {
        let described: std::collections::HashMap<String, UploadedVoice> = self
            .uploaded_voices
            .into_iter()
            .map(|v| (v.name.clone(), v))
            .collect();

        self.voices
            .into_iter()
            .map(|id| {
                let description = described.get(&id).and_then(|u| {
                    u.speaker_description
                        .clone()
                        .or_else(|| u.ref_text.as_ref().map(|t| format!("Cloned from: {t}")))
                });
                Voice {
                    name: id.clone(),
                    id,
                    description,
                    preview_url: None,
                }
            })
            .collect()
    }
}

#[allow(clippy::unwrap_used, clippy::expect_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn provider(config: VoxCpmTtsConfig) -> VoxCpmTts {
        VoxCpmTts::new(&config)
    }

    #[test]
    fn metadata_is_stable() {
        let tts = provider(VoxCpmTtsConfig::default());
        assert_eq!(tts.id(), "voxcpm");
        assert_eq!(tts.name(), "VoxCPM");
        assert!(!tts.supports_ssml());
    }

    #[test]
    fn defaults_are_not_considered_configured() {
        assert!(!provider(VoxCpmTtsConfig::default()).is_configured());
    }

    #[test]
    fn explicit_model_marks_configured() {
        let tts = provider(VoxCpmTtsConfig {
            model: Some("openbmb/VoxCPM2".into()),
            ..Default::default()
        });
        assert!(tts.is_configured());
    }

    #[test]
    fn custom_endpoint_marks_configured() {
        let tts = provider(VoxCpmTtsConfig {
            endpoint: "http://10.0.0.5:8000/v1".into(),
            ..Default::default()
        });
        assert!(tts.is_configured());
    }

    #[test]
    fn endpoint_trailing_slash_is_trimmed() {
        let tts = provider(VoxCpmTtsConfig {
            endpoint: "http://localhost:8000/v1/".into(),
            ..Default::default()
        });
        // Trimming must also keep the default recognisable, so a slash-only
        // difference does not silently look like a custom endpoint.
        assert_eq!(tts.endpoint, DEFAULT_ENDPOINT);
        assert!(!tts.is_configured());
    }

    #[test]
    fn blank_endpoint_falls_back_to_default() {
        let tts = provider(VoxCpmTtsConfig {
            endpoint: "   ".into(),
            ..Default::default()
        });
        assert_eq!(tts.endpoint, DEFAULT_ENDPOINT);
    }

    #[test]
    fn blank_voice_and_model_are_treated_as_unset() {
        let tts = provider(VoxCpmTtsConfig {
            model: Some("  ".into()),
            voice: Some(String::new()),
            ..Default::default()
        });
        assert!(tts.model.is_none());
        assert!(tts.voice.is_none());
        assert!(!tts.is_configured());
    }

    #[test]
    fn unsupported_formats_report_what_was_actually_returned() {
        // AAC and WebM are not served by vLLM-Omni; the reported format must
        // match the bytes we asked for, not the bytes we wanted.
        assert_eq!(
            VoxCpmTts::negotiate_format(AudioFormat::Aac),
            ("mp3", AudioFormat::Mp3)
        );
        assert_eq!(
            VoxCpmTts::negotiate_format(AudioFormat::Webm),
            ("opus", AudioFormat::Opus)
        );
        assert_eq!(
            VoxCpmTts::negotiate_format(AudioFormat::Wav),
            ("wav", AudioFormat::Wav)
        );
        assert_eq!(
            VoxCpmTts::negotiate_format(AudioFormat::Pcm),
            ("pcm", AudioFormat::Pcm)
        );
    }

    #[test]
    fn persona_block_becomes_a_single_design_clause() {
        let rendered = "Persona: Alfred\nProfile: A wise British butler\nStyle: Dry wit";
        assert_eq!(
            flatten_voice_design(rendered),
            "Persona: Alfred, Profile: A wise British butler, Style: Dry wit"
        );
    }

    #[test]
    fn design_clause_drops_inner_parentheses() {
        // Unbalanced parentheses would break VoxCPM's prefix delimiter.
        assert_eq!(
            flatten_voice_design("Profile: A butler (retired)"),
            "Profile: A butler retired"
        );
    }

    #[test]
    fn voice_design_prefixes_the_text() {
        let out = apply_voice_design("Good evening.", Some("Style: Dry wit"), true);
        assert_eq!(out, "(Style: Dry wit)Good evening.");
    }

    #[test]
    fn voice_design_can_be_disabled() {
        let out = apply_voice_design("Good evening.", Some("Style: Dry wit"), false);
        assert_eq!(out, "Good evening.");
    }

    #[test]
    fn voice_design_is_skipped_without_instructions() {
        assert_eq!(
            apply_voice_design("Good evening.", None, true),
            "Good evening."
        );
    }

    #[test]
    fn existing_prefix_is_not_double_wrapped() {
        let out = apply_voice_design("(A calm narrator)Hello.", Some("Style: Dry wit"), true);
        assert_eq!(out, "(A calm narrator)Hello.");
    }

    #[test]
    fn whitespace_only_instructions_do_not_prefix() {
        assert_eq!(apply_voice_design("Hello.", Some("  \n "), true), "Hello.");
    }

    #[test]
    fn voices_response_merges_uploaded_metadata() {
        let raw = r#"{
            "voices": ["alice", "preset_one"],
            "uploaded_voices": [
                {"name": "alice", "speaker_description": "warm narrator"}
            ]
        }"#;
        let parsed: VoicesResponse =
            serde_json::from_str(raw).expect("voices payload should deserialize");
        let voices = parsed.into_voices();

        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].id, "alice");
        assert_eq!(voices[0].description.as_deref(), Some("warm narrator"));
        assert_eq!(voices[1].id, "preset_one");
        assert!(voices[1].description.is_none());
    }

    #[test]
    fn voices_response_falls_back_to_reference_transcript() {
        let raw = r#"{
            "voices": ["bob"],
            "uploaded_voices": [{"name": "bob", "ref_text": "hello there"}]
        }"#;
        let parsed: VoicesResponse =
            serde_json::from_str(raw).expect("voices payload should deserialize");
        let voices = parsed.into_voices();
        assert_eq!(
            voices[0].description.as_deref(),
            Some("Cloned from: hello there")
        );
    }

    #[test]
    fn voices_response_tolerates_missing_uploaded_list() {
        let parsed: VoicesResponse = serde_json::from_str(r#"{"voices":["x"]}"#)
            .expect("uploaded_voices should be optional");
        assert_eq!(parsed.into_voices().len(), 1);
    }

    #[test]
    fn request_omits_voice_and_model_when_unset() {
        let body = SpeechRequest {
            model: None,
            input: "hi",
            voice: None,
            response_format: "wav",
            speed: None,
        };
        let json = serde_json::to_string(&body).expect("request should serialize");
        // Omitting `voice` is what selects zero-shot synthesis; sending a
        // placeholder like "alloy" would be rejected by VoxCPM.
        assert_eq!(json, r#"{"input":"hi","response_format":"wav"}"#);
    }
}
