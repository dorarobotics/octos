//! Admin API handlers for profile and gateway management.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use chrono::Utc;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::AppState;
use crate::profiles::{ProfileConfig, UserProfile, mask_secrets};

// ── Request / Response types ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateProfileRequest {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub data_dir: Option<String>,
    #[serde(default)]
    pub config: ProfileConfig,
}

#[derive(Deserialize)]
pub struct UpdateProfileRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub data_dir: Option<Option<String>>,
    #[serde(default)]
    pub config: Option<ProfileConfig>,
}

#[derive(Serialize)]
pub struct ProfileResponse {
    #[serde(flatten)]
    pub profile: UserProfile,
    pub status: crate::process_manager::ProcessStatus,
}

#[derive(Serialize)]
pub struct OverviewResponse {
    pub total_profiles: usize,
    pub running: usize,
    pub stopped: usize,
    pub profiles: Vec<ProfileResponse>,
}

#[derive(Serialize)]
pub struct ActionResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Serialize)]
pub struct BulkActionResponse {
    pub ok: bool,
    pub count: usize,
}

// ── Handlers ──────────────────────────────────────────────────────────

/// GET /api/admin/overview
pub async fn overview(
    State(state): State<Arc<AppState>>,
) -> Result<Json<OverviewResponse>, (StatusCode, String)> {
    let profiles = state
        .profile_store
        .as_ref()
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "admin not configured".into(),
        ))?
        .list()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let mut running = 0;
    let mut items = Vec::with_capacity(profiles.len());
    for p in profiles {
        let status = pm.status(&p.id).await;
        if status.running {
            running += 1;
        }
        items.push(ProfileResponse {
            profile: mask_secrets(&p),
            status,
        });
    }

    let total = items.len();
    Ok(Json(OverviewResponse {
        total_profiles: total,
        running,
        stopped: total - running,
        profiles: items,
    }))
}

/// GET /api/admin/profiles
pub async fn list_profiles(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ProfileResponse>>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let profiles = store
        .list()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut items = Vec::with_capacity(profiles.len());
    for p in profiles {
        let status = pm.status(&p.id).await;
        items.push(ProfileResponse {
            profile: mask_secrets(&p),
            status,
        });
    }
    Ok(Json(items))
}

/// GET /api/admin/profiles/:id
pub async fn get_profile(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ProfileResponse>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let profile = store
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, format!("profile '{id}' not found")))?;

    let status = pm.status(&id).await;
    Ok(Json(ProfileResponse {
        profile: mask_secrets(&profile),
        status,
    }))
}

/// POST /api/admin/profiles
pub async fn create_profile(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateProfileRequest>,
) -> Result<(StatusCode, Json<ProfileResponse>), (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    // Check for duplicates
    if store
        .get(&req.id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .is_some()
    {
        return Err((
            StatusCode::CONFLICT,
            format!("profile '{}' already exists", req.id),
        ));
    }

    let now = Utc::now();
    let profile = UserProfile {
        id: req.id,
        name: req.name,
        enabled: req.enabled,
        data_dir: req.data_dir,
        parent_id: None,
        config: req.config,
        created_at: now,
        updated_at: now,
    };

    store
        .save(&profile)
        .map_err(|e| {
            tracing::error!(profile = %profile.id, error = %e, "failed to create profile");
            (StatusCode::BAD_REQUEST, e.to_string())
        })?;

    tracing::info!(profile = %profile.id, name = %profile.name, "profile created");
    let status = pm.status(&profile.id).await;
    Ok((
        StatusCode::CREATED,
        Json(ProfileResponse {
            profile: mask_secrets(&profile),
            status,
        }),
    ))
}

/// PUT /api/admin/profiles/:id
pub async fn update_profile(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: String,
) -> Result<Json<ProfileResponse>, (StatusCode, String)> {
    let req: UpdateProfileRequest = serde_json::from_str(&body).map_err(|e| {
        tracing::warn!(profile_id = %id, error = %e, body = %body, "failed to parse profile update request");
        (StatusCode::BAD_REQUEST, format!("Invalid request body: {e}"))
    })?;
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let mut profile = store
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, format!("profile '{id}' not found")))?;

    if let Some(name) = req.name {
        profile.name = name;
    }
    if let Some(enabled) = req.enabled {
        profile.enabled = enabled;
    }
    if let Some(data_dir) = req.data_dir {
        profile.data_dir = data_dir;
    }
    if let Some(new_config) = req.config {
        profile.config = new_config;
    }
    profile.updated_at = Utc::now();

    store
        .save_with_merge(&mut profile)
        .map_err(|e| {
            tracing::error!(profile = %id, error = %e, "failed to update profile");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    tracing::info!(profile = %id, "profile updated");
    let status = pm.status(&id).await;
    Ok(Json(ProfileResponse {
        profile: mask_secrets(&profile),
        status,
    }))
}

/// DELETE /api/admin/profiles/:id
pub async fn delete_profile(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ActionResponse>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    // Stop the gateway if running
    let _ = pm.stop(&id).await;

    // Cascade: stop and delete all sub-accounts
    if let Ok(subs) = store.list_sub_accounts(&id) {
        for sub in &subs {
            let _ = pm.stop(&sub.id).await;
            let _ = store.delete(&sub.id);
        }
    }

    let deleted = store
        .delete(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !deleted {
        return Err((StatusCode::NOT_FOUND, format!("profile '{id}' not found")));
    }

    tracing::info!(profile = %id, "profile deleted");
    Ok(Json(ActionResponse {
        ok: true,
        message: Some(format!("profile '{id}' deleted")),
    }))
}

/// POST /api/admin/profiles/:id/start
pub async fn start_gateway(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ActionResponse>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let profile = store
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, format!("profile '{id}' not found")))?;

    // Validate LLM provider is configured (resolve inheritance for sub-accounts)
    let effective = crate::profiles::resolve_effective_profile(store, &profile)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if effective.config.provider.is_none() && effective.config.model.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Cannot start: LLM provider must be configured first".into(),
        ));
    }

    if let Err(e) = pm.start(&profile).await {
        tracing::error!(profile = %id, error = %e, "admin gateway failed to start");
        return Err((StatusCode::CONFLICT, e.to_string()));
    }

    tracing::info!(profile = %id, "admin gateway started");
    Ok(Json(ActionResponse {
        ok: true,
        message: Some(format!("gateway '{id}' started")),
    }))
}

/// POST /api/admin/profiles/:id/stop
pub async fn stop_gateway(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ActionResponse>, (StatusCode, String)> {
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let stopped = pm
        .stop(&id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !stopped {
        tracing::warn!(profile = %id, "stop requested but gateway not running");
        return Err((
            StatusCode::NOT_FOUND,
            format!("gateway '{id}' is not running"),
        ));
    }

    tracing::info!(profile = %id, "admin gateway stopped");
    Ok(Json(ActionResponse {
        ok: true,
        message: Some(format!("gateway '{id}' stopped")),
    }))
}

/// POST /api/admin/profiles/:id/restart
pub async fn restart_gateway(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ActionResponse>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let profile = store
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, format!("profile '{id}' not found")))?;

    if let Err(e) = pm.restart(&profile).await {
        tracing::error!(profile = %id, error = %e, "admin gateway failed to restart");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
    }

    tracing::info!(profile = %id, "admin gateway restarted");
    Ok(Json(ActionResponse {
        ok: true,
        message: Some(format!("gateway '{id}' restarted")),
    }))
}

/// GET /api/admin/profiles/:id/status
pub async fn gateway_status(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<crate::process_manager::ProcessStatus>, (StatusCode, String)> {
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    Ok(Json(pm.status(&id).await))
}

/// GET /api/admin/profiles/:id/metrics — Provider QoS metrics.
pub async fn provider_metrics(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    match pm.read_metrics(&id).await {
        Some(metrics) => Ok(Json(metrics)),
        None => Ok(Json(serde_json::json!(null))),
    }
}

/// GET /api/admin/profiles/:id/logs — SSE log stream.
pub async fn gateway_logs(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<
    Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>,
    (StatusCode, String),
> {
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    // Get buffered history first, then subscribe for live logs.
    let history = pm.log_history(&id).await;
    let rx = pm.subscribe_logs(&id).await.ok_or((
        StatusCode::NOT_FOUND,
        format!("gateway '{id}' is not running"),
    ))?;

    // Emit history lines first, then stream live.
    let history_stream = futures::stream::iter(
        history
            .into_iter()
            .map(|line| Ok(Event::default().data(line))),
    );
    let live_stream = futures::stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok(line) => {
                    let event: Result<Event, std::convert::Infallible> =
                        Ok(Event::default().data(line));
                    return Some((event, rx));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
            }
        }
    });

    Ok(Sse::new(history_stream.chain(live_stream)).keep_alive(KeepAlive::default()))
}

/// GET /api/admin/profiles/:id/whatsapp/qr
pub async fn whatsapp_qr(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<crate::process_manager::BridgeQrInfo>, (StatusCode, String)> {
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let info = pm.bridge_qr(&id).await.ok_or((
        StatusCode::NOT_FOUND,
        format!("no managed WhatsApp bridge for '{id}'"),
    ))?;

    Ok(Json(info))
}

/// POST /api/admin/test-provider or /api/my/test-provider
///
/// Verify an LLM provider/model/key combo works. Accepts either:
/// - `api_key`: raw key (for newly entered, unsaved keys)
/// - `api_key_env`: env var name to resolve from the user's saved profile
///   (used when the key is already saved and the frontend only has the masked value)
pub async fn test_provider(
    State(state): State<Arc<AppState>>,
    identity: Option<axum::Extension<super::router::AuthIdentity>>,
    Json(req): Json<TestProviderRequest>,
) -> Result<Json<TestProviderResponse>, (StatusCode, String)> {
    use crew_core::{Message, MessageRole};
    use crew_llm::{ChatConfig, LlmProvider};

    // Resolve the API key: prefer raw api_key, fall back to reading from saved profile
    let api_key = if let Some(ref key) = req.api_key {
        if !key.is_empty() && !key.contains("***") {
            key.clone()
        } else {
            resolve_saved_key(&state, &identity, &req)?
        }
    } else {
        resolve_saved_key(&state, &identity, &req)?
    };

    if api_key.is_empty() {
        return Ok(Json(TestProviderResponse {
            ok: false,
            message: String::new(),
            error: Some("No API key provided".into()),
        }));
    }

    let provider: Arc<dyn LlmProvider> = {
        let params = crew_llm::registry::CreateParams {
            api_key: Some(api_key.clone()),
            model: Some(req.model.clone()),
            base_url: req.base_url.clone(),
            model_hints: None,
        };
        match crew_llm::registry::lookup(&req.provider) {
            Some(entry) => (entry.create)(params)
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("provider error: {e:#}")))?,
            None => {
                // Unknown provider — assume OpenAI-compatible with custom base URL.
                let url = req
                    .base_url
                    .as_deref()
                    .unwrap_or("https://api.openai.com/v1");
                Arc::new(
                    crew_llm::openai::OpenAIProvider::new(&api_key, &req.model).with_base_url(url),
                )
            }
        }
    };

    let messages = vec![Message {
        role: MessageRole::User,
        content: "Say OK".into(),
        media: vec![],
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
        timestamp: chrono::Utc::now(),
    }];
    // Gemini 2.5+ "thinking" models consume tokens on internal reasoning,
    // so 16 tokens is too small — they return empty content.  Use 128 for
    // Gemini and keep 16 for everyone else (fast, cheap connectivity check).
    let max_tokens = if req.provider == "gemini" { 128 } else { 16 };
    let config = ChatConfig {
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..Default::default()
    };

    match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        provider.chat(&messages, &[], &config),
    )
    .await
    {
        Ok(Ok(resp)) => {
            tracing::info!(provider = %req.provider, model = %req.model, "test-provider succeeded");
            Ok(Json(TestProviderResponse {
                ok: true,
                message: resp.content.unwrap_or_default(),
                error: None,
            }))
        }
        Ok(Err(e)) => {
            tracing::warn!(provider = %req.provider, model = %req.model, error = %e, "test-provider failed");
            Ok(Json(TestProviderResponse {
                ok: false,
                message: String::new(),
                error: Some(format!("{e:#}")),
            }))
        }
        Err(_) => {
            tracing::warn!(provider = %req.provider, model = %req.model, "test-provider timed out");
            Ok(Json(TestProviderResponse {
                ok: false,
                message: String::new(),
                error: Some("Request timed out after 30 seconds".into()),
            }))
        }
    }
}

/// Resolve an API key from the user's saved profile by env var name.
fn resolve_saved_key(
    state: &AppState,
    identity: &Option<axum::Extension<super::router::AuthIdentity>>,
    req: &TestProviderRequest,
) -> Result<String, (StatusCode, String)> {
    let env_name = match &req.api_key_env {
        Some(name) if !name.is_empty() => name,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "No api_key or api_key_env provided".into(),
            ));
        }
    };

    // Get the user's profile from the store
    let ps = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "profile store not configured".into(),
    ))?;

    let profile_id = match identity {
        Some(axum::Extension(super::router::AuthIdentity::User { id, .. })) => id.clone(),
        Some(axum::Extension(super::router::AuthIdentity::Admin)) => {
            // Admin: resolve from first profile (same as /api/my/* endpoints)
            let profiles = ps
                .list()
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            profiles
                .first()
                .map(|p| p.id.clone())
                .ok_or((StatusCode::NOT_FOUND, "no profiles exist".into()))?
        }
        None => {
            return Err((StatusCode::UNAUTHORIZED, "not authenticated".into()));
        }
    };

    let profile = ps
        .get(&profile_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "profile not found".into()))?;

    Ok(profile
        .config
        .env_vars
        .get(env_name)
        .cloned()
        .unwrap_or_default())
}

#[derive(Deserialize)]
pub struct TestProviderRequest {
    /// Native provider name: "anthropic", "openai", "gemini", "openrouter"
    pub provider: String,
    pub model: String,
    /// Raw API key (for new/unsaved keys).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Env var name to resolve from saved profile (for already-saved keys).
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Serialize)]
pub struct TestProviderResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// POST /api/my/test-search
///
/// Verify a web search API key works. Makes a minimal search request.
pub async fn test_search(
    State(state): State<Arc<AppState>>,
    identity: Option<axum::Extension<super::router::AuthIdentity>>,
    Json(req): Json<TestSearchRequest>,
) -> Result<Json<TestSearchResponse>, (StatusCode, String)> {
    // Resolve the API key
    let api_key = if let Some(ref key) = req.api_key {
        if !key.is_empty() && !key.contains("***") {
            key.clone()
        } else {
            resolve_saved_search_key(&state, &identity, &req)?
        }
    } else {
        resolve_saved_search_key(&state, &identity, &req)?
    };

    if api_key.is_empty() {
        return Ok(Json(TestSearchResponse {
            ok: false,
            message: String::new(),
            error: Some("No API key provided".into()),
        }));
    }

    let client = reqwest::Client::new();
    let query = "test";

    let result = match req.provider.as_str() {
        "perplexity" => {
            let body = serde_json::json!({
                "model": "sonar",
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 32
            });
            let resp = client
                .post("https://api.perplexity.ai/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
            if resp.status().is_success() {
                Ok("Perplexity Sonar API connected successfully".to_string())
            } else {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                Err(format!("Perplexity API error ({status}): {body}"))
            }
        }
        "brave" => {
            let resp = client
                .get("https://api.search.brave.com/res/v1/web/search")
                .header("X-Subscription-Token", &api_key)
                .header("Accept", "application/json")
                .query(&[("q", query), ("count", "1")])
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
            if resp.status().is_success() {
                Ok("Brave Search API connected successfully".to_string())
            } else {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                Err(format!("Brave Search API error ({status}): {body}"))
            }
        }
        "you" => {
            let resp = client
                .get("https://ydc-index.io/v1/search")
                .header("X-API-Key", &api_key)
                .query(&[("query", query), ("count", "1")])
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
            if resp.status().is_success() {
                Ok("You.com Search API connected successfully".to_string())
            } else {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                Err(format!("You.com API error ({status}): {body}"))
            }
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Unknown search provider: {other}"),
            ));
        }
    };

    match result {
        Ok(msg) => Ok(Json(TestSearchResponse {
            ok: true,
            message: msg,
            error: None,
        })),
        Err(err) => Ok(Json(TestSearchResponse {
            ok: false,
            message: String::new(),
            error: Some(err),
        })),
    }
}

fn resolve_saved_search_key(
    state: &AppState,
    identity: &Option<axum::Extension<super::router::AuthIdentity>>,
    req: &TestSearchRequest,
) -> Result<String, (StatusCode, String)> {
    let env_name = match &req.api_key_env {
        Some(name) if !name.is_empty() => name,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "No api_key or api_key_env provided".into(),
            ));
        }
    };

    let ps = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "profile store not configured".into(),
    ))?;

    let profile_id = match identity {
        Some(axum::Extension(super::router::AuthIdentity::User { id, .. })) => id.clone(),
        Some(axum::Extension(super::router::AuthIdentity::Admin)) => {
            // Admin: resolve from first profile (same as /api/my/* endpoints)
            let profiles = ps
                .list()
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            profiles
                .first()
                .map(|p| p.id.clone())
                .ok_or((StatusCode::NOT_FOUND, "no profiles exist".into()))?
        }
        None => {
            return Err((StatusCode::UNAUTHORIZED, "not authenticated".into()));
        }
    };

    let profile = ps
        .get(&profile_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "profile not found".into()))?;

    Ok(profile
        .config
        .env_vars
        .get(env_name)
        .cloned()
        .unwrap_or_default())
}

#[derive(Deserialize)]
pub struct TestSearchRequest {
    /// Search provider: "perplexity", "brave", "you"
    pub provider: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
}

#[derive(Serialize)]
pub struct TestSearchResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// POST /api/admin/start-all
pub async fn start_all(
    State(state): State<Arc<AppState>>,
) -> Result<Json<BulkActionResponse>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let profiles = store
        .list()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    tracing::info!("start-all requested");
    let mut started = 0;
    for p in &profiles {
        if p.enabled {
            match pm.start(p).await {
                Ok(()) => started += 1,
                Err(e) => tracing::warn!(profile = %p.id, error = %e, "start-all: failed to start"),
            }
        }
    }

    tracing::info!(count = started, "start-all completed");
    Ok(Json(BulkActionResponse {
        ok: true,
        count: started,
    }))
}

/// POST /api/admin/stop-all
pub async fn stop_all(
    State(state): State<Arc<AppState>>,
) -> Result<Json<BulkActionResponse>, (StatusCode, String)> {
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    tracing::info!("stop-all requested");
    let count = pm.stop_all().await;
    tracing::info!(count = count, "stop-all completed");
    Ok(Json(BulkActionResponse { ok: true, count }))
}

// ── Sub-account endpoints ────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateSubAccountRequest {
    pub name: String,
    #[serde(default)]
    pub channels: Vec<crate::profiles::ChannelCredentials>,
    #[serde(default)]
    pub gateway: Option<crate::profiles::GatewaySettings>,
    #[serde(default)]
    pub env_vars: std::collections::HashMap<String, String>,
}

/// GET /api/admin/profiles/:id/accounts — List sub-accounts for a profile.
pub async fn list_sub_accounts(
    State(state): State<Arc<AppState>>,
    Path(parent_id): Path<String>,
) -> Result<Json<Vec<ProfileResponse>>, (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let subs = store
        .list_sub_accounts(&parent_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut items = Vec::with_capacity(subs.len());
    for s in subs {
        let status = pm.status(&s.id).await;
        items.push(ProfileResponse {
            profile: mask_secrets(&s),
            status,
        });
    }
    Ok(Json(items))
}

/// POST /api/admin/profiles/:id/accounts — Create a sub-account.
pub async fn create_sub_account(
    State(state): State<Arc<AppState>>,
    Path(parent_id): Path<String>,
    Json(req): Json<CreateSubAccountRequest>,
) -> Result<(StatusCode, Json<ProfileResponse>), (StatusCode, String)> {
    let store = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;
    let pm = state.process_manager.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let mut sub = store
        .create_sub_account(
            &parent_id,
            &req.name,
            req.channels,
            req.gateway.unwrap_or_default(),
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Set channel-specific env vars if provided
    if !req.env_vars.is_empty() {
        sub.config.env_vars = req.env_vars;
        sub.updated_at = Utc::now();
        store
            .save(&sub)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    let status = pm.status(&sub.id).await;
    Ok((
        StatusCode::CREATED,
        Json(ProfileResponse {
            profile: mask_secrets(&sub),
            status,
        }),
    ))
}
