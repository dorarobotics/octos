//! Authentication and user self-service API handlers.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::AppState;
use super::admin::ProfileResponse;
use crate::profiles::mask_secrets;
use crate::user_store::{User, UserRole};

use super::router::AuthIdentity;

// ── Request / Response types ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct SendCodeRequest {
    pub email: String,
}

#[derive(Serialize)]
pub struct SendCodeResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Deserialize)]
pub struct VerifyRequest {
    pub email: String,
    pub code: String,
}

#[derive(Serialize)]
pub struct VerifyResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<User>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Serialize)]
pub struct MeResponse {
    pub user: User,
    pub profile: Option<ProfileResponse>,
}

#[derive(Serialize)]
pub struct ActionResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

// ── Public auth endpoints (no auth required) ──────────────────────────

/// POST /api/auth/send-code
pub async fn send_code(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SendCodeRequest>,
) -> Result<Json<SendCodeResponse>, StatusCode> {
    let auth_mgr = state
        .auth_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    tracing::info!(email = %req.email, "login OTP requested");
    match auth_mgr.send_otp(&req.email).await {
        Ok(true) => Ok(Json(SendCodeResponse {
            ok: true,
            message: Some("Verification code sent to your email".into()),
        })),
        Ok(false) => Ok(Json(SendCodeResponse {
            ok: true, // Don't reveal rate-limit state to prevent enumeration
            message: Some("Verification code sent to your email".into()),
        })),
        Err(e) => {
            // Log but don't leak internal errors
            tracing::warn!(error = %e, "send_otp failed");
            Ok(Json(SendCodeResponse {
                ok: true,
                message: Some("Verification code sent to your email".into()),
            }))
        }
    }
}

/// POST /api/auth/verify
pub async fn verify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, StatusCode> {
    let auth_mgr = state
        .auth_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    match auth_mgr.verify_otp(&req.email, &req.code).await {
        Ok(Some(token)) => {
            tracing::info!(email = %req.email, "user logged in");
            // Get the user to return
            let user_store = state
                .user_store
                .as_ref()
                .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
            let user = user_store
                .get_by_email(&req.email)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            // Auto-create profile if user has none
            if let Some(ref user) = user {
                if let Some(ref profile_store) = state.profile_store {
                    if profile_store.get(&user.id).unwrap_or(None).is_none() {
                        let profile = crate::profiles::UserProfile {
                            id: user.id.clone(),
                            name: user.name.clone(),
                            enabled: false,
                            data_dir: None,
                            parent_id: None,
                            config: crate::profiles::ProfileConfig::default(),
                            created_at: chrono::Utc::now(),
                            updated_at: chrono::Utc::now(),
                        };
                        if let Err(e) = profile_store.save(&profile) {
                            tracing::warn!(user_id = %user.id, error = %e, "failed to auto-create profile");
                        }
                    }
                }
            }

            Ok(Json(VerifyResponse {
                ok: true,
                token: Some(token),
                user,
                message: None,
            }))
        }
        Ok(None) => Ok(Json(VerifyResponse {
            ok: false,
            token: None,
            user: None,
            message: Some("Invalid or expired code".into()),
        })),
        Err(e) => {
            tracing::warn!(error = %e, "verify_otp error");
            Ok(Json(VerifyResponse {
                ok: false,
                token: None,
                user: None,
                message: Some("Invalid or expired code".into()),
            }))
        }
    }
}

/// POST /api/auth/logout
pub async fn logout(
    State(state): State<Arc<AppState>>,
    req: axum::http::Request<axum::body::Body>,
) -> Result<Json<ActionResponse>, StatusCode> {
    let auth_mgr = state
        .auth_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    if let Some(token) = extract_bearer_token(&req) {
        auth_mgr.revoke_session(&token).await;
        tracing::info!("user logged out");
    }

    Ok(Json(ActionResponse {
        ok: true,
        message: None,
    }))
}

/// GET /api/auth/me
pub async fn me(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<MeResponse>, StatusCode> {
    // Handle admin token first — no user_store needed
    if matches!(&identity, AuthIdentity::Admin) {
        return Ok(Json(MeResponse {
            user: User {
                id: "admin".into(),
                email: "admin@localhost".into(),
                name: "Admin".into(),
                role: UserRole::Admin,
                created_at: chrono::Utc::now(),
                last_login_at: None,
            },
            profile: None,
        }));
    }

    let user_id = match &identity {
        AuthIdentity::Admin => unreachable!(),
        AuthIdentity::User { id, .. } => id.clone(),
    };

    let user_store = state
        .user_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let user = user_store
        .get(&user_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let profile = if let Some(ref ps) = state.profile_store {
        if let Ok(Some(p)) = ps.get(&user.id) {
            let status = if let Some(ref pm) = state.process_manager {
                pm.status(&p.id).await
            } else {
                crate::process_manager::ProcessStatus {
                    running: false,
                    pid: None,
                    started_at: None,
                    uptime_secs: None,
                }
            };
            Some(ProfileResponse {
                profile: mask_secrets(&p),
                status,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(Json(MeResponse { user, profile }))
}

// ── User self-service endpoints (/api/my/*) ───────────────────────────

/// GET /api/my/profile
pub async fn my_profile(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<ProfileResponse>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let profile = resolve_my_profile(&identity, ps)?;

    let status = if let Some(ref pm) = state.process_manager {
        pm.status(&profile.id).await
    } else {
        crate::process_manager::ProcessStatus {
            running: false,
            pid: None,
            started_at: None,
            uptime_secs: None,
        }
    };

    Ok(Json(ProfileResponse {
        profile: mask_secrets(&profile),
        status,
    }))
}

/// PUT /api/my/profile
pub async fn update_my_profile(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
    body: String,
) -> Result<Json<ProfileResponse>, (StatusCode, String)> {
    let req: super::admin::UpdateProfileRequest = serde_json::from_str(&body).map_err(|e| {
        tracing::warn!(error = %e, body = %body, "failed to parse my profile update request");
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid request body: {e}"),
        )
    })?;
    let ps = state.profile_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "admin not configured".into(),
    ))?;

    let mut profile =
        resolve_my_profile(&identity, ps).map_err(|s| (s, "profile not found".into()))?;

    // Apply updates (same logic as admin::update_profile but scoped)
    if let Some(name) = req.name {
        profile.name = name;
    }
    if let Some(enabled) = req.enabled {
        profile.enabled = enabled;
    }
    if let Some(config) = req.config {
        profile.config = config;
    }
    profile.updated_at = chrono::Utc::now();

    ps.save_with_merge(&mut profile).map_err(|e| {
        tracing::error!(profile = %profile.id, error = %e, "failed to save user profile");
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    tracing::info!(profile = %profile.id, "user profile updated");
    let status = if let Some(ref pm) = state.process_manager {
        pm.status(&profile.id).await
    } else {
        crate::process_manager::ProcessStatus {
            running: false,
            pid: None,
            started_at: None,
            uptime_secs: None,
        }
    };

    Ok(Json(ProfileResponse {
        profile: mask_secrets(&profile),
        status,
    }))
}

/// POST /api/my/profile/start
pub async fn start_my_gateway(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<ActionResponse>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let profile = resolve_my_profile(&identity, ps)?;

    // Validate LLM provider is configured
    if profile.config.provider.is_none() && profile.config.model.is_none() {
        return Ok(Json(ActionResponse {
            ok: false,
            message: Some("Cannot start: LLM provider must be configured first".into()),
        }));
    }

    match pm.start(&profile).await {
        Ok(()) => {
            tracing::info!(profile = %profile.id, "user gateway started");
            Ok(Json(ActionResponse {
                ok: true,
                message: None,
            }))
        }
        Err(e) => {
            tracing::error!(profile = %profile.id, error = %e, "user gateway failed to start");
            Ok(Json(ActionResponse {
                ok: false,
                message: Some(e.to_string()),
            }))
        }
    }
}

/// POST /api/my/profile/stop
pub async fn stop_my_gateway(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<ActionResponse>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let profile_id = resolve_my_profile_id(&identity, ps)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let stopped = pm.stop(&profile_id).await.unwrap_or(false);
    if stopped {
        tracing::info!(profile = %profile_id, "user gateway stopped");
        Ok(Json(ActionResponse {
            ok: true,
            message: None,
        }))
    } else {
        tracing::warn!(profile = %profile_id, "user stop requested but gateway not running");
        Ok(Json(ActionResponse {
            ok: false,
            message: Some("Gateway not running".into()),
        }))
    }
}

/// POST /api/my/profile/restart
pub async fn restart_my_gateway(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<ActionResponse>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let profile = resolve_my_profile(&identity, ps)?;

    match pm.restart(&profile).await {
        Ok(()) => {
            tracing::info!(profile = %profile.id, "user gateway restarted");
            Ok(Json(ActionResponse {
                ok: true,
                message: None,
            }))
        }
        Err(e) => {
            tracing::error!(profile = %profile.id, error = %e, "user gateway failed to restart");
            Ok(Json(ActionResponse {
                ok: false,
                message: Some(e.to_string()),
            }))
        }
    }
}

/// GET /api/my/profile/status
pub async fn my_gateway_status(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<crate::process_manager::ProcessStatus>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let profile_id = resolve_my_profile_id(&identity, ps)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    Ok(Json(pm.status(&profile_id).await))
}

/// GET /api/my/profile/logs
pub async fn my_gateway_logs(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let profile_id = resolve_my_profile_id(&identity, ps)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    // Get buffered history first, then subscribe for live logs.
    let history = pm.log_history(&profile_id).await;
    let rx = pm
        .subscribe_logs(&profile_id)
        .await
        .ok_or(StatusCode::NOT_FOUND)?;

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

/// GET /api/my/profile/whatsapp/qr
pub async fn my_whatsapp_qr(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<crate::process_manager::BridgeQrInfo>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let profile_id = resolve_my_profile_id(&identity, ps)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    pm.bridge_qr(&profile_id)
        .await
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

/// GET /api/my/profile/metrics
pub async fn my_provider_metrics(
    State(state): State<Arc<AppState>>,
    axum::Extension(identity): axum::Extension<AuthIdentity>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let ps = state
        .profile_store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let profile_id = resolve_my_profile_id(&identity, ps)?;
    let pm = state
        .process_manager
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    match pm.read_metrics(&profile_id).await {
        Some(metrics) => Ok(Json(metrics)),
        None => Ok(Json(serde_json::json!(null))),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Resolve the profile ID for "my" endpoints.
/// For regular users, returns their user ID. For admin token, returns the first profile's ID.
fn resolve_my_profile_id(
    identity: &AuthIdentity,
    ps: &crate::profiles::ProfileStore,
) -> Result<String, StatusCode> {
    match identity {
        AuthIdentity::Admin => {
            let profiles = ps.list().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            profiles
                .first()
                .map(|p| p.id.clone())
                .ok_or(StatusCode::NOT_FOUND)
        }
        AuthIdentity::User { id, .. } => Ok(id.clone()),
    }
}

/// Resolve the full profile for "my" endpoints.
fn resolve_my_profile(
    identity: &AuthIdentity,
    ps: &crate::profiles::ProfileStore,
) -> Result<crate::profiles::UserProfile, StatusCode> {
    let id = resolve_my_profile_id(identity, ps)?;
    ps.get(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)
}

fn extract_bearer_token(req: &axum::http::Request<axum::body::Body>) -> Option<String> {
    req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(String::from)
}
