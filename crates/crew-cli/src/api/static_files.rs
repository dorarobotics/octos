//! Embedded static file serving for the built-in Web UI.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use rust_embed::Embed;

use super::AppState;

#[derive(Embed)]
#[folder = "static/"]
struct Assets;

/// Fallback handler: serves embedded static files, falls back to admin/index.html for SPA routing.
/// The admin dashboard SPA handles all UI routes (login, profiles, users, etc.).
pub async fn static_handler(State(_state): State<Arc<AppState>>, uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() {
        "admin/index.html"
    } else {
        path
    };

    // Try the exact path first, then under admin/ prefix (dashboard assets)
    if let Some(file) = Assets::get(path) {
        return serve_file(path, &file.data);
    }
    let admin_path = format!("admin/{path}");
    if let Some(file) = Assets::get(&admin_path) {
        return serve_file(&admin_path, &file.data);
    }

    // SPA fallback: serve admin/index.html for client-side routing
    match Assets::get("admin/index.html") {
        Some(file) => serve_file("admin/index.html", &file.data),
        None => (StatusCode::NOT_FOUND, "Not Found").into_response(),
    }
}

fn serve_file(path: &str, data: &[u8]) -> Response {
    let mime = match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, mime)],
        data.to_vec(),
    )
        .into_response()
}
