use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, Query,
    },
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use regex::Regex;
use rsmpeg::{
    avcodec::AVCodecContext,
    avformat::AVFormatContextInput,
    avutil::AVFrame,
    ffi,
    swscale::SwsContext,
};
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::io::{Cursor, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tempfile::NamedTempFile;
use tower_http::cors::{Any, CorsLayer};

use tspine::export::{
    audio::AudioModem, html::HtmlExporter, image::ImageExporter, svg::SvgExporter,
    terminal::TerminalExporter,
};
use tspine::scanner::image_scan_u::UniversalScanner;
use tspine::scanner::image_scan::ImageScanner;
use tspine::{ColorMode, DecodedPayload, EccLevel, TSpineEncoder};

// ============================================================================
// Data Transfer Objects
// ============================================================================

#[derive(Deserialize, Debug)]
pub struct GenerateParams {
    pub text: Option<String>,
    pub public: Option<String>,
    pub private: Option<String>,
    pub format: Option<String>,
    pub colors: Option<String>,
    pub ecc: Option<String>,
    pub nano: Option<bool>,
    pub size: Option<usize>,
    pub min_header: Option<bool>,
    pub password: Option<String>,
    pub sign: Option<String>,
    pub mod_size: Option<u32>,
    pub margin: Option<u32>,
}

#[derive(Serialize)]
pub struct MetadataResponse {
    pub size: usize,
    pub raw_bytes: usize,
    pub packed_bytes: usize,
    pub total_cap_bytes: usize,
    pub ecc_bytes: usize,
    pub colors: usize,
    pub bits_per_cell: usize,
    pub is_dual: bool,
    pub is_encrypted: bool,
    pub is_signed: bool,
    pub is_nano: bool,
}

#[derive(Serialize, Clone, Debug)]
pub struct UniversalScanOutput {
    pub detected: bool,
    pub text: Option<String>,
    pub size: usize,
    pub is_nano: bool,
    pub color_mode: String,
    pub confidence: f64,
    pub corners: serde_json::Value,
    pub elapsed_ms: f64,
    pub error: Option<String>,
}

// ============================================================================
// Main Application & Routing
// ============================================================================

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        // Generator & classic decode routes
        .route("/", get(handle_generate))
        .route("/generate", get(handle_generate))
        .route("/decode", post(handle_decode))
        // Webpage at /scanner and /scanner/
        .route("/scanner", get(handle_scanner_page))
        .route("/scanner/", get(handle_scanner_page))
        // Universal Scanner API & WebSocket
        .route("/scan", post(handle_scan_upload).get(handle_scan_ws_or_redirect))
        .route("/scan/ws", get(handle_ws_upgrade))
        .route("/scanner/ws", get(handle_ws_upgrade))
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], 9999));
    println!("🚀 T-Spine Universal Server running on http://0.0.0.0:9999");
    println!("📷 Scanner Webpage UI available at: http://localhost:9999/scanner/");
    println!("📡 WebSocket streaming endpoint: ws://localhost:9999/scan/ws");
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ============================================================================
// Scanner Web Interface Handler
// ============================================================================

async fn handle_scanner_page() -> Response {
    Html(SCANNER_HTML_PAGE).into_response()
}

// ============================================================================
// REST API (/scan)
// ============================================================================

async fn handle_scan_ws_or_redirect(ws: Option<WebSocketUpgrade>) -> Response {
    if let Some(ws) = ws {
        ws.on_upgrade(handle_ws_stream)
    } else {
        axum::response::Redirect::to("/scanner/").into_response()
    }
}

async fn handle_ws_upgrade(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_ws_stream)
}

/// Handles manual image or video file uploads (PNG, JPEG, WebP, MP4, etc.)
async fn handle_scan_upload(mut multipart: Multipart) -> Response {
    let mut file_bytes = Vec::new();
    let mut filename = String::new();
    let mut password = None;
    let mut verify_key = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" || name == "image" || name == "video" {
            filename = field.file_name().unwrap_or("blob.bin").to_string();
            if let Ok(bytes) = field.bytes().await {
                file_bytes = bytes.to_vec();
            }
        } else if name == "password" {
            if let Ok(txt) = field.text().await {
                if !txt.is_empty() {
                    password = Some(txt);
                }
            }
        } else if name == "verify" {
            if let Ok(txt) = field.text().await {
                if !txt.is_empty() {
                    verify_key = Some(txt);
                }
            }
        }
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "detected": false, "error": "No file uploaded" })),
        )
            .into_response();
    }

    let is_video = filename.ends_with(".mp4")
        || filename.ends_with(".webm")
        || filename.ends_with(".mov")
        || filename.ends_with(".avi")
        || filename.ends_with(".mkv");

    let start = Instant::now();
    let pwd = password.clone();
    let vkey = verify_key.clone();

    // CPU-bound task offloaded to blocking thread pool
    let scan_result = tokio::task::spawn_blocking(move || {
        if is_video {
            scan_video_media(&file_bytes, pwd.as_deref(), vkey.as_deref())
        } else {
            scan_image_bytes(&file_bytes, pwd.as_deref(), vkey.as_deref())
        }
    })
    .await;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match scan_result {
        Ok(Ok(mut out)) => {
            out.elapsed_ms = elapsed;
            Json(out).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "detected": false,
                "confidence": 0.0,
                "elapsed_ms": elapsed,
                "error": e
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "detected": false,
                "confidence": 0.0,
                "elapsed_ms": elapsed,
                "error": e.to_string()
            })),
        )
            .into_response(),
    }
}

// ============================================================================
// WebSocket Streaming Handler (Optimized for High Concurrency)
// ============================================================================

async fn handle_ws_stream(socket: WebSocket) {
    let (mut ws_sink, mut ws_stream) = socket.split();

    // High efficiency: A bounded channel of 1 frame ensures that client frames are never
    // queued up behind slow CPU scans. If the worker is busy, stale frames drop immediately.
    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<(Vec<u8>, Option<String>, Option<String>)>(1);
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<String>(8);

    // Background sender task
    let send_task = tokio::spawn(async move {
        while let Some(msg_text) = result_rx.recv().await {
            if ws_sink.send(Message::Text(msg_text)).await.is_err() {
                break;
            }
        }
    });

    // Worker task executing UniversalScanner in spawn_blocking
    let worker_task = tokio::spawn(async move {
        while let Some((bytes, pwd, vkey)) = frame_rx.recv().await {
            let start = Instant::now();
            let res = tokio::task::spawn_blocking(move || {
                scan_image_bytes(&bytes, pwd.as_deref(), vkey.as_deref())
            })
            .await;

            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let response = match res {
                Ok(Ok(mut out)) => {
                    out.elapsed_ms = elapsed;
                    serde_json::to_string(&out).unwrap()
                }
                Ok(Err(err)) => serde_json::json!({
                    "detected": false,
                    "confidence": 0.0,
                    "elapsed_ms": elapsed,
                    "error": err
                })
                .to_string(),
                Err(je) => serde_json::json!({
                    "detected": false,
                    "confidence": 0.0,
                    "elapsed_ms": elapsed,
                    "error": je.to_string()
                })
                .to_string(),
            };

            if result_tx.send(response).await.is_err() {
                break;
            }
        }
    });

    // Ingress reader loop
    let mut password: Option<String> = None;
    let mut verify_key: Option<String> = None;

    while let Some(msg_res) = ws_stream.next().await {
        let msg = match msg_res {
            Ok(m) => m,
            Err(_) => break,
        };

        match msg {
            Message::Binary(bytes) => {
                // Drop frame if scanner is busy processing the previous one (zero-latency backpressure)
                let _ = frame_tx.try_send((bytes, password.clone(), verify_key.clone()));
            }
            Message::Text(cfg_str) => {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&cfg_str) {
                    if let Some(p) = json.get("password").and_then(|v| v.as_str()) {
                        password = if p.is_empty() { None } else { Some(p.to_string()) };
                    }
                    if let Some(v) = json.get("verify").and_then(|v| v.as_str()) {
                        verify_key = if v.is_empty() { None } else { Some(v.to_string()) };
                    }
                }
            }
            Message::Ping(_) => {}
            Message::Close(_) => break,
            _ => {}
        }
    }

    send_task.abort();
    worker_task.abort();
}

// ============================================================================
// Scanning Logic & rsmpeg Integration
// ============================================================================

/// Scans raw image memory buffer using UniversalScanner
fn scan_image_bytes(
    bytes: &[u8],
    password: Option<&str>,
    verify_key: Option<&str>,
) -> Result<UniversalScanOutput, String> {
    let suffix = if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        ".png"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        ".jpg"
    } else if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        ".webp"
    } else {
        ".jpg"
    };

    let mut temp = tempfile::Builder::new()
        .prefix("tsc_scan_")
        .suffix(suffix)
        .tempfile()
        .map_err(|e| e.to_string())?;

    temp.write_all(bytes).map_err(|e| e.to_string())?;
    temp.flush().map_err(|e| e.to_string())?;

    let path_str = temp.path().to_str().ok_or("Invalid temporary path")?;
    let result = UniversalScanner::scan(path_str, password, verify_key, None)
        .map_err(|e| e.to_string())?;

    let corners_json = extract_corners_json(&result.corners);
    let text = result.text().map(|s| s.to_string());

    Ok(UniversalScanOutput {
        detected: true,
        text,
        size: result.size,
        is_nano: result.is_nano,
        color_mode: format!("{:?}", result.color_mode),
        confidence: result.confidence as f64,
        corners: corners_json,
        elapsed_ms: 0.0,
        error: None,
    })
}

/// Decodes uploaded video file using `rsmpeg`, scans frames, and selects highest confidence result
fn scan_video_media(
    video_bytes: &[u8],
    password: Option<&str>,
    verify_key: Option<&str>,
) -> Result<UniversalScanOutput, String> {
    let mut temp_video = tempfile::Builder::new()
        .prefix("upload_")
        .suffix(".mp4")
        .tempfile()
        .map_err(|e| e.to_string())?;

    temp_video.write_all(video_bytes).map_err(|e| e.to_string())?;
    temp_video.flush().map_err(|e| e.to_string())?;

    let frame_files = extract_video_frames_rsmpeg(temp_video.path(), 20)
        .map_err(|e| format!("rsmpeg decoding error: {}", e))?;

    if frame_files.is_empty() {
        return Err("Could not extract any video frames".to_string());
    }

    let mut best_result: Option<UniversalScanOutput> = None;

    for frame_file in frame_files {
        if let Ok(res) = scan_single_path(frame_file.path(), password, verify_key) {
            if best_result.as_ref().map_or(true, |b| res.confidence > b.confidence) {
                best_result = Some(res);
            }
        }
    }

    best_result.ok_or_else(|| "No barcode detected in any video frame".to_string())
}

fn scan_single_path(
    path: &Path,
    password: Option<&str>,
    verify_key: Option<&str>,
) -> Result<UniversalScanOutput, String> {
    let path_str = path.to_str().ok_or("Invalid file path")?.to_string();
    let pwd = password.map(str::to_string);
    let vkey = verify_key.map(str::to_string);

    // Safeguard against panics in the computer vision / decoding pipeline
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        UniversalScanner::scan(&path_str, pwd.as_deref(), vkey.as_deref(), None)
    }));

    match result {
        Ok(Ok(res)) => {
            let corners_json = extract_corners_json(&res.corners);
            let text = res.text().map(str::to_string);

            Ok(UniversalScanOutput {
                detected: true,
                text,
                size: res.size,
                is_nano: res.is_nano,
                color_mode: format!("{:?}", res.color_mode),
                confidence: res.confidence as f64,
                corners: corners_json,
                elapsed_ms: 0.0,
                error: None,
            })
        }
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("Invalid or unreadable quadrilateral".to_string()),
    }
}

/// Core FFmpeg frame extraction via rsmpeg
/// Core FFmpeg frame extraction via rsmpeg
fn extract_video_frames_rsmpeg<P: AsRef<Path>>(
    video_path: P,
    max_frames: usize,
) -> Result<Vec<NamedTempFile>, Box<dyn std::error::Error + Send + Sync>> {
    let path_str = video_path.as_ref().to_str().ok_or("Invalid path")?;
    let cstr = CString::new(path_str)?;
    
    let mut format_context = AVFormatContextInput::open(&cstr, None, &mut None)?;

    // Use ffi::AVMEDIA_TYPE_VIDEO
    let (stream_index, decoder) = format_context
        .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)?
        .ok_or("No video stream found in file")?;

    let stream = &format_context.streams()[stream_index];
    let mut codec_context = AVCodecContext::new(&decoder);
    codec_context.apply_codecpar(&stream.codecpar())?;
    codec_context.open(None)?;

    let width = codec_context.width;
    let height = codec_context.height;
    let pix_fmt = codec_context.pix_fmt;

    // Use ffi::AV_PIX_FMT_RGB24
    let mut sws_context = SwsContext::get_context(
        width,
        height,
        pix_fmt,
        width,
        height,
        ffi::AV_PIX_FMT_RGB24,
        ffi::SWS_BILINEAR,
        None,
        None,
        None,
    )
    .ok_or("Failed to initialize SwsContext")?;

    let mut frame_files = Vec::new();

    while let Some(packet) = format_context.read_packet()? {
        if packet.stream_index == stream_index as i32 {
            codec_context.send_packet(Some(&packet))?;
            while let Ok(frame) = codec_context.receive_frame() {
                let mut rgb_frame = AVFrame::new();
                // Use ffi::AV_PIX_FMT_RGB24
                rgb_frame.set_format(ffi::AV_PIX_FMT_RGB24);
                rgb_frame.set_width(width);
                rgb_frame.set_height(height);
                rgb_frame.alloc_buffer()?;

                sws_context.scale_frame(&frame, 0, height, &mut rgb_frame)?;

                let data = rgb_frame.data[0];
                let linesize = rgb_frame.linesize[0] as usize;
                let mut img_buf = image::RgbImage::new(width as u32, height as u32);
                for y in 0..height as usize {
                    let row = unsafe {
                        std::slice::from_raw_parts(data.add(y * linesize), width as usize * 3)
                    };
                    for x in 0..width as usize {
                        img_buf.put_pixel(
                            x as u32,
                            y as u32,
                            image::Rgb([row[x * 3], row[x * 3 + 1], row[x * 3 + 2]]),
                        );
                    }
                }

                let temp_png = tempfile::Builder::new().suffix(".png").tempfile()?;
                img_buf.save_with_format(temp_png.path(), image::ImageFormat::Png)?;
                frame_files.push(temp_png);

                if frame_files.len() >= max_frames {
                    return Ok(frame_files);
                }
            }
        }
    }

    Ok(frame_files)
}

/// Robust conversion of any corners representation into JSON `[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`
fn extract_corners_json<T: std::fmt::Debug>(corners: &T) -> serde_json::Value {
    let debug_str = format!("{:?}", corners);
    let re = Regex::new(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?").unwrap();
    let nums: Vec<f64> = re
        .find_iter(&debug_str)
        .filter_map(|m| m.as_str().parse::<f64>().ok())
        .collect();

    if nums.len() >= 8 {
        let mut pts = Vec::new();
        for chunk in nums.chunks(2) {
            if chunk.len() == 2 {
                pts.push(serde_json::json!([chunk[0], chunk[1]]));
            }
        }
        serde_json::json!(pts)
    } else {
        serde_json::json!(debug_str)
    }
}

// ============================================================================
// Existing Generators & Classic Decoders
// ============================================================================

async fn handle_generate(Query(params): Query<GenerateParams>) -> Response {
    let mode = params
        .colors
        .as_deref()
        .and_then(ColorMode::from_str_loose)
        .unwrap_or(ColorMode::FourColor);

    let ecc = params
        .ecc
        .as_deref()
        .map(EccLevel::from_str_loose)
        .unwrap_or(EccLevel::Medium);

    let format_str = params.format.as_deref().unwrap_or("png").to_lowercase();
    let mod_size = params.mod_size.unwrap_or(15).max(1);
    let margin = params.margin.unwrap_or(2);

    if format_str == "wav" {
        let text_data = params
            .text
            .unwrap_or_else(|| "T-Spine Audio Packet".to_string());
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        if let Err(e) = AudioModem::export_wav(
            text_data.as_bytes(),
            path,
            mode,
            ecc,
            params.password.as_deref(),
            0.5,
        ) {
            return (StatusCode::BAD_REQUEST, format!("Audio generation error: {}", e)).into_response();
        }

        let audio_bytes = std::fs::read(path).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
        headers.insert(
            header::CONTENT_DISPOSITION,
            HeaderValue::from_static("inline; filename=\"tspine.wav\""),
        );
        return (headers, audio_bytes).into_response();
    }

    let mut encoder = TSpineEncoder::new()
        .color_mode(mode)
        .ecc_level(ecc)
        .nano(params.nano.unwrap_or(false))
        .min_header(params.min_header.unwrap_or(false));

    if let Some(pwd) = params.password.as_deref() {
        encoder = encoder.password(pwd);
    }
    if let Some(key) = params.sign.as_deref() {
        encoder = encoder.sign_key(key);
    }
    if let Some(size) = params.size {
        encoder = encoder.forced_size(size);
    }

    let encode_result = if let (Some(pub_data), Some(priv_data)) =
        (params.public.as_deref(), params.private.as_deref())
    {
        encoder.encode_dual(pub_data.as_bytes(), priv_data.as_bytes())
    } else {
        let default_text = "Hello from T-Spine Code (TSC)!".to_string();
        let text_data = params.text.as_ref().unwrap_or(&default_text);
        encoder.encode(text_data.as_bytes())
    };

    let (grid, meta) = match encode_result {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    match format_str.as_str() {
        "svg" => {
            let svg = SvgExporter::to_svg_string(&grid, mod_size, margin);
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/svg+xml"));
            (headers, svg).into_response()
        }
        "html" => {
            let html = HtmlExporter::to_html_string(&grid, Some(&meta), mod_size.max(14), margin);
            Html(html).into_response()
        }
        "term" | "terminal" | "ansi" => {
            let term = TerminalExporter::render(&grid);
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            );
            (headers, term).into_response()
        }
        "json" => {
            let resp = MetadataResponse {
                size: meta.size,
                raw_bytes: meta.raw_bytes,
                packed_bytes: meta.packed_bytes,
                total_cap_bytes: meta.total_cap_bytes,
                ecc_bytes: meta.ecc_bytes,
                colors: meta.colors,
                bits_per_cell: meta.bits_per_cell,
                is_dual: meta.is_dual,
                is_encrypted: meta.is_encrypted,
                is_signed: meta.is_signed,
                is_nano: meta.is_nano,
            };
            Json(resp).into_response()
        }
        _ => {
            let img = ImageExporter::to_image_buffer(&grid, mod_size, margin);
            let mut png_bytes = Cursor::new(Vec::new());
            img.write_to(&mut png_bytes, image::ImageFormat::Png).unwrap();

            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
            headers.insert(
                header::CONTENT_DISPOSITION,
                HeaderValue::from_static("inline; filename=\"tspine.png\""),
            );
            (headers, png_bytes.into_inner()).into_response()
        }
    }
}

async fn handle_decode(mut multipart: Multipart) -> Response {
    let mut file_bytes = Vec::new();
    let mut password = None;
    let mut verify_key = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" || name == "image" {
            if let Ok(bytes) = field.bytes().await {
                file_bytes = bytes.to_vec();
            }
        } else if name == "password" {
            if let Ok(txt) = field.text().await {
                if !txt.is_empty() {
                    password = Some(txt);
                }
            }
        } else if name == "verify" {
            if let Ok(txt) = field.text().await {
                if !txt.is_empty() {
                    verify_key = Some(txt);
                }
            }
        }
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "No file uploaded" })),
        )
            .into_response();
    }

    if let Ok(dyn_img) = image::load_from_memory(&file_bytes) {
        let rgb_img = dyn_img.to_rgb8();
        return match ImageScanner::scan_image(
            &rgb_img,
            password.as_deref(),
            verify_key.as_deref(),
            None,
        ) {
            Ok(DecodedPayload::Text(t)) => {
                Json(serde_json::json!({ "type": "text", "data": t })).into_response()
            }
            Ok(DecodedPayload::Binary(b)) => {
                Json(serde_json::json!({ "type": "binary", "size": b.len(), "hex": hex_encode(&b) }))
                    .into_response()
            }
            Ok(DecodedPayload::Dual {
                public_data,
                private_data,
            }) => Json(serde_json::json!({
                "type": "dual",
                "public": public_data,
                "private": private_data
            }))
            .into_response(),
            Err(e) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response(),
        };
    }

    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": "Unsupported image format" })),
    )
        .into_response()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ============================================================================
// Modern Responsive Scanner Webpage (Embedded HTML/JS/CSS)
// ============================================================================

const SCANNER_HTML_PAGE: &str = r###"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>T-Spine Universal Scanner</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #131b2e;
            --card-border: #1e293b;
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.25);
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --cyan: #06b6d4;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); min-height: 100vh; padding: 24px 16px; }
        .container { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid var(--card-border); }
        .logo-box { display: flex; align-items: center; gap: 12px; }
        .logo-box h1 { font-size: 1.5rem; letter-spacing: -0.5px; background: linear-gradient(135deg, #10b981, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status-pill { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; padding: 6px 14px; border-radius: 9999px; background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent); color: var(--accent); }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        /* Highlight Box: Best Recorded Accuracy */
        .best-card {
            background: linear-gradient(145deg, #131e33, #0f172a);
            border: 2px solid var(--accent);
            box-shadow: 0 0 25px var(--accent-glow);
            border-radius: 16px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        .best-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .best-badge { font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; color: var(--accent); background: rgba(16, 185, 129, 0.15); padding: 4px 10px; border-radius: 6px; }
        .btn-reset { background: transparent; border: 1px solid var(--card-border); color: var(--text-sub); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; transition: 0.2s; }
        .btn-reset:hover { border-color: #ef4444; color: #ef4444; }
        .best-grid { display: grid; grid-template-columns: 200px 1fr; gap: 20px; align-items: center; }
        .acc-number { font-size: 3rem; font-weight: 800; color: #fff; line-height: 1; }
        .acc-label { font-size: 0.85rem; color: var(--text-sub); margin-top: 4px; }
        .best-payload-box { background: rgba(0, 0, 0, 0.4); border: 1px solid var(--card-border); border-radius: 10px; padding: 12px 16px; }
        .payload-text { font-family: monospace; font-size: 1.05rem; color: #38bdf8; word-break: break-all; max-height: 80px; overflow-y: auto; }
        .best-meta { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 0.8rem; color: var(--text-sub); }
        .meta-pill { background: rgba(255, 255, 255, 0.05); padding: 3px 8px; border-radius: 4px; }

        /* Main Workspace: Left Camera/Upload, Right Info */
        .workspace { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; }
        @media (max-width: 850px) { .workspace, .best-grid { grid-template-columns: 1fr; } }
        .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; padding: 18px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 14px; }
        .tab-btn { flex: 1; padding: 10px; background: rgba(255, 255, 255, 0.04); border: 1px solid var(--card-border); border-radius: 8px; color: var(--text-sub); font-weight: 600; cursor: pointer; transition: 0.2s; }
        .tab-btn.active { background: rgba(16, 185, 129, 0.15); border-color: var(--accent); color: #fff; }

        .viewport-wrapper { position: relative; width: 100%; height: 360px; background: #000; border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        video { width: 100%; height: 100%; object-fit: cover; }
        canvas.overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        .hud-stats { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; font-family: monospace; color: #a7f3d0; }

        /* File Upload Area */
        .upload-dropzone { border: 2px dashed var(--card-border); border-radius: 12px; padding: 36px 20px; text-align: center; cursor: pointer; transition: 0.2s; }
        .upload-dropzone:hover { border-color: var(--cyan); background: rgba(6, 182, 212, 0.04); }
        .upload-dropzone input { display: none; }

        /* Security Input Box */
        .config-box { margin-top: 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .input-group { display: flex; flex-direction: column; gap: 4px; }
        .input-group label { font-size: 0.75rem; color: var(--text-sub); font-weight: 600; }
        .input-group input { background: rgba(0,0,0,0.3); border: 1px solid var(--card-border); padding: 8px 12px; border-radius: 6px; color: #fff; font-size: 0.85rem; }
        .input-group input:focus { outline: none; border-color: var(--accent); }

        /* Details list */
        .meta-list { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
        .meta-row { display: flex; justify-content: space-between; font-size: 0.85rem; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .meta-row span:first-child { color: var(--text-sub); }
        .meta-row span:last-child { font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-box">
                <h1>T-Spine Universal Scanner</h1>
            </div>
            <div class="status-pill">
                <div class="status-dot"></div>
                <span id="wsStatus">Connecting...</span>
            </div>
        </header>

        <!-- Main Prominent Box: Best Recorded Accuracy Yet -->
        <div class="best-card">
            <div class="best-header">
                <div class="best-badge">Best Recorded Accuracy Yet</div>
                <button class="btn-reset" onclick="resetBestScan()">Reset Best</button>
            </div>
            <div class="best-grid">
                <div>
                    <div class="acc-number" id="bestConfidence">0.0%</div>
                    <div class="acc-label">Confidence Rating</div>
                </div>
                <div>
                    <div class="best-payload-box">
                        <div class="payload-text" id="bestPayload">No barcode recorded yet...</div>
                    </div>
                    <div class="best-meta">
                        <span class="meta-pill" id="bestSize">Grid: --</span>
                        <span class="meta-pill" id="bestLayout">Layout: --</span>
                        <span class="meta-pill" id="bestColor">Mode: --</span>
                        <span class="meta-pill" id="bestTime">Latency: --</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scanning Workspace -->
        <div class="workspace">
            <div class="card">
                <div class="tabs">
                    <button class="tab-btn active" id="tabCamera" onclick="switchMode('camera')">Live Camera Stream</button>
                    <button class="tab-btn" id="tabUpload" onclick="switchMode('upload')">File Upload (Image / Video)</button>
                </div>

                <!-- Live Camera Section -->
                <div id="cameraSection">
                    <div class="viewport-wrapper">
                        <video id="video" autoplay playsinline muted></video>
                        <canvas id="overlayCanvas" class="overlay"></canvas>
                        <div class="hud-stats" id="hudStats">FPS: 0 | Latency: 0ms</div>
                    </div>
                </div>

                <!-- File Upload Section -->
                <div id="uploadSection" style="display: none;">
                    <div class="upload-dropzone" onclick="document.getElementById('fileInput').click()">
                        <p style="font-size: 1.1rem; font-weight: 600;">Click or Drop File Here</p>
                        <p style="color: var(--text-sub); font-size: 0.85rem; margin-top: 6px;">Supports PNG, JPG, WebP, MP4, WebM (rsmpeg)</p>
                        <input type="file" id="fileInput" accept="image/*,video/*" onchange="uploadSelectedFile(event)">
                    </div>
                </div>

                <!-- Advanced Security Panel -->
                <div class="config-box">
                    <div class="input-group">
                        <label>Decryption Password (Optional)</label>
                        <input type="password" id="cfgPassword" placeholder="Passphrase" oninput="updateConfig()">
                    </div>
                    <div class="input-group">
                        <label>Verification Key (Optional)</label>
                        <input type="text" id="cfgVerify" placeholder="ECDSA Public Key" oninput="updateConfig()">
                    </div>
                </div>
            </div>

            <!-- Live Details Card -->
            <div class="card">
                <h3 style="font-size: 1.1rem; margin-bottom: 12px;">Live Scanner Diagnostics</h3>
                <div class="meta-list">
                    <div class="meta-row"><span>Status:</span><span id="liveStatus" style="color: var(--cyan);">Scanning...</span></div>
                    <div class="meta-row"><span>Current Confidence:</span><span id="liveConf">0.0%</span></div>
                    <div class="meta-row"><span>Grid Size:</span><span id="liveSize">--</span></div>
                    <div class="meta-row"><span>Barcode Layout:</span><span id="liveLayout">--</span></div>
                    <div class="meta-row"><span>Detected Color Mode:</span><span id="liveColor">--</span></div>
                    <div class="meta-row"><span>Corner Polygon:</span><span id="liveCorners" style="font-family: monospace; font-size: 0.75rem;">--</span></div>
                    <div class="meta-row"><span>Server Round-Trip:</span><span id="liveElapsed">0.0 ms</span></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Hidden canvas for capturing video frames -->
    <canvas id="captureCanvas" style="display: none;"></canvas>

    <script>
        let ws;
        let isConnected = false;
        let isProcessing = false;
        let video = document.getElementById('video');
        let captureCanvas = document.getElementById('captureCanvas');
        let overlayCanvas = document.getElementById('overlayCanvas');
        let overlayCtx = overlayCanvas.getContext('2d');
        let captureCtx = captureCanvas.getContext('2d');
        let stream = null;

        let bestAccuracy = 0.0;
        let frameCount = 0;
        let lastFpsUpdate = performance.now();
        let fps = 0;

        // WebSocket initialization
        function initWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${location.host}/scan/ws`;
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                isConnected = true;
                document.getElementById('wsStatus').innerText = 'Connected (Live WS)';
                document.getElementById('wsStatus').style.color = '#10b981';
                updateConfig();
            };

            ws.onclose = () => {
                isConnected = false;
                document.getElementById('wsStatus').innerText = 'Disconnected, retrying...';
                document.getElementById('wsStatus').style.color = '#ef4444';
                setTimeout(initWebSocket, 2000);
            };

            ws.onmessage = (event) => {
                isProcessing = false;
                const data = JSON.parse(event.data);
                handleScanResult(data);
            };
        }

        function updateConfig() {
            if (!isConnected) return;
            const payload = {
                password: document.getElementById('cfgPassword').value,
                verify: document.getElementById('cfgVerify').value
            };
            ws.send(JSON.stringify(payload));
        }

        // Camera loop
        async function startCamera() {
            try {
                // Request back camera with continuous focus mode
                const constraints = {
                    video: {
                        facingMode: { ideal: "environment" },
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        advanced: [
                            { focusMode: "continuous" },
                            { exposureMode: "continuous" }
                        ]
                    }
                };

                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;

                // Apply continuous hardware autofocus if supported
                const track = stream.getVideoTracks()[0];
                if (track && track.getCapabilities) {
                    const caps = track.getCapabilities();
                    if (caps.focusMode && caps.focusMode.includes("continuous")) {
                        await track.applyConstraints({
                            advanced: [{ focusMode: "continuous" }]
                        }).catch(() => {});
                    }
                }

                video.onloadedmetadata = () => {
                    overlayCanvas.width = video.videoWidth;
                    overlayCanvas.height = video.videoHeight;
                    requestAnimationFrame(renderLoop);
                };
            } catch (err) {
                console.error("Camera access failed:", err);
                document.getElementById('liveStatus').innerText = "Camera Access Error";
            }
        }

        function renderLoop() {
            calculateFps();

            if (isConnected && !isProcessing && video.videoWidth > 0 && document.getElementById('cameraSection').style.display !== 'none') {
                isProcessing = true;

                // Scale frame to a normalized max dimension of 800px for optimal speed and accuracy
                const maxDim = 800;
                let targetW = video.videoWidth;
                let targetH = video.videoHeight;
                if (targetW > maxDim || targetH > maxDim) {
                    const ratio = Math.min(maxDim / targetW, maxDim / targetH);
                    targetW = Math.round(targetW * ratio);
                    targetH = Math.round(targetH * ratio);
                }

                captureCanvas.width = targetW;
                captureCanvas.height = targetH;
                captureCtx.drawImage(video, 0, 0, targetW, targetH);

                captureCanvas.toBlob((blob) => {
                    if (blob && isConnected) {
                        blob.arrayBuffer().then(buf => ws.send(buf));
                    } else {
                        isProcessing = false;
                    }
                }, 'image/jpeg', 0.80);
            }

            requestAnimationFrame(renderLoop);
        }

        function calculateFps() {
            frameCount++;
            const now = performance.now();
            if (now - lastFpsUpdate >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
            }
        }

        function handleScanResult(data) {
            document.getElementById('hudStats').innerText = `FPS: ${fps} | Scan: ${data.elapsed_ms.toFixed(1)}ms`;
            document.getElementById('liveElapsed').innerText = `${data.elapsed_ms.toFixed(1)} ms`;

            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            if (data.detected) {
                document.getElementById('liveStatus').innerText = "Detected!";
                document.getElementById('liveStatus').style.color = "#10b981";
                document.getElementById('liveConf').innerText = `${data.confidence.toFixed(1)}%`;
                document.getElementById('liveSize').innerText = `${data.size}x${data.size}`;
                document.getElementById('liveLayout').innerText = data.is_nano ? "Nano" : "Standard";
                document.getElementById('liveColor').innerText = data.color_mode;
                document.getElementById('liveCorners').innerText = JSON.stringify(data.corners);

                // Draw corner polygon on canvas
                if (Array.isArray(data.corners) && data.corners.length >= 4) {
                    overlayCtx.strokeStyle = "#10b981";
                    overlayCtx.lineWidth = 4;
                    overlayCtx.beginPath();
                    overlayCtx.moveTo(data.corners[0][0], data.corners[0][1]);
                    for (let i = 1; i < data.corners.length; i++) {
                        overlayCtx.lineTo(data.corners[i][0], data.corners[i][1]);
                    }
                    overlayCtx.closePath();
                    overlayCtx.stroke();

                    // Corner dots
                    data.corners.forEach((pt, idx) => {
                        overlayCtx.fillStyle = idx === 0 ? "#ef4444" : "#06b6d4";
                        overlayCtx.beginPath();
                        overlayCtx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
                        overlayCtx.fill();
                    });
                }

                // Check and update "Best Recorded Accuracy"
                if (data.confidence > bestAccuracy) {
                    bestAccuracy = data.confidence;
                    document.getElementById('bestConfidence').innerText = `${bestAccuracy.toFixed(1)}%`;
                    document.getElementById('bestPayload').innerText = data.text || "[Binary or Dual Payload]";
                    document.getElementById('bestSize').innerText = `Grid: ${data.size}x${data.size}`;
                    document.getElementById('bestLayout').innerText = `Layout: ${data.is_nano ? "Nano" : "Standard"}`;
                    document.getElementById('bestColor').innerText = `Mode: ${data.color_mode}`;
                    document.getElementById('bestTime').innerText = `Latency: ${data.elapsed_ms.toFixed(1)}ms`;
                }
            } else {
                document.getElementById('liveStatus').innerText = "Searching...";
                document.getElementById('liveStatus').style.color = "#94a3b8";
                document.getElementById('liveConf').innerText = "0.0%";
            }
        }

        function resetBestScan() {
            bestAccuracy = 0.0;
            document.getElementById('bestConfidence').innerText = "0.0%";
            document.getElementById('bestPayload').innerText = "No barcode recorded yet...";
            document.getElementById('bestSize').innerText = "Grid: --";
            document.getElementById('bestLayout').innerText = "Layout: --";
            document.getElementById('bestColor').innerText = "Mode: --";
            document.getElementById('bestTime').innerText = "Latency: --";
        }

        // Mode Switching (Camera vs Upload)
        function switchMode(mode) {
            document.getElementById('tabCamera').classList.toggle('active', mode === 'camera');
            document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
            document.getElementById('cameraSection').style.display = mode === 'camera' ? 'block' : 'none';
            document.getElementById('uploadSection').style.display = mode === 'upload' ? 'block' : 'none';
        }

        // Manual File Upload
        async function uploadSelectedFile(event) {
            const file = event.target.files[0];
            if (!file) return;

            document.getElementById('liveStatus').innerText = "Uploading & Decoding...";
            const formData = new FormData();
            formData.append('file', file);
            formData.append('password', document.getElementById('cfgPassword').value);
            formData.append('verify', document.getElementById('cfgVerify').value);

            try {
                const res = await fetch('/scan', { method: 'POST', body: formData });
                const json = await res.json();
                handleScanResult(json);
            } catch (e) {
                console.error("Upload error:", e);
                document.getElementById('liveStatus').innerText = "Upload Decode Failed";
            }
        }

        window.addEventListener('load', () => {
            initWebSocket();
            startCamera();
        });
    </script>
</body>
</html>
"###;