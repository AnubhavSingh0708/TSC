use image::RgbImage;
use std::path::Path;

use crate::decoder::TSpineDecoder;
use crate::error::{Result, TSpineError};
use crate::grid::Grid;
use crate::types::{ColorMode, DecodedPayload, Rgb};

// ============================================================================
// Data Structures & Results
// ============================================================================

/// Result of a universal T-Spine scan.
#[derive(Debug, Clone)]
pub struct UniversalScanResult {
    /// Fully unpacked payload from the decoder.
    pub payload: DecodedPayload,
    /// Extracted UTF-8 text if available.
    pub text: Option<String>,
    /// Confidence score between 0.0 (unusable/corrupt) and 100.0 (pristine digital original).
    pub confidence: f32,
    /// Detected grid size (e.g. 5, 7, 9, ..., 251).
    pub size: usize,
    /// Whether the code uses the Nano layout.
    pub is_nano: bool,
    /// Detected color mode.
    pub color_mode: ColorMode,
    /// Quad corners in original image coordinates: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    pub corners: [(f32, f32); 4],
}

impl UniversalScanResult {
    /// Convenience helper to access the primary text content.
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }
}

// ============================================================================
// 2D Geometry & Projective Homography
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dist_sq(&self, other: &Point2D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    #[inline]
    pub fn dist(&self, other: &Point2D) -> f32 {
        self.dist_sq(other).sqrt()
    }
}

/// Heckbert Projective Mapping from unit square [0, 1] x [0, 1] to an arbitrary quadrilateral.
#[derive(Debug, Clone, Copy)]
pub struct ProjectiveMapping {
    a: f32, b: f32, c: f32,
    d: f32, e: f32, f: f32,
    g: f32, h: f32,
}

impl ProjectiveMapping {
    pub fn from_quad(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D) -> Option<Self> {
        let dx1 = p1.x - p2.x;
        let dx2 = p3.x - p2.x;
        let sum_x = p0.x - p1.x + p2.x - p3.x;
        let dy1 = p1.y - p2.y;
        let dy2 = p3.y - p2.y;
        let sum_y = p0.y - p1.y + p2.y - p3.y;

        if sum_x.abs() < 1e-5 && sum_y.abs() < 1e-5 {
            Some(Self {
                a: p1.x - p0.x,
                b: p3.x - p0.x,
                c: p0.x,
                d: p1.y - p0.y,
                e: p3.y - p0.y,
                f: p0.y,
                g: 0.0,
                h: 0.0,
            })
        } else {
            let det = dx1 * dy2 - dx2 * dy1;
            if det.abs() < 1e-7 {
                return None;
            }
            let g = (sum_x * dy2 - sum_y * dx2) / det;
            let h = (dx1 * sum_y - dy1 * sum_x) / det;
            let a = p1.x - p0.x + g * p1.x;
            let b = p3.x - p0.x + h * p3.x;
            let c = p0.x;
            let d = p1.y - p0.y + g * p1.y;
            let e = p3.y - p0.y + h * p3.y;
            let f = p0.y;

            Some(Self { a, b, c, d, e, f, g, h })
        }
    }

    #[inline]
    pub fn map(&self, u: f32, v: f32) -> (f32, f32) {
        let denom = self.g * u + self.h * v + 1.0;
        if denom.abs() < 1e-7 {
            return (self.c, self.f);
        }
        let x = (self.a * u + self.b * v + self.c) / denom;
        let y = (self.d * u + self.e * v + self.f) / denom;
        (x, y)
    }
}

// ============================================================================
// Subpixel Interpolation & Image Utilities
// ============================================================================

#[inline]
fn rgb_luminance(c: &Rgb) -> u8 {
    ((77 * c.0 as u32 + 150 * c.1 as u32 + 29 * c.2 as u32) >> 8) as u8
}

fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> Rgb {
    let w = img.width();
    let h = img.height();
    if w == 0 || h == 0 {
        return Rgb(0, 0, 0);
    }
    let cx = x.clamp(0.0, (w - 1) as f32);
    let cy = y.clamp(0.0, (h - 1) as f32);

    let x0 = cx.floor() as u32;
    let y0 = cy.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = cx - x0 as f32;
    let fy = cy - y0 as f32;

    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);

    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out[0], out[1], out[2])
}

// ============================================================================
// Contour Extraction & Polygon Simplification
// ============================================================================

fn perpendicular_distance(p: Point2D, a: Point2D, b: Point2D) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-6 {
        return p.dist(&a);
    }
    let num = (dy * p.x - dx * p.y + b.x * a.y - b.y * a.x).abs();
    num / len_sq.sqrt()
}

fn ramer_douglas_peucker(points: &[Point2D], epsilon: f32) -> Vec<Point2D> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut dmax = 0.0f32;
    let mut index = 0;
    let p_first = points[0];
    let p_last = points[points.len() - 1];

    for i in 1..(points.len() - 1) {
        let d = perpendicular_distance(points[i], p_first, p_last);
        if d > dmax {
            index = i;
            dmax = d;
        }
    }

    if dmax > epsilon {
        let mut left = ramer_douglas_peucker(&points[0..=index], epsilon);
        let right = ramer_douglas_peucker(&points[index..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![p_first, p_last]
    }
}

fn is_convex_quad(pts: &[Point2D]) -> bool {
    if pts.len() != 4 {
        return false;
    }
    let mut sign = 0.0f32;
    for i in 0..4 {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % 4];
        let p2 = pts[(i + 2) % 4];
        let cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);
        if cross.abs() < 1e-3 {
            return false;
        }
        if i == 0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

fn polygon_area(pts: &[Point2D]) -> f32 {
    let mut area = 0.0f32;
    let n = pts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    (area * 0.5).abs()
}

// ============================================================================
// Universal Scanner Engine
// ============================================================================

pub struct UniversalScanner {
    password: Option<String>,
    verify_key: Option<String>,
    forced_mode: Option<ColorMode>,
    forced_size: Option<usize>,
}

impl Default for UniversalScanner {
    fn default() -> Self {
        Self {
            password: None,
            verify_key: None,
            forced_mode: None,
            forced_size: None,
        }
    }
}
/// Extracts the 4 most prominent corner points from a slightly curved 5-7 point polygon
fn extract_extreme_quad(pts: &[Point2D]) -> Option<[Point2D; 4]> {
    if pts.len() < 4 {
        return None;
    }

    // Centroid of the polygon
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for p in pts {
        cx += p.x;
        cy += p.y;
    }
    cx /= pts.len() as f32;
    cy /= pts.len() as f32;

    // Find the furthest point in each of the 4 quadrants relative to centroid
    let mut best: [Option<Point2D>; 4] = [None; 4];
    let mut max_dist = [0.0f32; 4];

    for p in pts {
        let dx = p.x - cx;
        let dy = p.y - cy;
        let dist = dx * dx + dy * dy;

        let quad_idx = match (dx >= 0.0, dy >= 0.0) {
            (false, false) => 0, // Top-Left
            (true, false) => 1,  // Top-Right
            (true, true) => 2,   // Bottom-Right
            (false, true) => 3,  // Bottom-Left
        };

        if dist > max_dist[quad_idx] {
            max_dist[quad_idx] = dist;
            best[quad_idx] = Some(*p);
        }
    }

    if let (Some(p0), Some(p1), Some(p2), Some(p3)) = (best[0], best[1], best[2], best[3]) {
        let candidate: [Point2D; 4] = [p0, p1, p2, p3];
        if is_convex_quad(&candidate) {
            return Some(candidate);
        }
    }

    None
}

impl UniversalScanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn password(mut self, pwd: impl Into<String>) -> Self {
        self.password = Some(pwd.into());
        self
    }

    pub fn verify_key(mut self, key: impl Into<String>) -> Self {
        self.verify_key = Some(key.into());
        self
    }

    pub fn forced_mode(mut self, mode: ColorMode) -> Self {
        self.forced_mode = Some(mode);
        self
    }

    pub fn forced_size(mut self, size: usize) -> Self {
        self.forced_size = Some(size);
        self
    }

    pub fn scan_file(&self, path: impl AsRef<Path>) -> Result<UniversalScanResult> {
        let dyn_img = image::open(path)
            .map_err(|e| TSpineError::ScanFailed(format!("Failed to open image: {}", e)))?;
        let img = dyn_img.to_rgb8();
        self.scan_image(&img)
    }

    pub fn scan(
        path: impl AsRef<Path>,
        password: Option<&str>,
        verify_key: Option<&str>,
        forced_mode: Option<ColorMode>,
    ) -> Result<UniversalScanResult> {
        let mut scanner = Self::new();
        if let Some(p) = password {
            scanner = scanner.password(p);
        }
        if let Some(vk) = verify_key {
            scanner = scanner.verify_key(vk);
        }
        if let Some(m) = forced_mode {
            scanner = scanner.forced_mode(m);
        }
        scanner.scan_file(path)
    }

    pub fn scan_image(&self, img: &RgbImage) -> Result<UniversalScanResult> {
        let (w, h) = img.dimensions();
        if w < 5 || h < 5 {
            return Err(TSpineError::ScanFailed("Image dimensions too small".to_string()));
        }

        let decoder = TSpineDecoder::new();
        let decoder = if let Some(p) = &self.password {
            decoder.password(p)
        } else {
            decoder
        };
        let decoder = if let Some(vk) = &self.verify_key {
            decoder.verify_key(vk)
        } else {
            decoder
        };

        let candidate_quads = self.detect_candidate_quads(img);

        let modes_to_test = if let Some(m) = self.forced_mode {
            vec![m]
        } else {
            vec![ColorMode::FourColor, ColorMode::EightColor, ColorMode::Monochrome]
        };

        for quad in candidate_quads {
            for rot in 0..4 {
                let p0 = quad[rot];
                let p1 = quad[(rot + 1) % 4];
                let p2 = quad[(rot + 2) % 4];
                let p3 = quad[(rot + 3) % 4];

                let mapping = match ProjectiveMapping::from_quad(p0, p1, p2, p3) {
                    Some(m) => m,
                    None => continue,
                };

                // Relative Orientation Filter:
                // Bottom-Right must be visibly brighter than Bottom-Left
                let (br_x, br_y) = mapping.map(0.95, 0.95);
                let (bl_x, bl_y) = mapping.map(0.05, 0.95);
                let br_lum = rgb_luminance(&sample_bilinear(img, br_x, br_y)) as i32;
                let bl_lum = rgb_luminance(&sample_bilinear(img, bl_x, bl_y)) as i32;

                // Relative contrast check: avoids failing under bright or dim lighting
                if br_lum - bl_lum < 25 {
                    continue;
                }

                let candidate_sizes = self.determine_candidate_sizes(&mapping, img);

                for size in candidate_sizes {
                    let is_nano_grid = Grid::is_nano(size, false);
                    let mut layouts = Vec::new();
                    if size <= 7 {
                        layouts.push(true);
                        layouts.push(false);
                    } else if is_nano_grid {
                        layouts.push(true);
                    } else {
                        layouts.push(false);
                    }

                    for is_nano in layouts {
                        if !self.verify_skeleton(&mapping, img, size, is_nano) {
                            continue;
                        }

                        // Sample grid colors with bilinear interpolation
                        let mut sampled_colors = vec![vec![Rgb(0, 0, 0); size]; size];
                        for y in 0..size {
                            let v = (y as f32 + 0.5) / size as f32;
                            for x in 0..size {
                                let u = (x as f32 + 0.5) / size as f32;
                                let (ix, iy) = mapping.map(u, v);
                                sampled_colors[y][x] = sample_bilinear(img, ix, iy);
                            }
                        }

                        for &mode in &modes_to_test {
                            let mut grid = Grid::new(size, mode);
                            let mut palette = Rgb::PALETTE[..mode.num_colors()].to_vec();

                            if !is_nano {
                                let spine_x = size / 2;
                                for c in 0..mode.num_colors() {
                                    // SAFE SUBTRACTION: Prevents attempt to subtract with overflow
                                    if c + 1 < size {
                                        let y_pos = size - 1 - c;
                                        palette[c] = sampled_colors[y_pos][spine_x];
                                    }
                                }
                            }

                            for y in 0..size {
                                for x in 0..size {
                                    let color = sampled_colors[y][x];
                                    let best_idx = palette
                                        .iter()
                                        .enumerate()
                                        .min_by_key(|(_, p)| color.distance_sq(p))
                                        .map(|(i, _)| i as u8)
                                        .unwrap_or(0);
                                    grid.set(x, y, best_idx);
                                }
                            }

                            if let Ok(payload) = decoder.decode_grid(&grid, Some(is_nano)) {
                                let confidence = self.compute_confidence(
                                    [p0, p1, p2, p3],
                                    size,
                                    is_nano,
                                    &sampled_colors,
                                    &palette,
                                );

                                let text = match &payload {
                                    DecodedPayload::Text(s) => Some(s.clone()),
                                    DecodedPayload::Dual { public_data, .. } => Some(public_data.clone()),
                                    DecodedPayload::Binary(bytes) => String::from_utf8(bytes.clone()).ok(),
                                };

                                return Ok(UniversalScanResult {
                                    payload,
                                    text,
                                    confidence,
                                    size,
                                    is_nano,
                                    color_mode: mode,
                                    corners: [(p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)],
                                });
                            }
                        }
                    }
                }
            }
        }

        Err(TSpineError::ScanFailed(
            "Could not detect or decode a valid T-Spine barcode from image".to_string(),
        ))
    }

    // ========================================================================
    // Computer Vision Pipeline Internals
    // ========================================================================

    fn detect_candidate_quads(&self, img: &RgbImage) -> Vec<[Point2D; 4]> {
        let (w, h) = img.dimensions();
        let mut quads = Vec::new();

        // 1. Full image bounding rectangle
        quads.push([
            Point2D::new(0.0, 0.0),
            Point2D::new((w - 1) as f32, 0.0),
            Point2D::new((w - 1) as f32, (h - 1) as f32),
            Point2D::new(0.0, (h - 1) as f32),
        ]);

        // 2. Inset boxes for common quiet-zone margins (2%, 4%, 8%)
        for &margin_pct in &[0.02, 0.04, 0.08] {
            let mx = (w as f32 * margin_pct).max(1.0);
            let my = (h as f32 * margin_pct).max(1.0);
            if w as f32 > mx * 2.0 && h as f32 > my * 2.0 {
                quads.push([
                    Point2D::new(mx, my),
                    Point2D::new(w as f32 - 1.0 - mx, my),
                    Point2D::new(w as f32 - 1.0 - mx, h as f32 - 1.0 - my),
                    Point2D::new(mx, h as f32 - 1.0 - my),
                ]);
            }
        }

        // 3. Multi-scale Adaptive Binarization and Contour Extraction
        let max_dim = w.max(h);
        let scale = if max_dim > 700 {
            max_dim as f32 / 700.0
        } else {
            1.0
        };

        let sw = (w as f32 / scale).round().max(10.0) as u32;
        let sh = (h as f32 / scale).round().max(10.0) as u32;

        let mut gray = vec![0u8; (sw * sh) as usize];
        for sy in 0..sh {
            let orig_y = (sy as f32 * scale).min((h - 1) as f32);
            for sx in 0..sw {
                let orig_x = (sx as f32 * scale).min((w - 1) as f32);
                let p = img.get_pixel(orig_x as u32, orig_y as u32);
                gray[(sy * sw + sx) as usize] =
                    ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8;
            }
        }

        // Adaptive block size: larger window prevents quiet zones from washing out
        let block_size = 32u32;
        let blocks_x = (sw + block_size - 1) / block_size;
        let blocks_y = (sh + block_size - 1) / block_size;
        let mut block_means = vec![0u32; (blocks_x * blocks_y) as usize];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut sum = 0u32;
                let mut count = 0u32;
                for y in (by * block_size)..((by + 1) * block_size).min(sh) {
                    for x in (bx * block_size)..((bx + 1) * block_size).min(sw) {
                        sum += gray[(y * sw + x) as usize] as u32;
                        count += 1;
                    }
                }
                block_means[(by * blocks_x + bx) as usize] = if count > 0 { sum / count } else { 128 };
            }
        }

        let mut binary = vec![false; (sw * sh) as usize];
        for y in 0..sh {
            let by = (y / block_size).min(blocks_y.saturating_sub(1));
            for x in 0..sw {
                let bx = (x / block_size).min(blocks_x.saturating_sub(1));
                // Higher margin (10) provides cleaner edges in noisy lighting
                let local_th = (block_means[(by * blocks_x + bx) as usize] as i32 - 10).max(15) as u8;
                if gray[(y * sw + x) as usize] < local_th {
                    binary[(y * sw + x) as usize] = true;
                }
            }
        }

        let mut visited = vec![false; (sw * sh) as usize];
        let mut detected_contours = Vec::new();

        if sh > 4 && sw > 4 {
            for y in 2..(sh - 2) {
                for x in 2..(sw - 2) {
                    let idx = (y * sw + x) as usize;
                    if binary[idx] && !binary[idx - 1] && !visited[idx] {
                        let contour = self.trace_boundary(&binary, sw, sh, x, y, &mut visited);
                        if contour.len() >= 24 {
                            detected_contours.push(contour);
                        }
                    }
                }
            }
        }

        // Polygon approximation with fallback corner reduction for slightly curved quads
        for pts in detected_contours {
            let mut perimeter = 0.0f32;
            for i in 0..pts.len() {
                perimeter += pts[i].dist(&pts[(i + 1) % pts.len()]);
            }

           for eps_factor in &[0.02, 0.035, 0.05, 0.075] {
                let approx = ramer_douglas_peucker(&pts, perimeter * eps_factor);
                let mut clean = approx;
                if clean.len() > 1 && clean[0].dist(&clean[clean.len() - 1]) < 5.0 {
                    clean.pop();
                }

                // Explicitly annotate type as Option<[Point2D; 4]>
                let quad_candidate: Option<[Point2D; 4]> = if clean.len() == 4 && is_convex_quad(&clean) {
                    Some([clean[0], clean[1], clean[2], clean[3]])
                } else if clean.len() >= 5 && clean.len() <= 7 {
                    extract_extreme_quad(&clean)
                } else {
                    None
                };

                if let Some(cand) = quad_candidate {
                    let area = polygon_area(&cand);
                    let d01 = cand[0].dist(&cand[1]);
                    let d12 = cand[1].dist(&cand[2]);
                    let aspect = d01 / d12.max(1e-3);

                    if area > 120.0 && aspect > 0.35 && aspect < 2.8 {
                        let mut quad_pts = [
                            Point2D::new(cand[0].x * scale, cand[0].y * scale),
                            Point2D::new(cand[1].x * scale, cand[1].y * scale),
                            Point2D::new(cand[2].x * scale, cand[2].y * scale),
                            Point2D::new(cand[3].x * scale, cand[3].y * scale),
                        ];

                        let cross = (quad_pts[1].x - quad_pts[0].x) * (quad_pts[2].y - quad_pts[1].y)
                            - (quad_pts[1].y - quad_pts[0].y) * (quad_pts[2].x - quad_pts[1].x);
                        if cross < 0.0 {
                            quad_pts.swap(1, 3);
                        }

                        quads.push(quad_pts);
                        break;
                    }
                }
            }
        }

        let mut unique_quads: Vec<[Point2D; 4]> = Vec::new();
        for q in quads {
            let mut duplicate = false;
            for u in &unique_quads {
                let diff: f32 = (0..4).map(|i| q[i].dist(&u[i])).sum();
                if diff < 15.0 {
                    duplicate = true;
                    break;
                }
            }
            if !duplicate {
                unique_quads.push(q);
            }
        }

        unique_quads
    }

    fn trace_boundary(
        &self,
        binary: &[bool],
        w: u32,
        h: u32,
        sx: u32,
        sy: u32,
        visited: &mut [bool],
    ) -> Vec<Point2D> {
        let mut contour = Vec::new();
        let mut cx = sx;
        let mut cy = sy;
        let dxs = [0, 1, 1, 1, 0, -1, -1, -1];
        let dys = [-1, -1, 0, 1, 1, 1, 0, -1];
        let mut dir = 7;

        for _ in 0..4000 {
            contour.push(Point2D::new(cx as f32, cy as f32));
            visited[(cy * w + cx) as usize] = true;

            let mut found = false;
            for i in 0..8 {
                let next_dir = (dir + 5 + i) % 8;
                let nx = cx as i32 + dxs[next_dir];
                let ny = cy as i32 + dys[next_dir];
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    if binary[(ny as u32 * w + nx as u32) as usize] {
                        cx = nx as u32;
                        cy = ny as u32;
                        dir = next_dir;
                        found = true;
                        break;
                    }
                }
            }

            if !found || (cx == sx && cy == sy && contour.len() > 2) {
                break;
            }
        }

        contour
    }

    fn determine_candidate_sizes(&self, mapping: &ProjectiveMapping, img: &RgbImage) -> Vec<usize> {
        if let Some(forced) = self.forced_size {
            let odd = if forced % 2 == 0 { forced + 1 } else { forced };
            return vec![odd];
        }

        let mut dark_len = 0.0f32;
        let steps = 100;
        for s in 0..steps {
            let v = s as f32 / steps as f32;
            let (ix, iy) = mapping.map(0.25, v);
            let lum = rgb_luminance(&sample_bilinear(img, ix, iy));
            if lum < 128 {
                dark_len = v;
            } else {
                break;
            }
        }

        let mut prioritized = Vec::new();
        if dark_len > 0.003 && dark_len < 0.4 {
            let estimated_size = (1.0 / dark_len).round() as usize;
            let center = if estimated_size % 2 == 0 { estimated_size + 1 } else { estimated_size };
            for delta in &[0isize, -2, 2, -4, 4, -6, 6] {
                let sz = center as isize + delta;
                if sz >= 5 && sz <= 251 {
                    prioritized.push(sz as usize);
                }
            }
        }

        let mut all_sizes: Vec<usize> = (5..=251).step_by(2).collect();
        prioritized.extend(all_sizes.drain(..));
        prioritized.dedup();
        prioritized
    }

    fn verify_skeleton(
        &self,
        mapping: &ProjectiveMapping,
        img: &RgbImage,
        size: usize,
        is_nano: bool,
    ) -> bool {
        let (bl_x, bl_y) = mapping.map(0.5 / size as f32, (size as f32 - 0.5) / size as f32);
        let bl_lum = rgb_luminance(&sample_bilinear(img, bl_x, bl_y));

        let (br_x, br_y) = mapping.map((size as f32 - 0.5) / size as f32, (size as f32 - 0.5) / size as f32);
        let br_lum = rgb_luminance(&sample_bilinear(img, br_x, br_y));

        // Relative check: Bottom-Right must be brighter than Bottom-Left
        if (br_lum as i32) - (bl_lum as i32) < 20 {
            return false;
        }

        if is_nano {
            let t_cells = [(0, 0), (1, 0), (2, 0), (1, 1)];
            for (cx, cy) in t_cells {
                let (ix, iy) = mapping.map((cx as f32 + 0.5) / size as f32, (cy as f32 + 0.5) / size as f32);
                if rgb_luminance(&sample_bilinear(img, ix, iy)) > br_lum.saturating_sub(15) {
                    return false;
                }
            }
        } else {
            let spine_x = size / 2;
            let check_len = size.saturating_sub(8).max(1);
            let mut spine_dark = 0;
            for y in 0..check_len {
                let (ix, iy) = mapping.map((spine_x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
                if rgb_luminance(&sample_bilinear(img, ix, iy)) < br_lum.saturating_sub(20) {
                    spine_dark += 1;
                }
            }
            if spine_dark * 100 / check_len < 70 {
                return false;
            }
        }

        true
    }

    fn compute_confidence(
        &self,
        corners: [Point2D; 4],
        size: usize,
        is_nano: bool,
        sampled: &[Vec<Rgb>],
        palette: &[Rgb],
    ) -> f32 {
        let d01 = corners[0].dist(&corners[1]);
        let d12 = corners[1].dist(&corners[2]);
        let d23 = corners[2].dist(&corners[3]);
        let d30 = corners[3].dist(&corners[0]);

        let r_horiz = d01.min(d23) / d01.max(d23).max(1e-4);
        let r_vert = d12.min(d30) / d12.max(d30).max(1e-4);
        let avg_h = (d01 + d23) * 0.5;
        let avg_v = (d12 + d30) * 0.5;
        let r_aspect = avg_h.min(avg_v) / avg_h.max(avg_v).max(1e-4);

        let mut ortho_sum = 0.0f32;
        for i in 0..4 {
            let p_prev = corners[(i + 3) % 4];
            let p_curr = corners[i];
            let p_next = corners[(i + 1) % 4];
            let v1 = (p_prev.x - p_curr.x, p_prev.y - p_curr.y);
            let v2 = (p_next.x - p_curr.x, p_next.y - p_curr.y);
            let dot = v1.0 * v2.0 + v1.1 * v2.1;
            let l1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt().max(1e-4);
            let l2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt().max(1e-4);
            let cos_th = (dot / (l1 * l2)).abs();
            ortho_sum += (1.0 - cos_th).clamp(0.0, 1.0);
        }
        let s_ortho = ortho_sum * 0.25;
        let s_geom = (r_horiz * r_vert * r_aspect * s_ortho).sqrt().clamp(0.0, 1.0);

        let bl_lum = rgb_luminance(&sampled[size - 1][0]) as f32;
        let br_lum = rgb_luminance(&sampled[size - 1][size - 1]) as f32;
        let s_contrast = ((br_lum - bl_lum) / 255.0).clamp(0.0, 1.0);

        let data_coords = Grid::data_coordinates(size, is_nano);
        let mut margin_sum = 0.0f32;
        let mut count = 0;

        for (x, y) in data_coords {
            let c = sampled[y][x];
            let mut distances: Vec<u32> = palette.iter().map(|p| c.distance_sq(p)).collect();
            distances.sort_unstable();

            if distances.len() >= 2 {
                let d1 = (distances[0] as f32).sqrt();
                let d2 = (distances[1] as f32).sqrt();
                let margin = if d2 > 1e-4 { (d2 - d1) / d2 } else { 1.0 };
                margin_sum += margin.clamp(0.0, 1.0);
            } else {
                margin_sum += 1.0;
            }
            count += 1;
        }

        let s_color = if count > 0 {
            margin_sum / count as f32
        } else {
            1.0
        };

        let total = 0.30 * s_geom + 0.30 * s_contrast + 0.40 * s_color;
        let final_pct = (total * 100.0).clamp(0.0, 100.0);

        if final_pct > 99.5 {
            100.0
        } else {
            (final_pct * 10.0).round() / 10.0
        }
    }
}