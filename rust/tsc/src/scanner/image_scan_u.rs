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
// 2D Geometry, Line Representation & Projective Homography
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

/// A 2D line represented in standard form: a*x + b*y + c = 0, with a^2 + b^2 = 1.
#[derive(Debug, Clone, Copy)]
pub struct Line2D {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl Line2D {
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        let norm = (a * a + b * b).sqrt().max(1e-7);
        Self {
            a: a / norm,
            b: b / norm,
            c: c / norm,
        }
    }

    /// Fits a straight line to a set of points using Total Least Squares (orthogonal regression).
    pub fn fit_orthogonal(pts: &[Point2D]) -> Option<Self> {
        if pts.len() < 2 {
            return None;
        }

        let n = pts.len() as f32;
        let mut mx = 0.0f32;
        let mut my = 0.0f32;
        for p in pts {
            mx += p.x;
            my += p.y;
        }
        mx /= n;
        my /= n;

        let mut sxx = 0.0f32;
        let mut syy = 0.0f32;
        let mut sxy = 0.0f32;
        for p in pts {
            let dx = p.x - mx;
            let dy = p.y - my;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }

        // Angle of normal to the line
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy) + std::f32::consts::FRAC_PI_2;
        let a = theta.cos();
        let b = theta.sin();
        let c = -(a * mx + b * my);

        Some(Self::new(a, b, c))
    }

    /// Computes the intersection point of two lines.
    pub fn intersect(&self, other: &Line2D) -> Option<Point2D> {
        let det = self.a * other.b - self.b * other.a;
        if det.abs() < 1e-5 {
            return None;
        }
        let x = (self.b * other.c - other.b * self.c) / det;
        let y = (other.a * self.c - self.a * other.c) / det;
        Some(Point2D::new(x, y))
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
fn pixel_luminance(p: &image::Rgb<u8>) -> u8 {
    ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8
}

#[inline]
fn pixel_chroma(p: &image::Rgb<u8>) -> u8 {
    let max_c = p[0].max(p[1]).max(p[2]);
    let min_c = p[0].min(p[1]).min(p[2]);
    max_c - min_c
}

#[inline]
fn rgb_luminance(c: &Rgb) -> u8 {
    ((77 * c.0 as u32 + 150 * c.1 as u32 + 29 * c.2 as u32) >> 8) as u8
}

#[inline]
fn rgb_chroma(c: &Rgb) -> u8 {
    let max_c = c.0.max(c.1).max(c.2);
    let min_c = c.0.min(c.1).min(c.2);
    max_c - min_c
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

        for quad in candidate_quads {
            // Test 4 rotations to identify the correct orientation
            for rot in 0..4 {
                let p0 = quad[rot];
                let p1 = quad[(rot + 1) % 4];
                let p2 = quad[(rot + 2) % 4];
                let p3 = quad[(rot + 3) % 4];

                let mapping = match ProjectiveMapping::from_quad(p0, p1, p2, p3) {
                    Some(m) => m,
                    None => continue,
                };

                // Quick Top Bar check: Top edge (v=0.03) must be predominantly dark
                let mut top_dark_samples = 0;
                for step in 1..=5 {
                    let u = step as f32 / 6.0;
                    let (tx, ty) = mapping.map(u, 0.03);
                    if rgb_luminance(&sample_bilinear(img, tx, ty)) < 130 {
                        top_dark_samples += 1;
                    }
                }
                if top_dark_samples < 3 {
                    continue;
                }

                // Relative Orientation Filter:
                // In T-Spine Code: Bottom-Right is WHITE (0), Bottom-Left is BLACK (1)
                let (br_x, br_y) = mapping.map(0.95, 0.95);
                let (bl_x, bl_y) = mapping.map(0.05, 0.95);
                let br_lum = rgb_luminance(&sample_bilinear(img, br_x, br_y)) as i32;
                let bl_lum = rgb_luminance(&sample_bilinear(img, bl_x, bl_y)) as i32;

                if br_lum - bl_lum < 20 {
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

                        // Sample grid colors with subpixel bilinear interpolation
                        let mut sampled_colors = vec![vec![Rgb(0, 0, 0); size]; size];
                        for y in 0..size {
                            let v = (y as f32 + 0.5) / size as f32;
                            for x in 0..size {
                                let u = (x as f32 + 0.5) / size as f32;
                                let (ix, iy) = mapping.map(u, v);
                                sampled_colors[y][x] = sample_bilinear(img, ix, iy);
                            }
                        }

                        // Intelligent Color Mode Detection
                        let modes_to_test = if let Some(m) = self.forced_mode {
                            vec![m]
                        } else {
                            self.determine_mode_test_order(&sampled_colors, size, is_nano)
                        };

                        for &mode in &modes_to_test {
                            let mut grid = Grid::new(size, mode);
                            let palette = self.build_calibrated_palette(
                                &sampled_colors,
                                size,
                                mode,
                                is_nano,
                            );

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
                                    DecodedPayload::Dual { public_data, .. } => {
                                        Some(public_data.clone())
                                    }
                                    DecodedPayload::Binary(bytes) => {
                                        String::from_utf8(bytes.clone()).ok()
                                    }
                                };

                                return Ok(UniversalScanResult {
                                    payload,
                                    text,
                                    confidence,
                                    size,
                                    is_nano,
                                    color_mode: mode,
                                    corners: [
                                        (p0.x, p0.y),
                                        (p1.x, p1.y),
                                        (p2.x, p2.y),
                                        (p3.x, p3.y),
                                    ],
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
    // Robust Color Mode Identification & Palette Calibration
    // ========================================================================

    /// Dynamically determines the candidate test order for ColorMode.
    /// Eliminates false positives (such as 2-color identified as 4-color, or 4-color as 8-color)
    /// by combining spine calibration decoding with full-grid chromaticity analysis.
    fn determine_mode_test_order(
        &self,
        sampled: &[Vec<Rgb>],
        size: usize,
        is_nano: bool,
    ) -> Vec<ColorMode> {
        let detected = self.identify_color_mode(sampled, size, is_nano);
        match detected {
            ColorMode::Monochrome => vec![
                ColorMode::Monochrome,
                ColorMode::FourColor,
                ColorMode::EightColor,
            ],
            ColorMode::FourColor => vec![
                ColorMode::FourColor,
                ColorMode::EightColor,
                ColorMode::Monochrome,
            ],
            ColorMode::EightColor => vec![
                ColorMode::EightColor,
                ColorMode::FourColor,
                ColorMode::Monochrome,
            ],
        }
    }

    fn identify_color_mode(&self, sampled: &[Vec<Rgb>], size: usize, is_nano: bool) -> ColorMode {
        // 1. Standard mode: inspect the spine calibration cells directly
        if !is_nano && size >= 9 {
            let spine_x = size / 2;

            // In T-Spine:
            // y = size - 1 -> White (c = 0)
            // y = size - 2 -> Black (c = 1)
            // y = size - 3 -> Red   (c = 2 in FourColor & EightColor; Black in Monochrome)
            // y = size - 5 -> Green (c = 4 in EightColor; Black in FourColor & Monochrome)
            let c_red_candidate = sampled[size - 3][spine_x];
            let is_spine_red = c_red_candidate.0 > 100
                && (c_red_candidate.0 as i32 - c_red_candidate.1 as i32) > 30
                && (c_red_candidate.0 as i32 - c_red_candidate.2 as i32) > 30;

            if is_spine_red {
                if size >= 11 {
                    let c_green_candidate = sampled[size - 5][spine_x];
                    let is_spine_green = c_green_candidate.1 > 90
                        && (c_green_candidate.1 as i32 - c_green_candidate.0 as i32) > 25
                        && (c_green_candidate.1 as i32 - c_green_candidate.2 as i32) > 25;

                    if is_spine_green {
                        return ColorMode::EightColor;
                    }
                }
                return ColorMode::FourColor;
            } else {
                // If cell at size - 3 is not Red, it cannot be 4-color or 8-color
                return ColorMode::Monochrome;
            }
        }

        // 2. Data cells chromaticity analysis (effective for Nano mode and fallback)
        let data_coords = Grid::data_coordinates(size, is_nano);
        let mut chromatic_count = 0;
        let mut eight_color_only_count = 0;

        for &(x, y) in &data_coords {
            let c = sampled[y][x];
            let chroma = rgb_chroma(&c);

            if chroma > 45 {
                chromatic_count += 1;

                // EightColor unique signatures:
                // Green: high G, low R, low B
                // Cyan: high G & B, low R
                // Yellow: high R & G, low B
                // Magenta: high R & B, low G
                let r = c.0 as i32;
                let g = c.1 as i32;
                let b = c.2 as i32;

                let is_green = g > 90 && (g - r) > 25 && (g - b) > 25;
                let is_cyan = g > 90 && b > 90 && (r < 80);
                let is_yellow = r > 100 && g > 100 && (b < 80);
                let is_magenta = r > 100 && b > 100 && (g < 80);

                if is_green || is_cyan || is_yellow || is_magenta {
                    eight_color_only_count += 1;
                }
            }
        }

        let total_cells = data_coords.len().max(1) as f32;
        let chromatic_ratio = chromatic_count as f32 / total_cells;
        let eight_color_ratio = eight_color_only_count as f32 / total_cells;

        if chromatic_ratio < 0.03 {
            ColorMode::Monochrome
        } else if eight_color_ratio > 0.025 {
            ColorMode::EightColor
        } else {
            ColorMode::FourColor
        }
    }

    /// Builds a calibrated palette strictly bounded by the number of colors of the target mode.
    /// Prevents black spine entries from bleeding into and corrupting color palettes.
    fn build_calibrated_palette(
        &self,
        sampled: &[Vec<Rgb>],
        size: usize,
        mode: ColorMode,
        is_nano: bool,
    ) -> Vec<Rgb> {
        let mut palette = Rgb::PALETTE[..mode.num_colors()].to_vec();

        if !is_nano {
            let spine_x = size / 2;
            let num_colors = mode.num_colors();

            // Calibrate strictly up to mode.num_colors()
            for c in 0..num_colors {
                if c + 1 < size {
                    let y_pos = size - 1 - c;
                    let sample = sampled[y_pos][spine_x];

                    // Sanity check sample against expected color family
                    let valid_sample = match c {
                        0 => rgb_luminance(&sample) > 120,                          // White
                        1 => rgb_luminance(&sample) < 135,                          // Black
                        2 => sample.0 > 80 && sample.0 > sample.1 && sample.0 > sample.2, // Red
                        3 => sample.2 > 80 && sample.2 > sample.0 && sample.2 > sample.1, // Blue
                        4 => sample.1 > 80 && sample.1 > sample.0 && sample.1 > sample.2, // Green
                        _ => true,
                    };

                    if valid_sample {
                        palette[c] = sample;
                    }
                }
            }
        }

        palette
    }

    // ========================================================================
    // Computer Vision Pipeline: Side Detection & Corner Intersection
    // ========================================================================

    fn detect_candidate_quads(&self, img: &RgbImage) -> Vec<[Point2D; 4]> {
        let (w, h) = img.dimensions();
        let mut quads = Vec::new();

        // 1. Full image bounding rectangle (ideal for unpadded digital exports)
        quads.push([
            Point2D::new(0.0, 0.0),
            Point2D::new((w - 1) as f32, 0.0),
            Point2D::new((w - 1) as f32, (h - 1) as f32),
            Point2D::new(0.0, (h - 1) as f32),
        ]);

        // 2. Fixed quiet-zone insets (2%, 4%, 8%)
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

        // 3. Side-detection from white background boundary transitions
        if let Some(side_quad) = self.detect_quad_by_side_sweeping(img) {
            quads.push(side_quad);
        }

        // 4. Black Bar + Spine detector with side verification
        quads.extend(self.detect_quads_from_black_bars(img));

        // 5. Adaptive contour line-segment intersection
        quads.extend(self.detect_quads_from_contours(img));

        // Deduplicate and filter degenerate quads
        let mut unique_quads: Vec<[Point2D; 4]> = Vec::new();
        for q in quads {
            let area = polygon_area(&q);
            if area < 100.0 {
                continue;
            }

            let mut duplicate = false;
            for u in &unique_quads {
                let diff: f32 = (0..4).map(|i| q[i].dist(&u[i])).sum();
                if diff < 12.0 {
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

    /// Detects the 4 sides by ray-casting inwards from the image borders.
    /// Since the barcode is framed by a uniform white background, the transition points
    /// from white background to code modules define the 4 boundary lines.
    /// Corners are computed as the exact intersection of these lines.
    fn detect_quad_by_side_sweeping(&self, img: &RgbImage) -> Option<[Point2D; 4]> {
        let (w, h) = img.dimensions();
        if w < 20 || h < 20 {
            return None;
        }

        // Estimate background luminance from the image corners
        let c00 = pixel_luminance(img.get_pixel(0, 0));
        let c10 = pixel_luminance(img.get_pixel(w - 1, 0));
        let c01 = pixel_luminance(img.get_pixel(0, h - 1));
        let c11 = pixel_luminance(img.get_pixel(w - 1, h - 1));
        let bg_lum = ((c00 as u32 + c10 as u32 + c01 as u32 + c11 as u32) / 4) as u8;

        if bg_lum < 160 {
            return None; // Background is not sufficiently bright
        }

        let is_code_pixel = |p: &image::Rgb<u8>| -> bool {
            let lum = pixel_luminance(p);
            let chroma = pixel_chroma(p);
            // Non-white code module: either significantly darker than background or colored
            (lum as i32) < (bg_lum as i32 - 35) || chroma > 40
        };

        let num_rays = 32;
        let mut top_pts = Vec::new();
        let mut bottom_pts = Vec::new();
        let mut left_pts = Vec::new();
        let mut right_pts = Vec::new();

        // Sweep vertically: Top & Bottom
        for i in 1..num_rays {
            let x = (i * w) / num_rays;

            // From Top downward
            for y in 0..(h / 2) {
                if is_code_pixel(img.get_pixel(x, y)) {
                    top_pts.push(Point2D::new(x as f32, y as f32));
                    break;
                }
            }

            // From Bottom upward
            for y in (h / 2..h).rev() {
                if is_code_pixel(img.get_pixel(x, y)) {
                    bottom_pts.push(Point2D::new(x as f32, y as f32));
                    break;
                }
            }
        }

        // Sweep horizontally: Left & Right
        for i in 1..num_rays {
            let y = (i * h) / num_rays;

            // From Left inward
            for x in 0..(w / 2) {
                if is_code_pixel(img.get_pixel(x, y)) {
                    left_pts.push(Point2D::new(x as f32, y as f32));
                    break;
                }
            }

            // From Right inward
            for x in (w / 2..w).rev() {
                if is_code_pixel(img.get_pixel(x, y)) {
                    right_pts.push(Point2D::new(x as f32, y as f32));
                    break;
                }
            }
        }

        if top_pts.len() < 4 || bottom_pts.len() < 4 || left_pts.len() < 4 || right_pts.len() < 4 {
            return None;
        }

        let line_top = Line2D::fit_orthogonal(&top_pts)?;
        let line_bottom = Line2D::fit_orthogonal(&bottom_pts)?;
        let line_left = Line2D::fit_orthogonal(&left_pts)?;
        let line_right = Line2D::fit_orthogonal(&right_pts)?;

        let p_tl = line_top.intersect(&line_left)?;
        let p_tr = line_top.intersect(&line_right)?;
        let p_br = line_bottom.intersect(&line_right)?;
        let p_bl = line_bottom.intersect(&line_left)?;

        let quad = [p_tl, p_tr, p_br, p_bl];
        if is_convex_quad(&quad) {
            let area = polygon_area(&quad);
            let d_top = p_tl.dist(&p_tr);
            let d_left = p_tl.dist(&p_bl);
            let aspect = d_top / d_left.max(1e-3);

            if area > 120.0 && aspect > 0.6 && aspect < 1.6 {
                return Some(quad);
            }
        }

        None
    }

    /// Identifies the continuous horizontal black bar of the T-Spine,
    /// checks for the vertical spine stem, and verifies companion sides.
    /// Filters out spurious black bars (underlines, borders, table rules).
    fn detect_quads_from_black_bars(&self, img: &RgbImage) -> Vec<[Point2D; 4]> {
        let (w, h) = img.dimensions();
        let mut quads = Vec::new();

        let step_y = (h / 60).max(1);
        for y in (2..h.saturating_sub(10)).step_by(step_y as usize) {
            let mut dark_start = None;

            for x in 0..w {
                let lum = pixel_luminance(img.get_pixel(x, y));
                let is_dark = lum < 110;

                match (dark_start, is_dark) {
                    (None, true) => dark_start = Some(x),
                    (Some(start_x), false) => {
                        let bar_len = x - start_x;
                        if bar_len >= 20 {
                            if let Some(quad) = self.verify_and_build_bar_quad(img, start_x, x - 1, y) {
                                quads.push(quad);
                            }
                        }
                        dark_start = None;
                    }
                    _ => {}
                }
            }

            if let Some(start_x) = dark_start {
                let bar_len = w - start_x;
                if bar_len >= 20 {
                    if let Some(quad) = self.verify_and_build_bar_quad(img, start_x, w - 1, y) {
                        quads.push(quad);
                    }
                }
            }
        }

        quads
    }

    /// Validates whether a candidate black bar belongs to a T-Spine code.
    /// Rejects false positives by testing for the central stem and companion side boundaries.
    fn verify_and_build_bar_quad(
        &self,
        img: &RgbImage,
        x0: u32,
        x1: u32,
        y: u32,
    ) -> Option<[Point2D; 4]> {
        let (_, h) = img.dimensions();
        let bar_len = (x1 - x0) as f32;
        let expected_size = bar_len;

        if y as f32 + expected_size * 0.7 > h as f32 {
            return None;
        }

        // 1. Central Stem (Spine) Verification:
        // A genuine T-Spine must have a vertical dark stem extending downward from near the center.
        let mid_x = (x0 + x1) / 2;
        let spine_check_depth = (expected_size * 0.45).min((h - 1 - y) as f32) as u32;
        if spine_check_depth < 6 {
            return None;
        }

        let mut spine_dark_count = 0;
        for dy in 1..=spine_check_depth {
            let lum = pixel_luminance(img.get_pixel(mid_x, y + dy));
            if lum < 130 {
                spine_dark_count += 1;
            }
        }

        // If the central stem is absent (< 65% dark), this is an unwanted black bar (e.g. underline or border)
        if spine_dark_count * 100 / spine_check_depth < 65 {
            return None;
        }

        // 2. Flank Activity Check:
        // Regions to the left and right of the spine stem must contain code modules,
        // not uniform background or solid black fill.
        let left_sample_x = x0 + (bar_len * 0.25) as u32;
        let right_sample_x = x0 + (bar_len * 0.75) as u32;
        let mid_y = y + (spine_check_depth / 2);

        let lum_left = pixel_luminance(img.get_pixel(left_sample_x, mid_y));
        let lum_right = pixel_luminance(img.get_pixel(right_sample_x, mid_y));

        // Reject if entirely uniform or matches pure white background
        if lum_left > 240 && lum_right > 240 {
            return None;
        }

        // 3. Side & Bottom Boundary Matching:
        // Find the bottom boundary by scanning upward around expected depth
        let bottom_target_y = (y as f32 + expected_size).min((h - 1) as f32) as u32;
        let mut detected_bottom_y = None;

        let search_y_min = (y as f32 + expected_size * 0.7) as u32;
        let search_y_max = (y as f32 + expected_size * 1.3).min((h - 1) as f32) as u32;

        for check_y in (search_y_min..search_y_max).rev() {
            let lum_bl = pixel_luminance(img.get_pixel(x0, check_y));
            let lum_br = pixel_luminance(img.get_pixel(x1, check_y));

            // Bottom row has black at bottom-left and white at bottom-right
            if lum_bl < 130 && lum_br > 130 {
                detected_bottom_y = Some(check_y);
                break;
            }
        }

        let bottom_y = detected_bottom_y.unwrap_or(bottom_target_y) as f32;

        let p_tl = Point2D::new(x0 as f32, y as f32);
        let p_tr = Point2D::new(x1 as f32, y as f32);
        let p_br = Point2D::new(x1 as f32, bottom_y);
        let p_bl = Point2D::new(x0 as f32, bottom_y);

        let quad = [p_tl, p_tr, p_br, p_bl];
        if is_convex_quad(&quad) {
            Some(quad)
        } else {
            None
        }
    }

    /// Enhanced contour-based quad extraction using robust side-line fitting.
    /// Avoids cutting corners by intersecting fitted side lines.
    fn detect_quads_from_contours(&self, img: &RgbImage) -> Vec<[Point2D; 4]> {
        let (w, h) = img.dimensions();
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
                gray[(sy * sw + sx) as usize] = pixel_luminance(p);
            }
        }

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

        let mut quads = Vec::new();
        for pts in detected_contours {
            // Find centroid
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            for p in &pts {
                cx += p.x;
                cy += p.y;
            }
            cx /= pts.len() as f32;
            cy /= pts.len() as f32;

            // Partition contour points into 4 sides based on quadrant angles
            let mut side_top = Vec::new();
            let mut side_right = Vec::new();
            let mut side_bottom = Vec::new();
            let mut side_left = Vec::new();

            for p in &pts {
                let dx = p.x - cx;
                let dy = p.y - cy;
                let angle = dy.atan2(dx); // -PI to +PI

                if angle >= -std::f32::consts::FRAC_PI_4 * 3.0 && angle < -std::f32::consts::FRAC_PI_4 {
                    side_top.push(*p);
                } else if angle >= -std::f32::consts::FRAC_PI_4 && angle < std::f32::consts::FRAC_PI_4 {
                    side_right.push(*p);
                } else if angle >= std::f32::consts::FRAC_PI_4 && angle < std::f32::consts::FRAC_PI_4 * 3.0 {
                    side_bottom.push(*p);
                } else {
                    side_left.push(*p);
                }
            }

            if side_top.len() < 3 || side_right.len() < 3 || side_bottom.len() < 3 || side_left.len() < 3 {
                continue;
            }

            let l_top = match Line2D::fit_orthogonal(&side_top) {
                Some(l) => l,
                None => continue,
            };
            let l_right = match Line2D::fit_orthogonal(&side_right) {
                Some(l) => l,
                None => continue,
            };
            let l_bottom = match Line2D::fit_orthogonal(&side_bottom) {
                Some(l) => l,
                None => continue,
            };
            let l_left = match Line2D::fit_orthogonal(&side_left) {
                Some(l) => l,
                None => continue,
            };

            let p0 = match l_top.intersect(&l_left) {
                Some(p) => Point2D::new(p.x * scale, p.y * scale),
                None => continue,
            };
            let p1 = match l_top.intersect(&l_right) {
                Some(p) => Point2D::new(p.x * scale, p.y * scale),
                None => continue,
            };
            let p2 = match l_bottom.intersect(&l_right) {
                Some(p) => Point2D::new(p.x * scale, p.y * scale),
                None => continue,
            };
            let p3 = match l_bottom.intersect(&l_left) {
                Some(p) => Point2D::new(p.x * scale, p.y * scale),
                None => continue,
            };

            let mut quad = [p0, p1, p2, p3];
            let area = polygon_area(&quad);
            let d01 = quad[0].dist(&quad[1]);
            let d12 = quad[1].dist(&quad[2]);
            let aspect = d01 / d12.max(1e-3);

            if area > 120.0 && aspect > 0.4 && aspect < 2.5 {
                let cross = (quad[1].x - quad[0].x) * (quad[2].y - quad[1].y)
                    - (quad[1].y - quad[0].y) * (quad[2].x - quad[1].x);
                if cross < 0.0 {
                    quad.swap(1, 3);
                }
                if is_convex_quad(&quad) {
                    quads.push(quad);
                }
            }
        }

        quads
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

    // ========================================================================
    // Grid Size & Skeleton Verification
    // ========================================================================

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
            let center = if estimated_size % 2 == 0 {
                estimated_size + 1
            } else {
                estimated_size
            };
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

        let (br_x, br_y) = mapping.map(
            (size as f32 - 0.5) / size as f32,
            (size as f32 - 0.5) / size as f32,
        );
        let br_lum = rgb_luminance(&sample_bilinear(img, br_x, br_y));

        // Bottom-Right must be visibly brighter than Bottom-Left
        if (br_lum as i32) - (bl_lum as i32) < 18 {
            return false;
        }

        if is_nano {
            let t_cells = [(0, 0), (1, 0), (2, 0), (1, 1)];
            for (cx, cy) in t_cells {
                let (ix, iy) = mapping.map(
                    (cx as f32 + 0.5) / size as f32,
                    (cy as f32 + 0.5) / size as f32,
                );
                if rgb_luminance(&sample_bilinear(img, ix, iy)) > br_lum.saturating_sub(15) {
                    return false;
                }
            }
        } else {
            let spine_x = size / 2;
            let check_len = size.saturating_sub(8).max(1);
            let mut spine_dark = 0;
            for y in 0..check_len {
                let (ix, iy) = mapping.map(
                    (spine_x as f32 + 0.5) / size as f32,
                    (y as f32 + 0.5) / size as f32,
                );
                if rgb_luminance(&sample_bilinear(img, ix, iy)) < br_lum.saturating_sub(20) {
                    spine_dark += 1;
                }
            }
            if spine_dark * 100 / check_len < 65 {
                return false;
            }
        }

        true
    }

    // ========================================================================
    // Confidence Computation
    // ========================================================================

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