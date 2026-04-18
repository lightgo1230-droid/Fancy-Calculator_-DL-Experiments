#![windows_subsystem = "windows"]

use eframe::egui::{
    self, Color32, FontId, Margin, Painter, Pos2, Rect,
    RichText, Rounding, Stroke, Vec2,
};
use egui_plot::{Line, Plot, PlotPoint, PlotPoints, Points, VLine};
use serde_json::Value;
use std::fs;

const DATA_DIR: &str = r"C:\Users\USER\OneDrive\Desktop\youtube_predictor\data";
const MONTHS: [&str; 12] = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"];

// ── Palette ───────────────────────────────────────────────────
const C_BG:      Color32 = Color32::from_rgb(10, 12, 18);
const C_SURFACE: Color32 = Color32::from_rgb(18, 22, 32);
const C_CARD:    Color32 = Color32::from_rgb(24, 29, 42);
const C_BORDER:  Color32 = Color32::from_rgb(44, 52, 72);
const C_TEXT:    Color32 = Color32::from_rgb(218, 228, 245);
const C_MUTED:   Color32 = Color32::from_rgb(100, 112, 140);
const C_ACTUAL:  Color32 = Color32::from_rgb(148, 163, 184);  // gray-blue
const C_BASIC:   Color32 = Color32::from_rgb(96,  165, 250);  // blue
const C_ADV:     Color32 = Color32::from_rgb(52,  211, 153);  // emerald
const C_SEL:     Color32 = Color32::from_rgb(251, 191,  36);  // amber
const C_UP:      Color32 = Color32::from_rgb(52,  211, 153);
const C_DOWN:    Color32 = Color32::from_rgb(248,  113, 113);

// ── Model inference ───────────────────────────────────────────
struct Linear { w: Vec<Vec<f32>>, b: Vec<f32> }
struct BN     { w: Vec<f32>, b: Vec<f32>, mean: Vec<f32>, var: Vec<f32>, eps: f32 }
struct Block  { l1: Linear, b1: BN, l2: Linear, b2: BN }
struct Model  { sl: Linear, sbn: BN, blocks: Vec<Block>, hl1: Linear, hl2: Linear }
struct Tmpl   { subs_idx: usize, data: Vec<Vec<f32>>, slope: f32,
                last_subs: f32, sc_mean: Vec<f32>, sc_scale: Vec<f32> }

fn vf(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn mf(v: &Value) -> Vec<Vec<f32>> {
    v.as_array().unwrap().iter().map(|r| vf(r)).collect()
}
fn plin(v: &Value) -> Linear { Linear { w: mf(&v["weight"]), b: vf(&v["bias"]) } }
fn pbn(v: &Value) -> BN {
    BN { w: vf(&v["weight"]), b: vf(&v["bias"]),
         mean: vf(&v["running_mean"]), var: vf(&v["running_var"]),
         eps: v["eps"].as_f64().unwrap() as f32 }
}
fn load_model(path: &str) -> Model {
    let j: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let nb = j["n_blocks"].as_u64().unwrap() as usize;
    Model {
        sl: plin(&j["stem"]["linear"]), sbn: pbn(&j["stem"]["bn"]),
        blocks: (0..nb).map(|i| {
            let b = &j["blocks"][i];
            Block { l1: plin(&b["linear1"]), b1: pbn(&b["bn1"]),
                    l2: plin(&b["linear2"]), b2: pbn(&b["bn2"]) }
        }).collect(),
        hl1: plin(&j["head"]["linear1"]), hl2: plin(&j["head"]["linear2"]),
    }
}
fn load_tmpl(path: &str) -> Tmpl {
    let j: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    Tmpl {
        subs_idx:  j["subs_feature_idx"].as_u64().unwrap() as usize,
        data:      (1..=12).map(|m| vf(&j["templates"][m.to_string()])).collect(),
        slope:     j["subs_slope_per_month"].as_f64().unwrap() as f32,
        last_subs: j["last_log_subs"].as_f64().unwrap() as f32,
        sc_mean:   vf(&j["scaler_mean"]),
        sc_scale:  vf(&j["scaler_scale"]),
    }
}
fn fwd_lin(x: &[f32], l: &Linear) -> Vec<f32> {
    l.w.iter().zip(&l.b)
        .map(|(row, &b)| b + row.iter().zip(x).map(|(&w,&xi)| w*xi).sum::<f32>())
        .collect()
}
fn fwd_bn(x: &[f32], bn: &BN) -> Vec<f32> {
    x.iter().enumerate().map(|(i,&xi)|
        bn.w[i]*(xi-bn.mean[i])/(bn.var[i]+bn.eps).sqrt()+bn.b[i]).collect()
}
fn gelu(x: f32) -> f32 {
    x*0.5*(1.0+(0.7978845608_f32*(x+0.044715*x*x*x)).tanh())
}
fn act(v: Vec<f32>) -> Vec<f32> { v.into_iter().map(gelu).collect() }
fn fwd_block(x: &[f32], b: &Block) -> Vec<f32> {
    let h = act(fwd_bn(&fwd_lin(x, &b.l1), &b.b1));
    let h = fwd_bn(&fwd_lin(&h, &b.l2), &b.b2);
    x.iter().zip(&h).map(|(&xi,&hi)| gelu(xi+hi)).collect()
}
fn infer(m: &Model, xs: &[f32]) -> f32 {
    let mut x = act(fwd_bn(&fwd_lin(xs,&m.sl),&m.sbn));
    for b in &m.blocks { x = fwd_block(&x,b); }
    fwd_lin(&act(fwd_lin(&x,&m.hl1)),&m.hl2)[0]
}
fn gen_fc(model: &Model, t: &Tmpl, n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        let mut feat = t.data[i%12].clone();
        feat[t.subs_idx] = t.last_subs + t.slope*(i as f32+1.0);
        let xs: Vec<f32> = feat.iter().zip(&t.sc_mean).zip(&t.sc_scale)
            .map(|((v,m),s)| (v-m)/s).collect();
        ((infer(model,&xs) as f64).exp()-1.0) as f32/1_000_000.0
    }).collect()
}

// ── Dataset ───────────────────────────────────────────────────
struct Data {
    labels:   Vec<String>,
    hist:     Vec<f32>,
    fc_bas:   Vec<f32>,
    fc_adv:   Vec<f32>,
    hist_len: usize,
}

fn load_data() -> Result<Data, String> {
    let dj: Value = serde_json::from_str(
        &fs::read_to_string(format!("{DATA_DIR}/historical_monthly.json"))
            .map_err(|e| e.to_string())?
    ).map_err(|e| e.to_string())?;

    let hist_labels: Vec<String> = dj["labels"].as_array().unwrap()
        .iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let hist_raw: Vec<f32> = dj["views"].as_array().unwrap()
        .iter().map(|v| (v.as_f64().unwrap()/1_000_000.0) as f32).collect();

    let ma = load_model(&format!("{DATA_DIR}/model_a_weights.json"));
    let mb = load_model(&format!("{DATA_DIR}/model_b_weights.json"));
    let ta = load_tmpl(&format!("{DATA_DIR}/templates_a.json"));
    let tb = load_tmpl(&format!("{DATA_DIR}/templates_b.json"));
    let raw_a = gen_fc(&ma, &ta, 36);
    let raw_b = gen_fc(&mb, &tb, 36);

    let fc_labels: Vec<String> = (0..36)
        .map(|i| format!("{}  {}", ["2026","2027","2028"][i/12], MONTHS[i%12]))
        .collect();

    let hl  = hist_labels.len();
    let tot = hl + 36;
    let mut labels = hist_labels;
    labels.extend(fc_labels);

    // ── Anchor forecast level to last historical value ──────────
    // The model predicts relative trends; we rescale so the forecast
    // line starts exactly where the historical line ends.
    // This preserves the month-over-month trend shape while ensuring
    // visual continuity with the actual data.
    let last_h = *hist_raw.last().unwrap_or(&1.0);
    let scale_a = if raw_a[0] > 0.0 { last_h / raw_a[0] } else { 1.0 };
    let scale_b = if raw_b[0] > 0.0 { last_h / raw_b[0] } else { 1.0 };
    let raw_a: Vec<f32> = raw_a.iter().map(|&v| v * scale_a).collect();
    let raw_b: Vec<f32> = raw_b.iter().map(|&v| v * scale_b).collect();

    let mut hist = hist_raw;
    hist.extend(vec![f32::NAN; 36]);

    // Bridge: first forecast point = last historical value → visual continuity.
    // Guard hl==0 to prevent usize underflow.
    let mut fc_bas = vec![f32::NAN; tot];
    let mut fc_adv = vec![f32::NAN; tot];
    if hl > 0 {
        fc_bas[hl - 1] = last_h;
        fc_adv[hl - 1] = last_h;
    }
    for i in 0..36 { fc_bas[hl+i] = raw_a[i]; fc_adv[hl+i] = raw_b[i]; }

    Ok(Data { labels, hist, fc_bas, fc_adv, hist_len: hl })
}

// ── View range ────────────────────────────────────────────────
#[derive(PartialEq, Clone, Copy, Debug)]
enum View { All, Y2026, Y2027, Y2028 }

impl View {
    fn label(self) -> &'static str {
        match self { View::All=>"All Years", View::Y2026=>"2026", View::Y2027=>"2027", View::Y2028=>"2028" }
    }
    fn x_range(self, hl: usize) -> (usize, usize) {
        match self {
            View::All  => (0,       hl+35),
            View::Y2026 => (hl,     hl+11),
            View::Y2027 => (hl+12,  hl+23),
            View::Y2028 => (hl+24,  hl+35),
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────
fn fmt_m(v: f32) -> String {
    if v.is_nan() { "—".into() }
    else { format!("{:.3}M", v) }
}
fn fmt_big(v: f32) -> String {
    if v.is_nan() { "—".into() }
    else if v >= 10.0 { format!("{:.1}M", v) }
    else { format!("{:.2}M", v) }
}
fn mom(now: f32, prev: Option<f32>) -> Option<f32> {
    prev.filter(|&p| !p.is_nan() && p > 0.001 && !now.is_nan())
        .map(|p| (now-p)/p*100.0)
}

// ── App ───────────────────────────────────────────────────────
struct App {
    cursor: usize,
    view:   View,
    data:   Result<Data, String>,
}
impl App {
    fn new() -> Self {
        let data   = load_data();
        let cursor = data.as_ref().map(|d| d.hist_len).unwrap_or(0);
        Self { cursor, view: View::All, data }
    }
}

// ── Rendering helpers ─────────────────────────────────────────

fn paint_accent_bar(painter: &Painter, rect: Rect, color: Color32) {
    let bar = Rect::from_min_size(
        rect.min,
        Vec2::new(3.0, rect.height()),
    );
    painter.rect_filled(bar, Rounding::same(2.0), color);
}

/// Draws a single metric card: label on top, big value, optional MoM badge
fn metric_card(
    ui: &mut egui::Ui,
    label: &str, value: &str,
    mom_pct: Option<f32>,
    accent: Color32,
    width: f32, height: f32,
) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(width, height), egui::Sense::hover());
    let r = resp.rect;

    // background + border
    painter.rect_filled(r, Rounding::same(10.0), C_CARD);
    painter.rect_stroke(r, Rounding::same(10.0), Stroke::new(1.0, C_BORDER));
    // accent bar (left side)
    paint_accent_bar(&painter, r, accent);

    let inner = r.shrink2(Vec2::new(14.0, 10.0));
    let inner = Rect::from_min_size(
        Pos2::new(inner.min.x + 6.0, inner.min.y),
        Vec2::new(inner.width() - 6.0, inner.height()),
    );

    // label
    painter.text(
        Pos2::new(inner.min.x, inner.min.y + 2.0),
        egui::Align2::LEFT_TOP,
        label,
        FontId::proportional(11.0),
        C_MUTED,
    );

    // big value
    painter.text(
        Pos2::new(inner.min.x, inner.min.y + 18.0),
        egui::Align2::LEFT_TOP,
        value,
        FontId::proportional(26.0),
        C_TEXT,
    );

    // MoM badge
    if let Some(pct) = mom_pct {
        let (arrow, col) = if pct >= 0.0 { ("▲", C_UP) } else { ("▼", C_DOWN) };
        let txt = format!("{arrow} {pct:+.1}% from last month");
        painter.text(
            Pos2::new(inner.min.x, inner.min.y + 52.0),
            egui::Align2::LEFT_TOP,
            &txt,
            FontId::proportional(11.5),
            col,
        );
    }
}

/// Small legend pill
fn legend_item(ui: &mut egui::Ui, color: Color32, text: &str) {
    let (r, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(r.center(), 5.0, color);
    ui.add_space(5.0);
    ui.label(RichText::new(text).size(11.5).color(C_TEXT));
    ui.add_space(18.0);
}

/// View toggle pill button — returns true if clicked
fn view_btn(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let (fill, text_col, stroke_col) = if active {
        (C_SEL, C_BG, C_SEL)
    } else {
        (C_CARD, C_MUTED, C_BORDER)
    };
    let btn = egui::Button::new(RichText::new(label).size(12.0).color(text_col).strong())
        .fill(fill)
        .stroke(Stroke::new(1.0, stroke_col))
        .rounding(Rounding::same(16.0))
        .min_size(Vec2::new(72.0, 28.0));
    ui.add(btn).clicked()
}

// ── Main render ───────────────────────────────────────────────
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {

        // ── Header ───────────────────────────────────────────
        egui::TopBottomPanel::top("hdr")
            .exact_height(52.0)
            .frame(egui::Frame::default()
                .fill(C_SURFACE)
                .inner_margin(Margin::symmetric(20.0, 0.0))
                .stroke(Stroke::new(1.0, C_BORDER)))
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // YouTube red dot
                    let (r,_) = ui.allocate_exact_size(Vec2::splat(18.0), egui::Sense::hover());
                    ui.painter().circle_filled(r.center(), 9.0,
                        Color32::from_rgb(255, 50, 50));

                    ui.add_space(8.0);
                    ui.label(RichText::new("YouTube Views Dashboard")
                        .size(18.0).strong().color(C_TEXT));
                    ui.add_space(16.0);
                    ui.label(RichText::new("Actual 2020 – 2025   ·   AI Prediction 2026 – 2028")
                        .size(12.0).color(C_MUTED));

                    // Right: view range buttons
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        for &v in &[View::Y2028, View::Y2027, View::Y2026, View::All] {
                            if view_btn(ui, v.label(), self.view == v) {
                                self.view = v;
                                let (lo, _) = v.x_range(
                                    self.data.as_ref().map(|d| d.hist_len).unwrap_or(72)
                                );
                                self.cursor = lo;
                            }
                            ui.add_space(6.0);
                        }
                    });
                });
            });

        // ── Interpretation bar ───────────────────────────────
        egui::TopBottomPanel::bottom("interp")
            .exact_height(62.0)
            .frame(egui::Frame::default()
                .fill(Color32::from_rgb(18, 22, 32))
                .inner_margin(Margin::symmetric(20.0, 10.0))
                .stroke(Stroke::new(1.0, C_BORDER)))
            .show(ctx, |ui| {
                if let Ok(d) = &self.data {
                    let total    = d.labels.len();
                    let hl       = d.hist_len;
                    let cursor   = self.cursor.min(total - 1);
                    let is_fc    = cursor >= hl;
                    let h_val    = d.hist[cursor];
                    let b_val    = if cursor >= hl { d.fc_bas[cursor] } else { f32::NAN };
                    let v_val    = if cursor >= hl { d.fc_adv[cursor] } else { f32::NAN };
                    let lbl      = &d.labels[cursor];

                    // peak of all history
                    let peak_h = d.hist[..hl].iter().filter(|v| !v.is_nan())
                        .cloned().fold(0.0_f32, f32::max);
                    // 2025 avg (fallback NAN when no data, prevents misleading percentages)
                    let avg25_vals: Vec<f32> = d.hist[hl.saturating_sub(12)..hl]
                        .iter().filter(|v| !v.is_nan()).cloned().collect();
                    let avg25 = if avg25_vals.is_empty() { f32::NAN }
                        else { avg25_vals.iter().sum::<f32>() / avg25_vals.len() as f32 };

                    // Build interpretation sentence
                    let text = if !is_fc {
                        // Historical period
                        if !h_val.is_nan() {
                            let vs_peak = (h_val - peak_h) / peak_h * 100.0;
                            let peak_cmp = if vs_peak >= -1.0 {
                                "at the all-time high".to_string()
                            } else {
                                format!("{:.1}% below the all-time peak ({:.2}M)", vs_peak.abs(), peak_h)
                            };
                            format!(
                                "{lbl}  —  Average trending video views: {:.3}M.  This is {}.",
                                h_val, peak_cmp
                            )
                        } else {
                            format!("{lbl}  —  No recorded data for this month.")
                        }
                    } else {
                        // Forecast period
                        let mo_idx = cursor - hl;
                        let yr_str = ["2026","2027","2028"][mo_idx / 12];
                        // vs-2025 helper — empty string when avg25 is unavailable
                        let pct_vs25 = |val: f32| -> String {
                            if avg25.is_nan() || avg25.abs() < 0.001 { return String::new(); }
                            let pct = (val - avg25) / avg25 * 100.0;
                            let dir = if pct >= 0.0 { "up" } else { "down" };
                            format!(" ({} {:.1}% vs 2025)", dir, pct.abs())
                        };
                        let models_agree = !b_val.is_nan() && !v_val.is_nan()
                            && b_val.abs() > 0.001
                            && (b_val - v_val).abs() / b_val < 0.05;

                        if models_agree {
                            format!(
                                "{lbl}  —  Both models predict ~{:.3}M views{}.  \
                                 High agreement between models → higher confidence in this forecast.",
                                (b_val + v_val) / 2.0, pct_vs25((b_val + v_val) / 2.0)
                            )
                        } else if !b_val.is_nan() && !v_val.is_nan() && v_val > b_val {
                            format!(
                                "{lbl}  —  Basic model: {:.3}M{}.  \
                                 Advanced model: {:.3}M{}.  \
                                 Advanced model predicts higher engagement in {yr_str}.  \
                                 Gap of {:.3}M suggests uncertainty — use as a range.",
                                b_val, pct_vs25(b_val),
                                v_val, pct_vs25(v_val),
                                (v_val - b_val).abs()
                            )
                        } else if !b_val.is_nan() && !v_val.is_nan() {
                            format!(
                                "{lbl}  —  Basic model: {:.3}M{}.  \
                                 Advanced model: {:.3}M{}.  \
                                 Basic model predicts higher in {yr_str}.  \
                                 Gap of {:.3}M — treat predictions as a probable range.",
                                b_val, pct_vs25(b_val),
                                v_val, pct_vs25(v_val),
                                (b_val - v_val).abs()
                            )
                        } else {
                            format!("{lbl}  —  Forecast data not available for this month.")
                        }
                    };

                    ui.horizontal(|ui| {
                        // colored left tag
                        let (tag, tag_col) = if is_fc {
                            ("FORECAST", C_SEL)
                        } else {
                            ("ACTUAL", C_ACTUAL)
                        };
                        let (r, painter) = ui.allocate_painter(Vec2::new(64.0, 26.0), egui::Sense::hover());
                        let rr = r.rect;
                        painter.rect_filled(rr, Rounding::same(5.0),
                            Color32::from_rgba_unmultiplied(
                                tag_col.r(), tag_col.g(), tag_col.b(), 35));
                        painter.rect_stroke(rr, Rounding::same(5.0),
                            Stroke::new(1.0, tag_col));
                        painter.text(rr.center(), egui::Align2::CENTER_CENTER,
                            tag, FontId::proportional(10.5), tag_col);
                        ui.add_space(10.0);
                        ui.label(RichText::new(text).size(11.5).color(C_TEXT));
                    });
                }
            });

        // ── Slider ───────────────────────────────────────────
        egui::TopBottomPanel::bottom("slider")
            .exact_height(50.0)
            .frame(egui::Frame::default()
                .fill(C_SURFACE)
                .inner_margin(Margin::symmetric(20.0, 0.0))
                .stroke(Stroke::new(1.0, C_BORDER)))
            .show(ctx, |ui| {
                if let Ok(d) = &self.data {
                    let total = d.labels.len();
                    // slider always covers full timeline
                    let lo = 0usize;
                    let hi = total - 1;

                    ui.horizontal_centered(|ui| {
                        let lo_lbl = "2020".to_string();
                        let hi_lbl = "2028".to_string();
                        ui.label(RichText::new(lo_lbl).size(11.0).color(C_MUTED));
                        ui.add_space(8.0);

                        let s = egui::Slider::new(&mut self.cursor, lo..=hi)
                            .show_value(false)
                            .trailing_fill(true);
                        ui.add_sized(Vec2::new(ui.available_width() - 150.0, 18.0), s);

                        ui.add_space(8.0);
                        ui.label(RichText::new(hi_lbl).size(11.0).color(C_MUTED));
                        ui.add_space(16.0);

                        let c = self.cursor.min(total-1);
                        let hist_len = d.hist_len;
                        let zone = if c < hist_len { "ACTUAL" } else { "FORECAST" };
                        let zcol = if c < hist_len { C_ACTUAL } else { C_SEL };
                        ui.label(RichText::new(&d.labels[c])
                            .size(14.0).strong().color(C_SEL));
                        ui.add_space(8.0);
                        ui.label(RichText::new(zone).size(11.0).color(zcol));
                    });
                }
            });

        // ── Central ──────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(C_BG))
            .show(ctx, |ui| {
                if let Err(e) = &self.data {
                    ui.centered_and_justified(|ui| {
                        ui.label(RichText::new(
                            format!("Data load failed:\n\n{e}\n\nRun:  cargo run --release  inside youtube_predictor/")
                        ).size(14.0).color(C_DOWN));
                    });
                    return;
                }

                let d = self.data.as_ref().unwrap();
                let total = d.labels.len();
                let hl    = d.hist_len;
                let cursor = self.cursor.min(total-1);
                let (x_lo, x_hi) = self.view.x_range(hl);
                let x_lo = x_lo.min(total-1);
                let x_hi = x_hi.min(total-1);

                let h_val  = d.hist[cursor];
                let b_val  = if cursor >= hl { d.fc_bas[cursor] } else { f32::NAN };
                let v_val  = if cursor >= hl { d.fc_adv[cursor] } else { f32::NAN };
                let h_prev = cursor.checked_sub(1).map(|i| d.hist[i]);
                let b_prev = cursor.checked_sub(1).map(|i| if i >= hl { d.fc_bas[i] } else { f32::NAN });
                let v_prev = cursor.checked_sub(1).map(|i| if i >= hl { d.fc_adv[i] } else { f32::NAN });

                let avail_w = ui.available_width();
                let avail_h = ui.available_height();
                let side_w  = 268.0;
                let chart_w = avail_w - side_w - 10.0;
                let legend_h = 36.0;
                let chart_h  = avail_h - legend_h - 8.0;

                ui.horizontal(|ui| {
                    // ── LEFT side: legend + chart ─────────────
                    ui.vertical(|ui| {

                        // Legend row
                        ui.allocate_ui(Vec2::new(chart_w, legend_h), |ui| {
                            ui.horizontal_centered(|ui| {
                                ui.add_space(8.0);
                                legend_item(ui, C_ACTUAL, "Actual Data");
                                legend_item(ui, C_BASIC,  "Basic Prediction  (AI · 63% accuracy)");
                                legend_item(ui, C_ADV,    "Advanced Prediction  (AI · 99% accuracy)");
                                legend_item(ui, C_SEL,    "Selected Month");
                            });
                        });

                        // ── Always build FULL series (all years) ─────
                        let all_vals: Vec<f32> = d.hist.iter()
                            .chain(d.fc_bas.iter()).chain(d.fc_adv.iter())
                            .filter(|v| !v.is_nan()).cloned().collect();
                        let (y_min, y_max) = if all_vals.is_empty() { (0.0_f32, 1.0_f32) } else {
                            (all_vals.iter().cloned().fold(f32::MAX, f32::min),
                             all_vals.iter().cloned().fold(0.0_f32, f32::max))
                        };
                        let y_ceil = (y_max * 1.15) as f64;
                        let y_base = (y_min * 0.88) as f64;

                        let hist_line: PlotPoints = d.hist.iter().enumerate()
                            .filter(|(_,v)| !v.is_nan())
                            .map(|(i,&v)| [i as f64, v as f64]).collect();
                        let bas_line: PlotPoints = d.fc_bas.iter().enumerate()
                            .filter(|(_,v)| !v.is_nan())
                            .map(|(i,&v)| [i as f64, v as f64]).collect();
                        let adv_line: PlotPoints = d.fc_adv.iter().enumerate()
                            .filter(|(_,v)| !v.is_nan())
                            .map(|(i,&v)| [i as f64, v as f64]).collect();

                        // Area fills — full range
                        let make_area = |series: &Vec<f32>| -> Vec<[f64;2]> {
                            let mut pts: Vec<[f64;2]> = series.iter().enumerate()
                                .filter(|(_,v)| !v.is_nan())
                                .map(|(i,&v)| [i as f64, v as f64]).collect();
                            if !pts.is_empty() {
                                let x0 = pts[0][0]; let x1 = pts.last().unwrap()[0];
                                pts.push([x1, y_base]); pts.push([x0, y_base]);
                            }
                            pts
                        };
                        let hist_area = make_area(&d.hist);
                        let bas_area  = make_area(&d.fc_bas);
                        let adv_area  = make_area(&d.fc_adv);

                        // Spotlight: dim overlay polygons for non-selected regions
                        // (only when a specific year is chosen)
                        let spotlight = self.view != View::All;
                        let sp_lo = x_lo as f64;
                        let sp_hi = (x_hi + 1) as f64;
                        let total_f = total as f64;
                        let dim = Color32::from_rgba_unmultiplied(10, 12, 18, 168);

                        // Cursor dots
                        let dot_h: PlotPoints = if !h_val.is_nan() {
                            vec![[cursor as f64, h_val as f64]].into()
                        } else { vec![].into() };
                        let dot_b: PlotPoints = if !b_val.is_nan() {
                            vec![[cursor as f64, b_val as f64]].into()
                        } else { vec![].into() };
                        let dot_v: PlotPoints = if !v_val.is_nan() {
                            vec![[cursor as f64, v_val as f64]].into()
                        } else { vec![].into() };

                        let cx   = cursor as f64;
                        let hl_f = hl as f64;
                        let lbl_c = d.labels.clone();
                        let mut hover_x: Option<f64> = None;

                        let resp = ui.allocate_ui(Vec2::new(chart_w, chart_h), |ui| {
                            Plot::new("chart")
                                .allow_drag(true).allow_zoom(true).allow_scroll(false)
                                .show_grid(true)
                                .set_margin_fraction(Vec2::new(0.02, 0.12))
                                // always show full timeline
                                .include_x(-0.5)
                                .include_x(total_f + 0.5)
                                .x_axis_formatter(move |mark, _| {
                                    let i = mark.value.round() as usize;
                                    if i < lbl_c.len() && i % 12 == 0 {
                                        lbl_c[i][..4].to_string()
                                    } else { String::new() }
                                })
                                .y_axis_formatter(|mark, _| format!("{:.2}M", mark.value))
                                .label_formatter(|name, val| {
                                    if name.is_empty() { String::new() }
                                    else { format!("{name}\n{:.3}M views", val.y) }
                                })
                                .show(ui, |pu| {
                                    hover_x = pu.pointer_coordinate().map(|p: PlotPoint| p.x);

                                    // ── 1. Area fills ─────────────────────────
                                    let poly = |pts, col: [u8;4]| egui_plot::Polygon::new(
                                        PlotPoints::new(pts))
                                        .fill_color(Color32::from_rgba_unmultiplied(col[0],col[1],col[2],col[3]))
                                        .stroke(Stroke::NONE);

                                    if !hist_area.is_empty() {
                                        pu.polygon(poly(hist_area, [148,163,184, 15]));
                                    }
                                    if !bas_area.is_empty() {
                                        pu.polygon(poly(bas_area, [96,165,250, 15]));
                                    }
                                    if !adv_area.is_empty() {
                                        pu.polygon(poly(adv_area, [52,211,153, 15]));
                                    }

                                    // ── 2. Forecast zone tint ─────────────────
                                    pu.polygon(poly(vec![
                                        [hl_f, y_base], [total_f, y_base],
                                        [total_f, y_ceil], [hl_f, y_ceil],
                                    ], [80,120,255, 8]));

                                    // ── 3. Year separators + labels ───────────
                                    let yr_names = ["2020","2021","2022","2023","2024",
                                                    "2025","2026","2027","2028"];
                                    for (yr, &name) in yr_names.iter().enumerate() {
                                        let x = yr as f64 * 12.0;
                                        pu.vline(VLine::new(x)
                                            .color(Color32::from_rgba_unmultiplied(60,70,100, 70))
                                            .width(1.0));
                                        pu.text(egui_plot::Text::new(
                                            PlotPoint::new(x + 0.5, y_ceil * 0.97), name
                                        ).color(Color32::from_rgba_unmultiplied(140,155,185, 150))
                                         .anchor(egui::Align2::LEFT_TOP));
                                    }

                                    // ── 4. Actual/Forecast boundary ───────────
                                    pu.vline(VLine::new(hl_f)
                                        .color(Color32::from_rgba_unmultiplied(251,191,36, 130))
                                        .width(1.5));
                                    pu.text(egui_plot::Text::new(
                                        PlotPoint::new(hl_f - 0.6, y_ceil * 0.82), "ACTUAL"
                                    ).color(Color32::from_rgba_unmultiplied(148,163,184, 140))
                                     .anchor(egui::Align2::RIGHT_TOP));
                                    pu.text(egui_plot::Text::new(
                                        PlotPoint::new(hl_f + 0.6, y_ceil * 0.82), "FORECAST"
                                    ).color(Color32::from_rgba_unmultiplied(96,165,250, 140))
                                     .anchor(egui::Align2::LEFT_TOP));

                                    // ── 5. Data lines ─────────────────────────
                                    pu.line(Line::new(hist_line)
                                        .color(C_ACTUAL).width(2.0).name("Actual Data"));
                                    pu.line(Line::new(bas_line)
                                        .color(C_BASIC).width(2.5).name("Basic Prediction"));
                                    pu.line(Line::new(adv_line)
                                        .color(C_ADV).width(2.5).name("Advanced Prediction"));

                                    // ── 6. Spotlight dim overlay ──────────────
                                    // Drawn AFTER lines so it dims everything outside the focus
                                    if spotlight {
                                        // left dim
                                        if sp_lo > 0.0 {
                                            pu.polygon(egui_plot::Polygon::new(PlotPoints::new(vec![
                                                [-1.0, y_base], [sp_lo, y_base],
                                                [sp_lo, y_ceil], [-1.0, y_ceil],
                                            ])).fill_color(dim).stroke(Stroke::NONE));
                                        }
                                        // right dim
                                        if sp_hi < total_f {
                                            pu.polygon(egui_plot::Polygon::new(PlotPoints::new(vec![
                                                [sp_hi, y_base], [total_f+1.0, y_base],
                                                [total_f+1.0, y_ceil], [sp_hi, y_ceil],
                                            ])).fill_color(dim).stroke(Stroke::NONE));
                                        }
                                        // subtle highlight border on focused year
                                        pu.polygon(egui_plot::Polygon::new(PlotPoints::new(vec![
                                            [sp_lo, y_base], [sp_hi, y_base],
                                            [sp_hi, y_ceil], [sp_lo, y_ceil],
                                        ])).fill_color(Color32::from_rgba_unmultiplied(255,255,255, 4))
                                          .stroke(Stroke::new(1.5,
                                            Color32::from_rgba_unmultiplied(251,191,36, 80))));
                                    }

                                    // ── 7. Cursor ─────────────────────────────
                                    pu.vline(VLine::new(cx)
                                        .color(Color32::from_rgba_unmultiplied(251,191,36, 210))
                                        .width(1.5));
                                    pu.points(Points::new(dot_h).radius(6.0)
                                        .color(C_ACTUAL).filled(true));
                                    pu.points(Points::new(dot_b).radius(6.0)
                                        .color(C_BASIC).filled(true));
                                    pu.points(Points::new(dot_v).radius(6.0)
                                        .color(C_ADV).filled(true));
                                })
                        });

                        if resp.inner.response.hovered() {
                            if let Some(x) = hover_x {
                                self.cursor = (x.round() as i64)
                                    .clamp(0, total as i64 - 1) as usize;
                            }
                        }
                    }); // end chart vertical

                    ui.add_space(10.0);

                    // ── RIGHT side: info panel ────────────────
                    ui.allocate_ui(Vec2::new(side_w, avail_h), |ui| {
                        ui.vertical(|ui| {
                            let card_w = side_w;

                            // ── Date card ────────────────────
                            let is_fc = cursor >= hl;
                            let (zone_lbl, zone_col) =
                                if is_fc { ("Forecast", C_SEL) } else { ("Actual data", C_ACTUAL) };

                            let (date_r, date_painter) = ui.allocate_painter(
                                Vec2::new(card_w, 72.0), egui::Sense::hover());
                            let dr = date_r.rect;
                            date_painter.rect_filled(dr, Rounding::same(10.0), C_CARD);
                            date_painter.rect_stroke(dr, Rounding::same(10.0), Stroke::new(1.0, zone_col));
                            date_painter.text(
                                Pos2::new(dr.min.x + 14.0, dr.min.y + 10.0),
                                egui::Align2::LEFT_TOP,
                                zone_lbl,
                                FontId::proportional(11.0), zone_col,
                            );
                            let lbl = &d.labels[cursor];
                            date_painter.text(
                                Pos2::new(dr.min.x + 14.0, dr.min.y + 26.0),
                                egui::Align2::LEFT_TOP,
                                lbl,
                                FontId::proportional(24.0),
                                C_TEXT,
                            );

                            ui.add_space(8.0);

                            // ── Metric cards ─────────────────
                            let mc_h = 82.0;

                            // Actual
                            metric_card(ui,
                                "Actual Views",
                                &fmt_big(h_val),
                                mom(h_val, h_prev),
                                C_ACTUAL, card_w, mc_h,
                            );
                            ui.add_space(6.0);

                            // Basic prediction
                            metric_card(ui,
                                "Basic Prediction  (AI · 63%)",
                                &fmt_big(b_val),
                                mom(b_val, b_prev),
                                C_BASIC, card_w, mc_h,
                            );
                            ui.add_space(6.0);

                            // Advanced prediction
                            metric_card(ui,
                                "Advanced Prediction  (AI · 99%)",
                                &fmt_big(v_val),
                                mom(v_val, v_prev),
                                C_ADV, card_w, mc_h,
                            );

                            ui.add_space(10.0);

                            // ── Mini stats box ────────────────
                            let stats_h = avail_h - 72.0 - 3.0*mc_h - 10.0*4.0 - 20.0;
                            if stats_h > 100.0 {
                                let (sr, sp) = ui.allocate_painter(
                                    Vec2::new(card_w, stats_h.max(80.0)), egui::Sense::hover());
                                let r = sr.rect;
                                sp.rect_filled(r, Rounding::same(10.0), C_CARD);
                                sp.rect_stroke(r, Rounding::same(10.0), Stroke::new(1.0, C_BORDER));

                                let x = r.min.x + 14.0;
                                let mut y = r.min.y + 12.0;
                                let line = |sp: &Painter, label: &str, val: &str, vc: Color32, yp: f32| {
                                    sp.text(Pos2::new(x, yp), egui::Align2::LEFT_TOP,
                                        label, FontId::proportional(10.5), C_MUTED);
                                    sp.text(Pos2::new(r.max.x - 14.0, yp), egui::Align2::RIGHT_TOP,
                                        val, FontId::proportional(12.0), vc);
                                };

                                // Historical peak
                                let peak = d.hist.iter().filter(|v| !v.is_nan())
                                    .cloned().fold(0.0_f32, f32::max);
                                line(&sp, "Historical peak", &fmt_m(peak), C_ACTUAL, y); y += 18.0;

                                // 2025 avg
                                let sl = d.hist_len.saturating_sub(12);
                                let a2025: Vec<f32> = d.hist[sl..d.hist_len].iter()
                                    .filter(|v| !v.is_nan()).cloned().collect();
                                if !a2025.is_empty() {
                                    let avg = a2025.iter().sum::<f32>() / a2025.len() as f32;
                                    line(&sp, "2025 avg", &fmt_m(avg), C_ACTUAL, y); y += 18.0;
                                }

                                // 2026 avg basic
                                let fc26b: Vec<f32> = d.fc_bas[hl..hl+12].iter()
                                    .filter(|v| !v.is_nan()).cloned().collect();
                                if !fc26b.is_empty() {
                                    let avg = fc26b.iter().sum::<f32>() / fc26b.len() as f32;
                                    line(&sp, "2026 avg (Basic)", &fmt_m(avg), C_BASIC, y); y += 18.0;
                                }

                                // 2026 avg advanced
                                let fc26v: Vec<f32> = d.fc_adv[hl..hl+12].iter()
                                    .filter(|v| !v.is_nan()).cloned().collect();
                                if !fc26v.is_empty() {
                                    let avg = fc26v.iter().sum::<f32>() / fc26v.len() as f32;
                                    line(&sp, "2026 avg (Advanced)", &fmt_m(avg), C_ADV, y);
                                }

                                let _ = y;
                            }

                            // ── Tip ───────────────────────────
                            ui.add_space(8.0);
                            ui.label(RichText::new(
                                "Hover over the chart or drag the slider\nto explore monthly data."
                            ).size(10.5).color(C_MUTED));
                        });
                    });
                }); // end horizontal
            }); // end central
    }
}

// ── Icon generation (pure Rust, no extra deps) ───────────────
fn make_icon() -> egui::IconData {
    const S: usize = 64;
    let mut px = vec![0u8; S * S * 4];

    let cx = S as f32 / 2.0;
    let cy = S as f32 / 2.0;
    let r  = S as f32 / 2.0 - 2.5;

    // Lightning bolt — normalized vertices × S
    let nv: &[(f32, f32)] = &[
        (0.578, 0.078),
        (0.234, 0.531),
        (0.422, 0.531),
        (0.391, 0.938),
        (0.734, 0.453),
        (0.547, 0.453),
    ];
    let bolt: Vec<(f32, f32)> = nv.iter()
        .map(|&(nx, ny)| (nx * S as f32, ny * S as f32))
        .collect();

    for y in 0..S {
        for x in 0..S {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let dx = fx - cx;
            let dy = fy - cy;
            let d  = (dx * dx + dy * dy).sqrt();
            let i  = (y * S + x) * 4;

            if d > r + 1.0 { continue; }

            // anti-aliased circle edge
            let aa = ((r + 1.0 - d) * 255.0).clamp(0.0, 255.0) as u8;

            // dark background fill
            px[i]   = 13;
            px[i+1] = 17;
            px[i+2] = 23;
            px[i+3] = aa;

            // amber border ring (outer 3 px)
            if d > r - 3.0 {
                let t = ((d - (r - 3.0)) / 3.0).clamp(0.0, 1.0);
                icon_blend(&mut px[i..i+4], 251, 191, 36, (t * 210.0) as u8);
            }

            // center glow (radial gradient, very subtle)
            let glow = ((1.0 - d / r) * 28.0) as u8;
            icon_blend(&mut px[i..i+4], 251, 191, 36, glow);

            // lightning bolt fill
            if d <= r && pip(fx, fy, &bolt) {
                px[i]   = 251;
                px[i+1] = 191;
                px[i+2] = 36;
                px[i+3] = aa;
            }
        }
    }

    egui::IconData { rgba: px, width: S as u32, height: S as u32 }
}

fn icon_blend(px: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    let af = a as f32 / 255.0;
    let ia = 1.0 - af;
    px[0] = (px[0] as f32 * ia + r as f32 * af) as u8;
    px[1] = (px[1] as f32 * ia + g as f32 * af) as u8;
    px[2] = (px[2] as f32 * ia + b as f32 * af) as u8;
    if px[3] < a { px[3] = a; }
}

/// Point-in-polygon (ray casting)
fn pip(px: f32, py: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn main() -> eframe::Result<()> {
    let icon = make_icon();
    eframe::run_native(
        "Lightgo — YouTube Views Dashboard",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("Lightgo — YouTube Views Dashboard")
                .with_icon(icon)
                .with_inner_size([1480.0, 900.0])
                .with_min_inner_size([960.0, 620.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
