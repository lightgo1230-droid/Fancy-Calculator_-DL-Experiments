//! Global Inflation vs Interest Rates — Full 3-Year Forecast Dashboard
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{egui, NativeOptions};
use egui::{Color32, RichText, ScrollArea, Ui, Vec2};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, Points, VLine};
use std::collections::HashMap;

// ================================================================
// App icon  (abstract "L" logo)
// ================================================================

fn icon_lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t.clamp(0.0, 1.0) }

fn icon_in_l(col: f32, row: f32, size: f32) -> bool {
    let s = 32.0 / size;
    let (c, r) = (col * s, row * s);
    (c >= 6.0 && c < 14.0 && r >= 3.0 && r < 27.0) || (c >= 6.0 && c < 26.0 && r >= 20.0 && r < 27.0)
}

fn make_icon_rgba(size: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    let sz = size as f32;
    for row in 0..size {
        for col in 0..size {
            let (r, c) = (row as f32, col as f32);
            if icon_in_l(c, r, sz) {
                let t = r / sz;
                px.extend_from_slice(&[
                    icon_lerp(100.0, 38.0,  t) as u8,
                    icon_lerp(180.0, 198.0, t) as u8,
                    icon_lerp(255.0, 218.0, t) as u8,
                    255,
                ]);
            } else {
                let mut closest = f32::MAX;
                'g: for dr in -3i32..=3 { for dc in -3i32..=3 {
                    if icon_in_l(c + dc as f32, r + dr as f32, sz) {
                        let d = ((dr*dr + dc*dc) as f32).sqrt();
                        if d < closest { closest = d; }
                        if closest <= 1.2 { break 'g; }
                    }
                }}
                if      closest <= 1.2 { px.extend_from_slice(&[50, 100, 170, 210]); }
                else if closest <= 2.2 { px.extend_from_slice(&[25, 45,  90,  130]); }
                else                   { px.extend_from_slice(&[13, 17,  23,  255]); }
            }
        }
    }
    px
}

// ================================================================
// Constants
// ================================================================

const EMBEDDED_CSV: &[u8] = include_bytes!("rates_vs_cpi_panel.csv");

const PALETTE: [(u8, u8, u8); 13] = [
    (244, 67,  54),(33,  150, 243),(76,  175, 80),(255, 152, 0),
    (156, 39, 176),(0,  188, 212),(233, 30,  99),(255, 235, 59),
    (121, 85,  72),(96, 125, 139),(139, 195, 74),(255, 87,  34),(63, 81, 181),
];

const FORECAST_MONTHS: usize = 36;

// ================================================================
// Data types
// ================================================================

#[derive(Debug, Clone, PartialEq)]
enum RateAction { Cut, Hold, Hike, Unknown }

impl RateAction {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "cut" => Self::Cut, "hold" => Self::Hold, "hike" => Self::Hike, _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
struct Record {
    year_month: f64, date: String, economy: String,
    policy_rate: Option<f64>, cpi_yoy_pct: Option<f64>, real_rate_pct: Option<f64>,
    #[allow(dead_code)] rate_change_bps: Option<f64>,
    rate_action: RateAction,
    #[allow(dead_code)] cycle_cumul_bps: Option<f64>,
}

#[derive(Debug, Clone)]
struct EcoData {
    name: String,
    dates: Vec<f64>, date_labels: Vec<String>,
    policy_rates: Vec<Option<f64>>, cpi_values: Vec<Option<f64>>,
    real_rates: Vec<Option<f64>>, rate_actions: Vec<RateAction>,
    // Historical MAs
    ma3_cpi: Vec<Option<f64>>, ma6_cpi: Vec<Option<f64>>, ma12_cpi: Vec<Option<f64>>,
    // CPI 3-Year forecasts
    cpi_ma3_fcst:    Vec<[f64; 2]>,
    cpi_ma6_fcst:    Vec<[f64; 2]>,
    cpi_ma12_fcst:   Vec<[f64; 2]>,
    cpi_linear_fcst: Vec<[f64; 2]>,
    // Policy Rate 3-Year forecasts
    rate_ma3_fcst:    Vec<[f64; 2]>,
    rate_ma6_fcst:    Vec<[f64; 2]>,
    rate_ma12_fcst:   Vec<[f64; 2]>,
    rate_linear_fcst: Vec<[f64; 2]>,
    // Real Rate 3-Year forecast (CPI_ma12 - Rate_ma12)
    real_rate_fcst: Vec<[f64; 2]>,
    // Extended series (historical valid + ma12 forecast) for forecast-period heatmap
    dates_ext: Vec<f64>, cpi_ext: Vec<f64>,
    // Stats
    corr_rate_cpi: f64,
    n_cut: usize, n_hold: usize, n_hike: usize,
    cpi_max: f64, cpi_min: f64, rate_max: f64, rate_min: f64,
    last_date: f64,
}

// ================================================================
// Enums
// ================================================================

#[derive(PartialEq, Clone, Copy)]
enum Tab { Prediction, Scatter, Heatmap, Comparison, About }

#[derive(PartialEq, Clone, Copy)]
enum CompareMetric { CpiYoY, PolicyRate, RealRate }

// ================================================================
// Statistics
// ================================================================

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n < 3.0 { return f64::NAN; }
    let mx = a.iter().sum::<f64>() / n;
    let my = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b.iter()).map(|(x,y)| (x-mx)*(y-my)).sum::<f64>() / n;
    let sx = (a.iter().map(|x| (x-mx).powi(2)).sum::<f64>() / n).sqrt();
    let sy = (b.iter().map(|y| (y-my).powi(2)).sum::<f64>() / n).sqrt();
    if sx * sy < 1e-10 { return f64::NAN; }
    (cov / (sx * sy)).clamp(-1.0, 1.0)
}

fn moving_avg(vals: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut result = vec![None; vals.len()];
    for i in 0..vals.len() {
        if i + 1 < window { continue; }
        let slice: Vec<f64> = vals[(i+1-window)..=i].iter().filter_map(|v| *v).collect();
        if slice.len() >= (window + 1) / 2 {
            result[i] = Some(slice.iter().sum::<f64>() / slice.len() as f64);
        }
    }
    result
}

fn linear_reg(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len().min(ys.len()) as f64;
    if n < 2.0 { return (0.0, 0.0); }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys.iter()).map(|(x,y)| (x-mx)*(y-my)).sum();
    let den: f64 = xs.iter().map(|x| (x-mx).powi(2)).sum();
    if den.abs() < 1e-10 { return (0.0, my); }
    let s = num / den;
    (s, my - s * mx)
}

/// Recursive MA forecast — 36 months beyond last data point
fn forecast_ma(dates: &[f64], vals: &[Option<f64>], window: usize, horizon: usize) -> Vec<[f64; 2]> {
    if dates.is_empty() { return vec![]; }
    let valid: Vec<f64> = vals.iter().filter_map(|v| *v).collect();
    if valid.len() < 2 { return vec![]; }
    let seed_n = valid.len().min(window);
    let mut buf: Vec<f64> = valid[valid.len()-seed_n..].to_vec();
    let last_dt = *dates.last().unwrap();
    let mut out = Vec::with_capacity(horizon);
    for i in 1..=horizon {
        let pred = buf.iter().sum::<f64>() / buf.len() as f64;
        out.push([last_dt + i as f64 / 12.0, pred]);
        if buf.len() >= window { buf.remove(0); }
        buf.push(pred);
    }
    out
}

/// OLS trend on last `lookback` valid months, extrapolated `horizon` months
fn forecast_linear(dates: &[f64], vals: &[Option<f64>], lookback: usize, horizon: usize) -> Vec<[f64; 2]> {
    if dates.is_empty() { return vec![]; }
    let pairs: Vec<(f64,f64)> = dates.iter().zip(vals.iter()).rev()
        .filter_map(|(&d,&c)| c.map(|cv|(d,cv))).take(lookback)
        .collect::<Vec<_>>().into_iter().rev().collect();
    if pairs.len() < 4 { return vec![]; }
    let xs: Vec<f64> = pairs.iter().map(|(d,_)| *d).collect();
    let ys: Vec<f64> = pairs.iter().map(|(_,c)| *c).collect();
    let (slope, intercept) = linear_reg(&xs, &ys);
    let last_dt = *dates.last().unwrap();
    (1..=horizon).map(|i| { let x = last_dt + i as f64 / 12.0; [x, slope*x+intercept] }).collect()
}

/// Last valid (date, value) pair — used as bridge point
fn last_valid(dates: &[f64], vals: &[Option<f64>]) -> Option<[f64; 2]> {
    dates.iter().zip(vals.iter()).rev()
        .find(|(_,v)| v.is_some())
        .map(|(&d,&v)| [d, v.unwrap()])
}

/// Pearson r → color  red=-1 yellow=0 green=+1
fn corr_color(r: f64) -> Color32 {
    if r.is_nan() { return Color32::from_gray(28); }
    let t = ((r+1.0)/2.0).clamp(0.0,1.0) as f32;
    if t < 0.5 { let u=t*2.0; Color32::from_rgb((200.0-u*60.0) as u8,(u*180.0) as u8,20) }
    else        { let u=(t-0.5)*2.0; Color32::from_rgb(((1.0-u)*140.0) as u8,180,(u*40.0) as u8) }
}

fn ym_to_str(ym: f64) -> String {
    let yr = ym as i32;
    let mo = ((ym.fract())*12.0).round() as i32 + 1;
    format!("{}-{:02}", yr, mo.clamp(1,12))
}

fn axis_fmt(mark: egui_plot::GridMark, _: &std::ops::RangeInclusive<f64>) -> String {
    ym_to_str(mark.value)
}

// ================================================================
// CSV loader
// ================================================================

fn parse_ym(s: &str) -> f64 {
    let p: Vec<&str> = s.split('-').collect();
    let y = p.first().and_then(|s| s.parse::<f64>().ok()).unwrap_or(2015.0);
    let m = p.get(1).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
    y + (m-1.0)/12.0
}

fn load_csv_bytes(data: &[u8]) -> Result<Vec<Record>, String> {
    let mut rdr = csv::Reader::from_reader(data);
    let mut records = Vec::new();
    for row in rdr.records() {
        let row = row.map_err(|e| e.to_string())?;
        let economy = row.get(1).unwrap_or("").trim().to_string();
        if economy.is_empty() { continue; }
        let date = row.get(0).unwrap_or("").to_string();
        records.push(Record {
            year_month: parse_ym(&date), date, economy,
            policy_rate:     row.get(2).and_then(|s| s.parse().ok()),
            cpi_yoy_pct:     row.get(3).and_then(|s| s.parse().ok()),
            real_rate_pct:   row.get(4).and_then(|s| s.parse().ok()),
            rate_change_bps: row.get(5).and_then(|s| s.parse().ok()),
            rate_action:     row.get(6).map(RateAction::from_str).unwrap_or(RateAction::Unknown),
            cycle_cumul_bps: row.get(7).and_then(|s| s.parse().ok()),
        });
    }
    Ok(records)
}

// ================================================================
// Data processing
// ================================================================

fn build_eco_data(records: &[Record]) -> (Vec<String>, HashMap<String, EcoData>) {
    let mut by_eco: HashMap<String, Vec<&Record>> = HashMap::new();
    for r in records { by_eco.entry(r.economy.clone()).or_default().push(r); }
    let mut economies: Vec<String> = by_eco.keys().cloned().collect();
    economies.sort();

    let mut eco_map = HashMap::new();
    for eco in &economies {
        let mut rows: Vec<&Record> = by_eco[eco].clone();
        rows.sort_by(|a,b| a.year_month.partial_cmp(&b.year_month).unwrap_or(std::cmp::Ordering::Equal));

        let dates:        Vec<f64>         = rows.iter().map(|r| r.year_month).collect();
        let date_labels:  Vec<String>      = rows.iter().map(|r| r.date.chars().take(7).collect()).collect();
        let policy_rates: Vec<Option<f64>> = rows.iter().map(|r| r.policy_rate).collect();
        let cpi_values:   Vec<Option<f64>> = rows.iter().map(|r| r.cpi_yoy_pct).collect();
        let real_rates:   Vec<Option<f64>> = rows.iter().map(|r| r.real_rate_pct).collect();
        let rate_actions: Vec<RateAction>  = rows.iter().map(|r| r.rate_action.clone()).collect();

        // Historical MAs
        let ma3_cpi  = moving_avg(&cpi_values, 3);
        let ma6_cpi  = moving_avg(&cpi_values, 6);
        let ma12_cpi = moving_avg(&cpi_values, 12);

        // CPI 3-Year forecasts
        let cpi_ma3_fcst    = forecast_ma(&dates, &cpi_values, 3,  FORECAST_MONTHS);
        let cpi_ma6_fcst    = forecast_ma(&dates, &cpi_values, 6,  FORECAST_MONTHS);
        let cpi_ma12_fcst   = forecast_ma(&dates, &cpi_values, 12, FORECAST_MONTHS);
        let cpi_linear_fcst = forecast_linear(&dates, &cpi_values, 24, FORECAST_MONTHS);

        // Policy Rate 3-Year forecasts
        let rate_ma3_fcst    = forecast_ma(&dates, &policy_rates, 3,  FORECAST_MONTHS);
        let rate_ma6_fcst    = forecast_ma(&dates, &policy_rates, 6,  FORECAST_MONTHS);
        let rate_ma12_fcst   = forecast_ma(&dates, &policy_rates, 12, FORECAST_MONTHS);
        let rate_linear_fcst = forecast_linear(&dates, &policy_rates, 24, FORECAST_MONTHS);

        // Real Rate 3-Year forecast = CPI_ma12 - Rate_ma12
        let real_rate_fcst: Vec<[f64; 2]> = cpi_ma12_fcst.iter()
            .zip(rate_ma12_fcst.iter())
            .map(|(cp, rp)| [cp[0], cp[1] - rp[1]])
            .collect();

        // Extended series for forecast-period heatmap
        let mut dates_ext: Vec<f64> = dates.iter().zip(cpi_values.iter())
            .filter_map(|(&d, &c)| c.map(|_| d)).collect();
        let mut cpi_ext: Vec<f64> = cpi_values.iter().filter_map(|v| *v).collect();
        for pt in &cpi_ma12_fcst { dates_ext.push(pt[0]); cpi_ext.push(pt[1]); }

        let last_date = *dates.last().unwrap_or(&2024.0);

        let (pr_v, cv_v): (Vec<f64>, Vec<f64>) = policy_rates.iter().zip(cpi_values.iter())
            .filter_map(|(p,c)| match (p,c) { (Some(p),Some(c)) => Some((*p,*c)), _ => None }).unzip();
        let corr_rate_cpi = pearson(&pr_v, &cv_v);

        let n_cut  = rate_actions.iter().filter(|a| **a == RateAction::Cut).count();
        let n_hold = rate_actions.iter().filter(|a| **a == RateAction::Hold).count();
        let n_hike = rate_actions.iter().filter(|a| **a == RateAction::Hike).count();

        let cpi_vals:  Vec<f64> = cpi_values.iter().filter_map(|v| *v).collect();
        let rate_vals: Vec<f64> = policy_rates.iter().filter_map(|v| *v).collect();
        let cpi_max  = cpi_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let cpi_min  = cpi_vals.iter().cloned().fold(f64::INFINITY,     f64::min);
        let rate_max = rate_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rate_min = rate_vals.iter().cloned().fold(f64::INFINITY,     f64::min);

        eco_map.insert(eco.clone(), EcoData {
            name: eco.clone(), dates, date_labels,
            policy_rates, cpi_values, real_rates, rate_actions,
            ma3_cpi, ma6_cpi, ma12_cpi,
            cpi_ma3_fcst, cpi_ma6_fcst, cpi_ma12_fcst, cpi_linear_fcst,
            rate_ma3_fcst, rate_ma6_fcst, rate_ma12_fcst, rate_linear_fcst,
            real_rate_fcst, dates_ext, cpi_ext,
            corr_rate_cpi, n_cut, n_hold, n_hike,
            cpi_max, cpi_min, rate_max, rate_min, last_date,
        });
    }
    (economies, eco_map)
}

/// Build heatmap from the given (dates, cpi) data per economy
fn build_heatmap_from(
    economies: &[String],
    get_series: impl Fn(&str) -> (Vec<f64>, Vec<f64>), // (dates, cpi_vals)
) -> Vec<Vec<f64>> {
    let n = economies.len();
    let mut mat = vec![vec![f64::NAN; n]; n];
    let series: Vec<(Vec<f64>, Vec<f64>)> = economies.iter().map(|e| get_series(e)).collect();
    for i in 0..n {
        for j in 0..n {
            if i == j { mat[i][j] = 1.0; continue; }
            let date_map: HashMap<u64, f64> = series[i].0.iter()
                .zip(series[i].1.iter())
                .map(|(&d, &c)| ((d*10000.0) as u64, c))
                .collect();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (&d, &c) in series[j].0.iter().zip(series[j].1.iter()) {
                if let Some(&ai) = date_map.get(&((d*10000.0) as u64)) { a.push(ai); b.push(c); }
            }
            mat[i][j] = pearson(&a, &b);
        }
    }
    mat
}

// ================================================================
// App struct
// ================================================================

struct InflationApp {
    records: Vec<Record>,
    economies: Vec<String>,
    eco_map: HashMap<String, EcoData>,
    selected_eco: usize,
    tab: Tab,
    // Heatmap — historical
    hmap_economies: Vec<String>,
    hmap_matrix: Vec<Vec<f64>>,
    // Heatmap — with 3Y forecast appended
    hmap_matrix_fcst: Vec<Vec<f64>>,
    compare_metric: CompareMetric,
    show_eco: Vec<bool>,
    status_msg: String,
    // Display toggles
    show_ma3: bool, show_ma6: bool, show_ma12: bool,
    show_rate_on_pred: bool,
    show_fcst: bool,
    show_linear_fcst: bool,
    hmap_show_fcst: bool,
}

impl InflationApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill     = Color32::from_rgb(13, 17, 23);
        visuals.window_fill    = Color32::from_rgb(22, 27, 34);
        visuals.faint_bg_color = Color32::from_rgb(22, 27, 34);
        cc.egui_ctx.set_visuals(visuals);

        let (records, economies, eco_map, hmap_economies, hmap_matrix, hmap_matrix_fcst, status_msg, n_eco) =
            match load_csv_bytes(EMBEDDED_CSV) {
                Ok(recs) => {
                    let (ecos, edata) = build_eco_data(&recs);
                    let hmap_ecos = ecos.clone();
                    // Historical heatmap
                    let hmap_hist = build_heatmap_from(&hmap_ecos, |eco| {
                        let e = &edata[eco];
                        let dates: Vec<f64> = e.dates.iter().zip(e.cpi_values.iter())
                            .filter_map(|(&d,&c)| c.map(|_| d)).collect();
                        let cpis: Vec<f64>  = e.cpi_values.iter().filter_map(|v| *v).collect();
                        (dates, cpis)
                    });
                    // Forecast-extended heatmap
                    let hmap_fcst = build_heatmap_from(&hmap_ecos, |eco| {
                        let e = &edata[eco];
                        (e.dates_ext.clone(), e.cpi_ext.clone())
                    });
                    let n = ecos.len();
                    let msg = format!(
                        "Loaded {} economies  {}  records  |  3-year forecast ready ({}~{})",
                        n, recs.len(),
                        edata.values().next().map(|e| ym_to_str(e.last_date)).unwrap_or_default(),
                        edata.values().next().map(|e| ym_to_str(e.last_date + FORECAST_MONTHS as f64/12.0)).unwrap_or_default(),
                    );
                    (recs, ecos, edata, hmap_ecos, hmap_hist, hmap_fcst, msg, n)
                }
                Err(e) => (vec![], vec![], HashMap::new(), vec![], vec![], vec![],
                           format!("Load failed: {}", e), 0),
            };

        Self {
            records, economies, eco_map, selected_eco: 0,
            tab: Tab::Prediction,
            hmap_economies, hmap_matrix, hmap_matrix_fcst,
            compare_metric: CompareMetric::CpiYoY,
            show_eco: vec![true; n_eco],
            status_msg,
            show_ma3: true, show_ma6: true, show_ma12: false,
            show_rate_on_pred: true,
            show_fcst: true, show_linear_fcst: true,
            hmap_show_fcst: false,
        }
    }

    fn current_eco(&self) -> Option<&EcoData> {
        self.economies.get(self.selected_eco).and_then(|n| self.eco_map.get(n))
    }

    // ----------------------------------------------------------
    // Tab 1: CPI Forecast + 3-Year Projection
    // ----------------------------------------------------------
    fn show_prediction(&self, ui: &mut Ui) {
        let eco = match self.current_eco() {
            Some(e) => e,
            None => { ui.label("No data"); return; }
        };

        ui.horizontal(|ui| {
            ui.heading(RichText::new(format!("{}  —  CPI YoY%  Forecast  +  3-Year Projection", eco.name))
                .color(Color32::from_rgb(100, 180, 255)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let corr = if eco.corr_rate_cpi.is_nan() { "N/A".into() }
                           else { format!("{:+.3}", eco.corr_rate_cpi) };
                ui.label(RichText::new(format!("Rate vs CPI r = {}  |  Hike {}x  Hold {}x  Cut {}x",
                    corr, eco.n_hike, eco.n_hold, eco.n_cut)).weak());
            });
        });
        if self.show_fcst {
            ui.label(RichText::new(format!(
                "  Forecast window: {} — {}   (MA-3/6/12 recursive  +  OLS linear trend on last 24 months)",
                ym_to_str(eco.last_date), ym_to_str(eco.last_date + FORECAST_MONTHS as f64/12.0)
            )).size(10.5).color(Color32::from_rgb(150, 210, 255)));
        }
        ui.separator();

        let plot_h = (ui.available_height() - 4.0).max(300.0);
        Plot::new("pred")
            .height(plot_h).legend(egui_plot::Legend::default())
            .x_axis_formatter(axis_fmt).y_axis_label("(%)")
            .show(ui, |pui| {
                if self.show_fcst {
                    pui.vline(VLine::new(eco.last_date)
                        .color(Color32::from_rgba_unmultiplied(200,200,200,70))
                        .width(1.2).style(egui_plot::LineStyle::Dashed { length: 6.0 })
                        .name("Data End"));
                }
                // Actual CPI
                let pts: Vec<[f64;2]> = eco.dates.iter().zip(eco.cpi_values.iter())
                    .filter_map(|(&d,&c)| c.map(|v| [d,v])).collect();
                if !pts.is_empty() {
                    pui.line(Line::new(PlotPoints::new(pts))
                        .color(Color32::from_rgb(244,67,54)).name("Actual CPI YoY%").width(2.2));
                }
                // MA-3 history + forecast
                if self.show_ma3 {
                    let hist: Vec<[f64;2]> = eco.dates.iter().zip(eco.ma3_cpi.iter())
                        .filter_map(|(&d,&c)| c.map(|v| [d,v])).collect();
                    if !hist.is_empty() {
                        pui.line(Line::new(PlotPoints::new(hist))
                            .color(Color32::from_rgb(33,150,243)).name("MA-3").width(1.6)
                            .style(egui_plot::LineStyle::Dashed { length: 10.0 }));
                    }
                    if self.show_fcst && !eco.cpi_ma3_fcst.is_empty() {
                        let mut f = eco.cpi_ma3_fcst.clone();
                        if let Some(b) = last_valid(&eco.dates, &eco.ma3_cpi) { f.insert(0,b); }
                        pui.line(Line::new(PlotPoints::new(f))
                            .color(Color32::from_rgba_unmultiplied(33,150,243,160))
                            .name("MA-3  +3Y").width(1.3)
                            .style(egui_plot::LineStyle::Dashed { length: 5.0 }));
                    }
                }
                // MA-6
                if self.show_ma6 {
                    let hist: Vec<[f64;2]> = eco.dates.iter().zip(eco.ma6_cpi.iter())
                        .filter_map(|(&d,&c)| c.map(|v| [d,v])).collect();
                    if !hist.is_empty() {
                        pui.line(Line::new(PlotPoints::new(hist))
                            .color(Color32::from_rgb(255,152,0)).name("MA-6").width(1.6)
                            .style(egui_plot::LineStyle::Dashed { length: 14.0 }));
                    }
                    if self.show_fcst && !eco.cpi_ma6_fcst.is_empty() {
                        let mut f = eco.cpi_ma6_fcst.clone();
                        if let Some(b) = last_valid(&eco.dates, &eco.ma6_cpi) { f.insert(0,b); }
                        pui.line(Line::new(PlotPoints::new(f))
                            .color(Color32::from_rgba_unmultiplied(255,152,0,160))
                            .name("MA-6  +3Y").width(1.3)
                            .style(egui_plot::LineStyle::Dashed { length: 5.0 }));
                    }
                }
                // MA-12
                if self.show_ma12 {
                    let hist: Vec<[f64;2]> = eco.dates.iter().zip(eco.ma12_cpi.iter())
                        .filter_map(|(&d,&c)| c.map(|v| [d,v])).collect();
                    if !hist.is_empty() {
                        pui.line(Line::new(PlotPoints::new(hist))
                            .color(Color32::from_rgb(156,39,176)).name("MA-12").width(1.6)
                            .style(egui_plot::LineStyle::Dashed { length: 20.0 }));
                    }
                    if self.show_fcst && !eco.cpi_ma12_fcst.is_empty() {
                        let mut f = eco.cpi_ma12_fcst.clone();
                        if let Some(b) = last_valid(&eco.dates, &eco.ma12_cpi) { f.insert(0,b); }
                        pui.line(Line::new(PlotPoints::new(f))
                            .color(Color32::from_rgba_unmultiplied(156,39,176,160))
                            .name("MA-12  +3Y").width(1.3)
                            .style(egui_plot::LineStyle::Dashed { length: 5.0 }));
                    }
                }
                // Linear OLS forecast
                if self.show_fcst && self.show_linear_fcst && !eco.cpi_linear_fcst.is_empty() {
                    let mut f = eco.cpi_linear_fcst.clone();
                    if let Some(b) = last_valid(&eco.dates, &eco.cpi_values) { f.insert(0,b); }
                    pui.line(Line::new(PlotPoints::new(f))
                        .color(Color32::from_rgba_unmultiplied(255,230,80,200))
                        .name("OLS Linear  +3Y").width(1.5)
                        .style(egui_plot::LineStyle::Dashed { length: 8.0 }));
                }
                // Policy rate
                if self.show_rate_on_pred {
                    let pts: Vec<[f64;2]> = eco.dates.iter().zip(eco.policy_rates.iter())
                        .filter_map(|(&d,&r)| r.map(|v| [d,v])).collect();
                    if !pts.is_empty() {
                        pui.line(Line::new(PlotPoints::new(pts))
                            .color(Color32::from_rgb(76,175,80)).name("Policy Rate").width(1.6));
                    }
                    // Rate forecast
                    if self.show_fcst && !eco.rate_ma12_fcst.is_empty() {
                        let mut f = eco.rate_ma12_fcst.clone();
                        if let Some(b) = last_valid(&eco.dates, &eco.policy_rates) { f.insert(0,b); }
                        pui.line(Line::new(PlotPoints::new(f))
                            .color(Color32::from_rgba_unmultiplied(76,175,80,150))
                            .name("Policy Rate  +3Y").width(1.2)
                            .style(egui_plot::LineStyle::Dashed { length: 5.0 }));
                    }
                }
                // Rate markers
                let (mut cut_pts, mut hike_pts) = (Vec::new(), Vec::new());
                for ((&d,&c), act) in eco.dates.iter().zip(eco.cpi_values.iter()).zip(eco.rate_actions.iter()) {
                    if let Some(cv) = c {
                        match act {
                            RateAction::Cut  => cut_pts.push([d,cv]),
                            RateAction::Hike => hike_pts.push([d,cv]),
                            _ => {}
                        }
                    }
                }
                if !hike_pts.is_empty() {
                    pui.points(Points::new(PlotPoints::new(hike_pts))
                        .color(Color32::from_rgb(239,83,80)).name("Rate Hike").radius(4.5));
                }
                if !cut_pts.is_empty() {
                    pui.points(Points::new(PlotPoints::new(cut_pts))
                        .color(Color32::from_rgb(38,166,154)).name("Rate Cut").radius(4.5));
                }
            });
    }

    // ----------------------------------------------------------
    // Tab 2: Rate vs CPI Scatter + 3Y Trajectory + Bar Chart
    // ----------------------------------------------------------
    fn show_scatter(&self, ui: &mut Ui) {
        let eco = match self.current_eco() {
            Some(e) => e,
            None => { ui.label("No data"); return; }
        };

        let corr_str = if eco.corr_rate_cpi.is_nan() { "N/A".into() }
                       else {
                           let dir = if eco.corr_rate_cpi > 0.3 { "positive" }
                                     else if eco.corr_rate_cpi < -0.3 { "negative" }
                                     else { "weak" };
                           format!("{:+.3}  ({})", eco.corr_rate_cpi, dir)
                       };

        ui.horizontal(|ui| {
            ui.heading(RichText::new(format!("{}  —  Policy Rate vs CPI", eco.name))
                .color(Color32::from_rgb(100, 180, 255)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("Pearson r = {}", corr_str)).weak());
            });
        });
        if self.show_fcst {
            ui.label(RichText::new(format!(
                "  Magenta trail = MA-12 forecast trajectory in (Rate, CPI) space  {}~{}",
                ym_to_str(eco.last_date), ym_to_str(eco.last_date + FORECAST_MONTHS as f64/12.0)
            )).size(10.5).color(Color32::from_rgb(220, 150, 255)));
        }
        ui.separator();

        let avail     = ui.available_height();
        let scatter_h = (avail * 0.72).max(200.0);
        let bar_h     = (avail * 0.22).max(80.0);

        let (mut cut_pts, mut hold_pts, mut hike_pts) = (vec![], vec![], vec![]);
        for ((pr, cv), act) in eco.policy_rates.iter().zip(eco.cpi_values.iter()).zip(eco.rate_actions.iter()) {
            if let (Some(r), Some(c)) = (pr, cv) {
                match act {
                    RateAction::Cut  => cut_pts.push([*r, *c]),
                    RateAction::Hold => hold_pts.push([*r, *c]),
                    RateAction::Hike => hike_pts.push([*r, *c]),
                    _ => {}
                }
            }
        }

        let all_pairs: Vec<(f64,f64)> = eco.policy_rates.iter().zip(eco.cpi_values.iter())
            .filter_map(|(p,c)| match (p,c) { (Some(p),Some(c)) => Some((*p,*c)), _ => None }).collect();

        let trend_line = if all_pairs.len() > 2 {
            let xs: Vec<f64> = all_pairs.iter().map(|(x,_)| *x).collect();
            let ys: Vec<f64> = all_pairs.iter().map(|(_,y)| *y).collect();
            let (s,b) = linear_reg(&xs, &ys);
            let (xmin, xmax) = (eco.rate_min - 0.2, eco.rate_max + 0.2);
            Some(vec![[xmin, s*xmin+b], [xmax, s*xmax+b]])
        } else { None };

        // Build forecast trajectory (rate_ma12 vs cpi_ma12 — each future month as a point)
        let fcst_trail: Vec<[f64; 2]> = {
            let mut trail = Vec::new();
            // Start from last actual observation
            if let (Some(lr), Some(lc)) = (
                eco.policy_rates.iter().rev().find(|v| v.is_some()).and_then(|v| *v),
                eco.cpi_values.iter().rev().find(|v| v.is_some()).and_then(|v| *v),
            ) { trail.push([lr, lc]); }
            for (rp, cp) in eco.rate_ma12_fcst.iter().zip(eco.cpi_ma12_fcst.iter()) {
                trail.push([rp[1], cp[1]]);
            }
            trail
        };

        Plot::new("scatter")
            .height(scatter_h).legend(egui_plot::Legend::default())
            .x_axis_label("Policy Rate (%)").y_axis_label("CPI YoY (%)")
            .show(ui, |pui| {
                if !hike_pts.is_empty() {
                    pui.points(Points::new(PlotPoints::new(hike_pts))
                        .color(Color32::from_rgb(239,83,80)).name("Hike month").radius(5.0));
                }
                if !hold_pts.is_empty() {
                    pui.points(Points::new(PlotPoints::new(hold_pts))
                        .color(Color32::from_rgb(255,152,0)).name("Hold month").radius(3.5));
                }
                if !cut_pts.is_empty() {
                    pui.points(Points::new(PlotPoints::new(cut_pts))
                        .color(Color32::from_rgb(38,166,154)).name("Cut month").radius(5.0));
                }
                if let Some(pts) = trend_line {
                    pui.line(Line::new(PlotPoints::new(pts))
                        .color(Color32::from_rgb(200,200,200)).name("OLS Trend").width(1.5)
                        .style(egui_plot::LineStyle::Dashed { length: 8.0 }));
                }
                // 3Y Forecast trajectory
                if self.show_fcst && fcst_trail.len() > 1 {
                    pui.line(Line::new(PlotPoints::new(fcst_trail.clone()))
                        .color(Color32::from_rgba_unmultiplied(220,100,255,220))
                        .name("MA-12 Trajectory +3Y").width(2.0)
                        .style(egui_plot::LineStyle::Dashed { length: 7.0 }));
                    // Mark the end point
                    if let Some(&end) = fcst_trail.last() {
                        pui.points(Points::new(PlotPoints::new(vec![end]))
                            .color(Color32::from_rgb(220,100,255))
                            .name("Forecast End").radius(6.0));
                    }
                }
            });

        ui.add_space(6.0);
        ui.label(RichText::new("Rate Decision Counts").weak().size(11.0));
        let bars = vec![
            Bar::new(0.0, eco.n_hike as f64).name("Hike").fill(Color32::from_rgb(239,83,80)),
            Bar::new(1.0, eco.n_hold as f64).name("Hold").fill(Color32::from_rgb(255,152,0)),
            Bar::new(2.0, eco.n_cut  as f64).name("Cut") .fill(Color32::from_rgb(38,166,154)),
        ];
        Plot::new("action_bar")
            .height(bar_h).legend(egui_plot::Legend::default())
            .x_axis_formatter(|m: egui_plot::GridMark, _: &std::ops::RangeInclusive<f64>| {
                match m.value as i32 { 0=>"Hike".into(), 1=>"Hold".into(), 2=>"Cut".into(), _=>String::new() }
            })
            .show(ui, |pui| { pui.bar_chart(BarChart::new(bars).width(0.6)); });
    }

    // ----------------------------------------------------------
    // Tab 3: Correlation Heatmap  (toggle: historical / +3Y forecast)
    // ----------------------------------------------------------
    fn show_heatmap(&self, ui: &mut Ui) {
        let matrix = if self.hmap_show_fcst { &self.hmap_matrix_fcst } else { &self.hmap_matrix };
        let mode_label = if self.hmap_show_fcst { "Historical + 3Y Forecast" } else { "Historical Only" };

        ui.horizontal(|ui| {
            ui.heading(RichText::new(format!("CPI Co-movement Heatmap  [{mode_label}]"))
                .color(Color32::from_rgb(100, 180, 255)));
        });
        ui.label(RichText::new(
            "Each cell = Pearson r of CPI YoY% time series between two economies  \
             |  Green=co-move  |  Red=inverse  |  Toggle mode in sidebar"
        ).weak().size(11.0));
        ui.separator();

        let n = self.hmap_economies.len();
        if n == 0 { ui.label("No data"); return; }

        let cell: f32 = 60.0; let label_w: f32 = 130.0; let label_h: f32 = 70.0;
        let total_w = label_w + n as f32 * cell + 30.0;
        let total_h = label_h + n as f32 * cell + 30.0;

        ScrollArea::both().show(ui, |ui| {
            let (resp, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), egui::Sense::hover());
            let ox = resp.rect.min.x; let oy = resp.rect.min.y;

            for (j, name) in self.hmap_economies.iter().enumerate() {
                let cx = ox + label_w + j as f32 * cell + cell/2.0;
                let short: String = name.chars().take(10).collect();
                painter.text(egui::pos2(cx, oy+label_h-4.0), egui::Align2::CENTER_BOTTOM,
                    short, egui::FontId::proportional(9.5), Color32::from_gray(200));
            }
            for (i, name_i) in self.hmap_economies.iter().enumerate() {
                let ry = oy + label_h + i as f32 * cell + cell/2.0;
                painter.text(egui::pos2(ox+label_w-6.0, ry), egui::Align2::RIGHT_CENTER,
                    name_i.as_str(), egui::FontId::proportional(9.5), Color32::from_gray(200));
                for j in 0..n {
                    let r = matrix[i][j];
                    let color = corr_color(r);
                    let cx = ox + label_w + j as f32 * cell;
                    let cy = oy + label_h + i as f32 * cell;
                    let rect = egui::Rect::from_min_size(egui::pos2(cx+1.5, cy+1.5), Vec2::splat(cell-3.0));
                    painter.rect_filled(rect, 4.0, color);
                    let text    = if r.is_nan() { "N/A".into() } else { format!("{:.2}", r) };
                    let txt_col = if i==j { Color32::BLACK }
                                  else if r.abs() > 0.5 { Color32::WHITE }
                                  else { Color32::from_gray(220) };
                    painter.text(rect.center(), egui::Align2::CENTER_CENTER,
                        text, egui::FontId::proportional(9.5), txt_col);
                }
            }
            let ly = oy + total_h - 18.0;
            let lx = ox + label_w;
            let lw = n as f32 * cell;
            for k in 0..60_usize {
                let t = k as f32 / 59.0;
                let col = corr_color((t*2.0-1.0) as f64);
                painter.rect_filled(egui::Rect::from_min_size(
                    egui::pos2(lx+t*lw, ly), Vec2::new(lw/60.0+0.5, 10.0)), 0.0, col);
            }
            for (anch, lbl, pos) in [
                (egui::Align2::LEFT_TOP,   "-1.0", egui::pos2(lx,         ly+12.0)),
                (egui::Align2::CENTER_TOP, " 0  ", egui::pos2(lx+lw/2.0, ly+12.0)),
                (egui::Align2::RIGHT_TOP,  "+1.0", egui::pos2(lx+lw,     ly+12.0)),
            ] {
                painter.text(pos, anch, lbl, egui::FontId::proportional(9.0), Color32::GRAY);
            }
        });
    }

    // ----------------------------------------------------------
    // Tab 4: Economy Comparison  —  all metrics  +  3Y forecast
    // ----------------------------------------------------------
    fn show_comparison(&mut self, ui: &mut Ui) {
        ui.heading(RichText::new("Economy Comparison  —  All Metrics  +  3-Year Forecast")
            .color(Color32::from_rgb(100, 180, 255)));
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(RichText::new("Metric:").strong());
            ui.selectable_value(&mut self.compare_metric, CompareMetric::CpiYoY,    "CPI YoY%");
            ui.selectable_value(&mut self.compare_metric, CompareMetric::PolicyRate, "Policy Rate");
            ui.selectable_value(&mut self.compare_metric, CompareMetric::RealRate,   "Real Rate");
            ui.separator();
            ui.label(RichText::new("Show:").strong());
            if ui.small_button("All").clicked()  { self.show_eco.iter_mut().for_each(|v| *v=true); }
            if ui.small_button("None").clicked() { self.show_eco.iter_mut().for_each(|v| *v=false); }
        });

        if self.show_fcst {
            let fcst_label = match self.compare_metric {
                CompareMetric::CpiYoY    => "CPI MA-12 forecast",
                CompareMetric::PolicyRate => "Rate MA-12 forecast",
                CompareMetric::RealRate   => "Real Rate (CPI_ma12 - Rate_ma12) forecast",
            };
            ui.label(RichText::new(format!("  Dashed extension = {}  (lighter color)", fcst_label))
                .size(10.5).color(Color32::from_rgb(150, 210, 255)));
        }

        ui.horizontal_wrapped(|ui| {
            for (i, eco) in self.economies.iter().enumerate() {
                if i < self.show_eco.len() {
                    let c = PALETTE[i % PALETTE.len()];
                    ui.colored_label(Color32::from_rgb(c.0,c.1,c.2), "●");
                    ui.checkbox(&mut self.show_eco[i], eco.as_str());
                }
            }
        });
        ui.separator();

        let metric    = self.compare_metric;
        let show_fcst = self.show_fcst;

        let hist_data: Vec<(String, Vec<[f64;2]>, usize)> = self.economies.iter()
            .enumerate()
            .filter(|(i,_)| self.show_eco.get(*i).copied().unwrap_or(false))
            .filter_map(|(i, name)| {
                self.eco_map.get(name).map(|eco| {
                    let vals: &Vec<Option<f64>> = match metric {
                        CompareMetric::CpiYoY    => &eco.cpi_values,
                        CompareMetric::PolicyRate => &eco.policy_rates,
                        CompareMetric::RealRate   => &eco.real_rates,
                    };
                    let pts: Vec<[f64;2]> = eco.dates.iter()
                        .zip(vals.iter()).filter_map(|(&d,&v)| v.map(|vv| [d,vv])).collect();
                    (name.clone(), pts, i)
                })
            }).collect();

        let fcst_data: Vec<(String, Vec<[f64;2]>, usize)> = if show_fcst {
            self.economies.iter().enumerate()
                .filter(|(i,_)| self.show_eco.get(*i).copied().unwrap_or(false))
                .filter_map(|(i, name)| {
                    self.eco_map.get(name).map(|eco| {
                        // Choose correct forecast and bridge based on metric
                        let (fcst, bridge_src) = match metric {
                            CompareMetric::CpiYoY    => (eco.cpi_ma12_fcst.clone(),
                                last_valid(&eco.dates, &eco.cpi_values)),
                            CompareMetric::PolicyRate => (eco.rate_ma12_fcst.clone(),
                                last_valid(&eco.dates, &eco.policy_rates)),
                            CompareMetric::RealRate   => (eco.real_rate_fcst.clone(),
                                last_valid(&eco.dates, &eco.real_rates)),
                        };
                        let mut pts = fcst;
                        if let Some(b) = bridge_src { pts.insert(0, b); }
                        (name.clone(), pts, i)
                    })
                }).collect()
        } else { vec![] };

        let y_label = match metric {
            CompareMetric::CpiYoY    => "CPI YoY (%)",
            CompareMetric::PolicyRate => "Policy Rate (%)",
            CompareMetric::RealRate   => "Real Rate (%)",
        };

        let plot_h = (ui.available_height() - 4.0).max(300.0);
        Plot::new("comparison")
            .height(plot_h).legend(egui_plot::Legend::default())
            .x_axis_formatter(axis_fmt).y_axis_label(y_label)
            .show(ui, |pui| {
                // Vertical separator
                if show_fcst {
                    if let Some(last_dt) = self.eco_map.values()
                        .filter_map(|e| if e.last_date > 0.0 { Some(e.last_date) } else { None })
                        .reduce(f64::max)
                    {
                        pui.vline(VLine::new(last_dt)
                            .color(Color32::from_rgba_unmultiplied(200,200,200,60))
                            .width(1.0).style(egui_plot::LineStyle::Dashed { length: 6.0 })
                            .name("Data End"));
                    }
                }
                for (name, pts, idx) in &hist_data {
                    if pts.is_empty() { continue; }
                    let c = PALETTE[idx % PALETTE.len()];
                    pui.line(Line::new(PlotPoints::new(pts.clone()))
                        .color(Color32::from_rgb(c.0,c.1,c.2))
                        .name(name.as_str()).width(1.8));
                }
                for (name, pts, idx) in &fcst_data {
                    if pts.is_empty() { continue; }
                    let c = PALETTE[idx % PALETTE.len()];
                    pui.line(Line::new(PlotPoints::new(pts.clone()))
                        .color(Color32::from_rgba_unmultiplied(c.0, c.1, c.2, 140))
                        .name(format!("{} +3Y", name)).width(1.3)
                        .style(egui_plot::LineStyle::Dashed { length: 5.0 }));
                }
            });
    }

    // ----------------------------------------------------------
    // Tab 5: About
    // ----------------------------------------------------------
    fn show_about(&self, ui: &mut Ui) {
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(6.0);
            ui.heading(RichText::new("Global Inflation vs Interest Rates  —  Full Forecast Dashboard")
                .color(Color32::from_rgb(100, 200, 255)).size(20.0));
            ui.add_space(8.0);

            egui::Frame::none().fill(Color32::from_rgb(22,27,34))
                .inner_margin(egui::Margin::same(12.0)).rounding(8.0)
                .show(ui, |ui| {
                    egui::Grid::new("meta").num_columns(2).spacing([24.0,6.0]).show(ui, |ui| {
                        ui.label(RichText::new("Data file").strong());
                        ui.label("rates_vs_cpi_panel.csv");
                        ui.end_row();
                        ui.label(RichText::new("Economies").strong());
                        ui.label(self.economies.join("  /  "));
                        ui.end_row();
                        if let Some(eco) = self.eco_map.values().next() {
                            ui.label(RichText::new("Historical period").strong());
                            ui.label(format!("{} ~ {}", eco.date_labels.first().cloned().unwrap_or_default(),
                                eco.date_labels.last().cloned().unwrap_or_default()));
                            ui.end_row();
                            ui.label(RichText::new("Forecast window").strong());
                            ui.label(format!("{} ~ {}  (36 months, MA-3/6/12 + OLS linear)",
                                ym_to_str(eco.last_date),
                                ym_to_str(eco.last_date + FORECAST_MONTHS as f64/12.0)));
                            ui.end_row();
                        }
                        ui.label(RichText::new("Total records").strong());
                        ui.label(format!("{}", self.records.len()));
                        ui.end_row();
                    });
                });

            ui.add_space(14.0);
            ui.label(RichText::new("Tab-by-Tab Forecast Features").strong().size(15.0));
            ui.separator();

            let descs: &[(&str, &str, &[&str])] = &[
                ("CPI Forecast  +  3Y Projection",
                 "Actual CPI YoY% vs moving-average fits, all extended 36 months as forecasts.",
                 &[
                    "  Red solid         Actual CPI YoY%",
                    "  Blue dashed       MA-3 fit  →  lighter dashed: MA-3 +3Y recursive forecast",
                    "  Orange dashed     MA-6 fit  →  lighter dashed: MA-6 +3Y recursive forecast",
                    "  Purple dashed     MA-12 fit →  lighter dashed: MA-12 +3Y recursive forecast",
                    "  Yellow dashed     OLS linear trend on last 24mo, extrapolated 3Y",
                    "  Green solid       Policy Rate  →  lighter: Rate MA-12 +3Y forecast",
                    "  Grey VLine        Boundary between historical data and forecast zone",
                 ]),
                ("Rate vs CPI Scatter  +  3Y Trajectory",
                 "Historical scatter coloured by rate decision. Magenta trail shows the MA-12 forecast path in (Rate, CPI) phase space.",
                 &[
                    "  Red/Orange/Teal scatter  Historical hike/hold/cut months",
                    "  White dashed             OLS linear trend across all historical points",
                    "  Magenta dashed trail      36-month projected path in (Rate, CPI) space",
                    "  Magenta filled dot        Forecast endpoint (3 years ahead)",
                    "  Trajectory shape: flat = stable state;  curve = converging/diverging rate-inflation dynamic",
                 ]),
                ("Heatmap  —  Historical vs Historical+Forecast",
                 "Toggle in sidebar between historical-only Pearson r matrix and the matrix computed over historical + 36-month MA-12 forecast period.",
                 &[
                    "  Green (+1.0) = perfect co-movement",
                    "  Yellow ( 0.0) = no correlation",
                    "  Red   (-1.0) = perfect inverse",
                    "  Historical Only:  correlations based on observed CPI data",
                    "  With +3Y Forecast:  each economy's CPI extended by MA-12 forecast before computing r",
                    "  If correlations shift when forecast is added, it indicates forecast divergence across economies",
                 ]),
                ("Comparison  —  All Metrics  +  3Y Forecast",
                 "All three metrics (CPI, Policy Rate, Real Rate) support 3-year forecast overlays.",
                 &[
                    "  CPI YoY%    forecast = MA-12 recursive",
                    "  Policy Rate forecast = Rate MA-12 recursive",
                    "  Real Rate   forecast = CPI_ma12 − Rate_ma12 (per month)",
                    "  Lighter dashed extensions visible for all selected economies",
                    "  Real Rate < 0 in forecast = policy expected to remain accommodative in real terms",
                 ]),
            ];

            for (title, summary, bullets) in descs.iter() {
                ui.add_space(10.0);
                egui::Frame::none().fill(Color32::from_rgb(18,22,30))
                    .inner_margin(egui::Margin::same(10.0)).rounding(6.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new(*title).strong().size(13.5)
                            .color(Color32::from_rgb(100,200,255)));
                        ui.add_space(4.0); ui.label(*summary); ui.add_space(4.0);
                        for b in bullets.iter() {
                            ui.label(RichText::new(*b).weak().size(11.5));
                        }
                    });
            }

            ui.add_space(16.0);
            ui.label(RichText::new("Python ML Pipeline (Backend)").strong().size(15.0));
            ui.separator();
            egui::Frame::none().fill(Color32::from_rgb(18,22,30))
                .inner_margin(egui::Margin::same(10.0)).rounding(6.0)
                .show(ui, |ui| {
                    egui::Grid::new("ml_grid").num_columns(2).spacing([20.0,6.0]).show(ui, |ui| {
                        ui.label(RichText::new("01_data_prep.py").strong().color(Color32::from_rgb(255,152,0)));
                        ui.label("Global panel preprocessing v3.1 — CPI index to YoY%, Taylor Rule deviation, inverse class weights");
                        ui.end_row();
                        ui.label(RichText::new("02_lstm_baseline.py").strong().color(Color32::from_rgb(33,150,243)));
                        ui.label("BiLSTM + Temporal Attention multitask v3 — CPI regression + rate decision 3-class");
                        ui.end_row();
                        ui.label(RichText::new("03_tft_model.py").strong().color(Color32::from_rgb(156,39,176)));
                        ui.label("Temporal Fusion Transformer v3 — multi-horizon quantile forecast (q10/q50/q90)");
                        ui.end_row();
                        ui.label(RichText::new("Baseline targets").strong().color(Color32::from_rgb(76,175,80)));
                        ui.label("LSTM: Persistence MAE x0.70  /  TFT: Persistence MAE x0.60");
                        ui.end_row();
                    });
                });
            ui.add_space(20.0);
        });
    }

    // ----------------------------------------------------------
    // Bottom status bar
    // ----------------------------------------------------------
    fn show_desc_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("© lightgo  |  lightgo1230@gmail.com")
                .size(10.0).color(Color32::from_rgb(120, 160, 220)));
            ui.separator();
            let eco_info = self.current_eco().map(|e| {
                let corr = if e.corr_rate_cpi.is_nan() { "N/A".into() } else { format!("{:+.3}", e.corr_rate_cpi) };
                format!("{}  r={}  Hike{} Hold{} Cut{}  CPI[{:.1}%~{:.1}%]  Fcst→{}",
                    e.name, corr, e.n_hike, e.n_hold, e.n_cut, e.cpi_min, e.cpi_max,
                    ym_to_str(e.last_date + FORECAST_MONTHS as f64/12.0))
            }).unwrap_or_default();

            let desc = match self.tab {
                Tab::Prediction  => format!("[CPI+3Y] {}", eco_info),
                Tab::Scatter     => format!("[Scatter+Traj] {} | Magenta=MA12 trajectory", eco_info),
                Tab::Heatmap     => format!("[Heatmap] {} | Toggle sidebar: Historical / +3Y Forecast",
                    if self.hmap_show_fcst { "Historical+Forecast" } else { "Historical Only" }),
                Tab::Comparison  => {
                    let m = match self.compare_metric {
                        CompareMetric::CpiYoY    => "CPI YoY% + MA12 fcst",
                        CompareMetric::PolicyRate => "Policy Rate + MA12 fcst",
                        CompareMetric::RealRate   => "Real Rate + (CPI-Rate) fcst",
                    };
                    format!("[Comparison] {}  metric={}", eco_info, m)
                }
                Tab::About => "[About] All tabs include 3-year forecast (MA-3/6/12 + OLS linear)".into(),
            };

            ui.label(RichText::new(desc).size(11.0).color(Color32::from_gray(190)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let c = if self.status_msg.contains("failed") { Color32::from_rgb(244,67,54) }
                        else { Color32::from_rgb(76,175,80) };
                ui.label(RichText::new(&self.status_msg).color(c).size(10.5));
            });
        });
    }
}

// ================================================================
// eframe App
// ================================================================

impl eframe::App for InflationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Global Inflation vs Interest Rates")
                    .strong().color(Color32::from_rgb(100,200,255)).size(16.0));
                ui.add_space(16.0);
                for (tab, lbl) in [
                    (Tab::Prediction, "CPI Forecast"),
                    (Tab::Scatter,    "Rate vs CPI"),
                    (Tab::Heatmap,    "Heatmap"),
                    (Tab::Comparison, "Comparison"),
                    (Tab::About,      "About"),
                ] {
                    if ui.selectable_label(self.tab == tab, lbl).clicked() { self.tab = tab; }
                    ui.add_space(2.0);
                }
            });
            ui.add_space(3.0);
        });

        egui::TopBottomPanel::bottom("desc")
            .min_height(28.0).max_height(36.0)
            .show(ctx, |ui| { self.show_desc_bar(ui); });

        egui::SidePanel::left("side")
            .min_width(170.0).max_width(230.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(RichText::new("Economy").strong());
                ui.separator();
                ScrollArea::vertical().show(ui, |ui| {
                    for (i, eco) in self.economies.iter().enumerate() {
                        let corr  = self.eco_map.get(eco).map(|e| e.corr_rate_cpi).unwrap_or(f64::NAN);
                        let label = if corr.is_nan() { eco.clone() }
                                    else { format!("{}  ({:+.2})", eco, corr) };
                        if ui.selectable_label(self.selected_eco == i, label).clicked() {
                            self.selected_eco = i;
                        }
                    }
                });
                ui.separator();
                ui.label(RichText::new("Value = Rate vs CPI Pearson r").weak().size(9.5));

                // ── Prediction tab options
                if self.tab == Tab::Prediction {
                    ui.add_space(6.0); ui.separator();
                    ui.label(RichText::new("Historical MA").strong().size(11.0));
                    ui.checkbox(&mut self.show_ma3,          "MA-3");
                    ui.checkbox(&mut self.show_ma6,          "MA-6");
                    ui.checkbox(&mut self.show_ma12,         "MA-12");
                    ui.checkbox(&mut self.show_rate_on_pred, "Policy Rate");
                }

                // ── Heatmap tab options
                if self.tab == Tab::Heatmap {
                    ui.add_space(6.0); ui.separator();
                    ui.label(RichText::new("Heatmap Mode").strong().size(11.0));
                    ui.radio_value(&mut self.hmap_show_fcst, false, "Historical Only");
                    ui.radio_value(&mut self.hmap_show_fcst, true,  "Historical + 3Y Forecast");
                    ui.label(RichText::new("Forecast mode extends each\neconomy's CPI with MA-12\nbefore computing Pearson r").weak().size(9.0));
                }

                // ── Global forecast options (all tabs)
                ui.add_space(6.0); ui.separator();
                ui.label(RichText::new("3-Year Forecast").strong().size(11.0)
                    .color(Color32::from_rgb(150,220,255)));
                ui.checkbox(&mut self.show_fcst, "Show All Forecasts");
                if self.show_fcst {
                    if self.tab == Tab::Prediction {
                        ui.checkbox(&mut self.show_linear_fcst, "OLS Linear Trend");
                    }
                    let eco = self.eco_map.values().next();
                    if let Some(e) = eco {
                        ui.label(RichText::new(format!(
                            "{} → {}", ym_to_str(e.last_date),
                            ym_to_str(e.last_date + FORECAST_MONTHS as f64/12.0)
                        )).weak().size(9.0).color(Color32::from_rgb(150,220,255)));
                    }
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(2.0);
            match self.tab {
                Tab::Prediction => self.show_prediction(ui),
                Tab::Scatter    => self.show_scatter(ui),
                Tab::Heatmap    => self.show_heatmap(ui),
                Tab::Comparison => self.show_comparison(ui),
                Tab::About      => self.show_about(ui),
            }
        });
    }
}

// ================================================================
// Entry point
// ================================================================

fn main() -> eframe::Result<()> {
    let icon = egui::viewport::IconData { rgba: make_icon_rgba(32), width: 32, height: 32 };
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Global Inflation vs Interest Rates — 3Y Forecast Dashboard")
            .with_inner_size([1440.0, 860.0])
            .with_min_inner_size([900.0, 600.0])
            .with_icon(std::sync::Arc::new(icon)),
        ..Default::default()
    };
    eframe::run_native("inflation_viewer", options,
        Box::new(|cc| Ok(Box::new(InflationApp::new(cc)))))
}
