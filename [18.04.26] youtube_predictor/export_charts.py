"""
export_charts.py
Generates one PNG chart per view (All / 2026 / 2027 / 2028)
and saves them to addction_results/.
Matches the Lightgo dashboard colour scheme.
"""

import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\USER\OneDrive\Desktop\youtube_predictor\data"
OUT_DIR  = r"C:\Users\USER\OneDrive\Desktop\결과물\addction_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────
C_BG      = "#0A0C12"
C_SURFACE = "#12161F"
C_CARD    = "#181D2A"
C_BORDER  = "#2C3448"
C_TEXT    = "#DAE4F5"
C_MUTED   = "#64708C"
C_ACTUAL  = "#94A3B8"
C_BASIC   = "#60A5FA"
C_ADV     = "#34D399"
C_SEL     = "#FBBF24"
C_DIM     = (0.039, 0.047, 0.071, 0.70)   # rgba for dimmed overlay

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Model helpers ──────────────────────────────────────────────
def vf(v): return np.array(v, dtype=np.float32)
def mf(v): return np.array(v, dtype=np.float32)

def load_model(path):
    with open(path) as f: j = json.load(f)
    nb = j["n_blocks"]
    return dict(
        sl  = dict(w=mf(j["stem"]["linear"]["weight"]),   b=vf(j["stem"]["linear"]["bias"])),
        sbn = dict(w=vf(j["stem"]["bn"]["weight"]),        b=vf(j["stem"]["bn"]["bias"]),
                   mean=vf(j["stem"]["bn"]["running_mean"]),
                   var=vf(j["stem"]["bn"]["running_var"]),  eps=j["stem"]["bn"]["eps"]),
        blocks=[
            dict(
                l1=dict(w=mf(j["blocks"][i]["linear1"]["weight"]),  b=vf(j["blocks"][i]["linear1"]["bias"])),
                b1=dict(w=vf(j["blocks"][i]["bn1"]["weight"]),       b=vf(j["blocks"][i]["bn1"]["bias"]),
                        mean=vf(j["blocks"][i]["bn1"]["running_mean"]),
                        var=vf(j["blocks"][i]["bn1"]["running_var"]),  eps=j["blocks"][i]["bn1"]["eps"]),
                l2=dict(w=mf(j["blocks"][i]["linear2"]["weight"]),  b=vf(j["blocks"][i]["linear2"]["bias"])),
                b2=dict(w=vf(j["blocks"][i]["bn2"]["weight"]),       b=vf(j["blocks"][i]["bn2"]["bias"]),
                        mean=vf(j["blocks"][i]["bn2"]["running_mean"]),
                        var=vf(j["blocks"][i]["bn2"]["running_var"]),  eps=j["blocks"][i]["bn2"]["eps"]),
            ) for i in range(nb)
        ],
        hl1 = dict(w=mf(j["head"]["linear1"]["weight"]), b=vf(j["head"]["linear1"]["bias"])),
        hl2 = dict(w=mf(j["head"]["linear2"]["weight"]), b=vf(j["head"]["linear2"]["bias"])),
    )

def load_tmpl(path):
    with open(path) as f: j = json.load(f)
    return dict(
        subs_idx  = j["subs_feature_idx"],
        data      = [vf(j["templates"][str(m)]) for m in range(1, 13)],
        slope     = float(j["subs_slope_per_month"]),
        last_subs = float(j["last_log_subs"]),
        sc_mean   = vf(j["scaler_mean"]),
        sc_scale  = vf(j["scaler_scale"]),
    )

def fwd_lin(x, l):  return l["w"] @ x + l["b"]
def fwd_bn(x, bn):  return bn["w"] * (x - bn["mean"]) / np.sqrt(bn["var"] + bn["eps"]) + bn["b"]
def gelu(x):        return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

def fwd_block(x, blk):
    h = gelu(fwd_bn(fwd_lin(x, blk["l1"]), blk["b1"]))
    h = fwd_bn(fwd_lin(h, blk["l2"]), blk["b2"])
    return gelu(x + h)

def infer(model, xs):
    x = gelu(fwd_bn(fwd_lin(xs, model["sl"]), model["sbn"]))
    for blk in model["blocks"]: x = fwd_block(x, blk)
    return fwd_lin(gelu(fwd_lin(x, model["hl1"])), model["hl2"])[0]

def gen_fc(model, tmpl, n=36):
    out = []
    for i in range(n):
        feat = tmpl["data"][i % 12].copy()
        feat[tmpl["subs_idx"]] = tmpl["last_subs"] + tmpl["slope"] * (i + 1)
        xs = (feat - tmpl["sc_mean"]) / tmpl["sc_scale"]
        out.append((float(np.exp(infer(model, xs))) - 1.0) / 1_000_000.0)
    return out

# ── Load data ──────────────────────────────────────────────────
print("Loading historical data...")
with open(f"{DATA_DIR}/historical_monthly.json") as f: dj = json.load(f)
hist_labels = dj["labels"]
hist_raw    = [v / 1_000_000.0 for v in dj["views"]]

print("Running model inference...")
ma, mb = load_model(f"{DATA_DIR}/model_a_weights.json"), load_model(f"{DATA_DIR}/model_b_weights.json")
ta, tb = load_tmpl(f"{DATA_DIR}/templates_a.json"),      load_tmpl(f"{DATA_DIR}/templates_b.json")
raw_a, raw_b = gen_fc(ma, ta, 36), gen_fc(mb, tb, 36)

hl       = len(hist_labels)
last_h   = hist_raw[-1]
scale_a  = last_h / raw_a[0] if raw_a[0] > 0 else 1.0
scale_b  = last_h / raw_b[0] if raw_b[0] > 0 else 1.0
raw_a    = [v * scale_a for v in raw_a]
raw_b    = [v * scale_b for v in raw_b]

fc_labels = [f"{['2026','2027','2028'][i//12]}  {MONTHS[i%12]}" for i in range(36)]
all_labels = hist_labels + fc_labels
total      = len(all_labels)

# Full index arrays
x_all   = list(range(total))
x_hist  = list(range(hl))
y_hist  = hist_raw

# Forecast including bridge point (hl-1 = last historical value)
x_fc    = [hl - 1] + list(range(hl, hl + 36))
y_bas   = [last_h] + raw_a
y_adv   = [last_h] + raw_b

all_y   = y_hist + raw_a
y_min   = min(all_y) * 0.88
y_max   = max(all_y) * 1.15

# ── Chart factory ──────────────────────────────────────────────
def make_chart(title_suffix, x_focus=None, spotlight=False):
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    # Grid
    ax.grid(color=C_BORDER, linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Forecast zone tint
    ax.axvspan(hl - 0.5, total - 0.5, color="#5078FF", alpha=0.04, zorder=1)

    # Year dividers + labels
    yr_names = ["2020","2021","2022","2023","2024","2025","2026","2027","2028"]
    for yi, name in enumerate(yr_names):
        xi = yi * 12
        ax.axvline(xi, color="#3C4664", linewidth=0.8, alpha=0.7, zorder=2)
        ax.text(xi + 0.4, y_max * 0.985, name,
                color="#8C9BB9", fontsize=8.5, alpha=0.85,
                va="top", ha="left", zorder=3)

    # Actual/Forecast boundary
    ax.axvline(hl, color=C_SEL, linewidth=1.2, alpha=0.55, zorder=3)
    ax.text(hl - 0.5, y_max * 0.88, "ACTUAL",
            color=C_ACTUAL, fontsize=8, alpha=0.75, ha="right", va="top")
    ax.text(hl + 0.5, y_max * 0.88, "FORECAST",
            color=C_BASIC,  fontsize=8, alpha=0.75, ha="left",  va="top")

    # Area fills
    ax.fill_between(x_hist, y_hist, y_min,  color=C_ACTUAL, alpha=0.06, zorder=2)
    ax.fill_between(x_fc,   y_bas,  y_min,  color=C_BASIC,  alpha=0.06, zorder=2)
    ax.fill_between(x_fc,   y_adv,  y_min,  color=C_ADV,    alpha=0.06, zorder=2)

    # Data lines
    ax.plot(x_hist, y_hist, color=C_ACTUAL, linewidth=2.0,  label="Actual Data",          zorder=4)
    ax.plot(x_fc,   y_bas,  color=C_BASIC,  linewidth=2.2,  label="Basic Prediction (63%)", zorder=4)
    ax.plot(x_fc,   y_adv,  color=C_ADV,    linewidth=2.2,  label="Advanced Prediction (99%)", zorder=4)

    # Spotlight dim overlay
    if spotlight and x_focus is not None:
        lo, hi = x_focus
        if lo > 0:
            ax.axvspan(-1, lo, color=C_DIM[:3], alpha=C_DIM[3], zorder=5)
        if hi < total - 1:
            ax.axvspan(hi + 1, total, color=C_DIM[:3], alpha=C_DIM[3], zorder=5)
        # Focus border
        ax.axvspan(lo, hi + 1,
                   facecolor=(1,1,1,0.02),
                   edgecolor=(251/255, 191/255, 36/255, 0.5),
                   linewidth=1.5, zorder=6)

    # Axes styling
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-1, total)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_BORDER)

    # X-axis: show year labels at multiples of 12
    xticks = [i * 12 for i in range(len(yr_names))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(yr_names, color=C_MUTED, fontsize=9)

    # Y-axis: format as M
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}M"))
    ax.tick_params(axis="y", colors=C_MUTED)

    # Title
    fig.text(0.013, 0.97,
             f"YouTube Views Dashboard — {title_suffix}",
             color=C_TEXT, fontsize=13, fontweight="bold", va="top")
    fig.text(0.013, 0.925,
             "Actual 2020–2025  ·  AI Prediction 2026–2028  (views in millions)",
             color=C_MUTED, fontsize=9, va="top")

    # Legend
    legend_handles = [
        Line2D([0],[0], color=C_ACTUAL, linewidth=2,   label="Actual Data"),
        Line2D([0],[0], color=C_BASIC,  linewidth=2.2, label="Basic Prediction  (AI · 63% accuracy)"),
        Line2D([0],[0], color=C_ADV,    linewidth=2.2, label="Advanced Prediction  (AI · 99% accuracy)"),
    ]
    legend = ax.legend(handles=legend_handles,
                       loc="upper left", framealpha=0.25,
                       facecolor=C_CARD, edgecolor=C_BORDER,
                       labelcolor=C_TEXT, fontsize=9)

    # Watermark
    fig.text(0.988, 0.015, "Lightgo · 2026",
             color=C_MUTED, fontsize=8, ha="right", va="bottom", alpha=0.6)

    plt.tight_layout(rect=[0, 0.0, 1, 0.92])
    return fig

# ── Generate 4 charts ──────────────────────────────────────────
views = [
    ("All Years",  None,             False),
    ("2026",       (hl,     hl + 11), True),
    ("2027",       (hl + 12, hl + 23), True),
    ("2028",       (hl + 24, hl + 35), True),
]

for label, focus, spot in views:
    safe = label.replace(" ", "_")
    path = os.path.join(OUT_DIR, f"chart_{safe}.png")
    print(f"  Rendering {label} → {path}")
    fig = make_chart(label, x_focus=focus, spotlight=spot)
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone — {len(views)} charts saved to:\n  {OUT_DIR}")
