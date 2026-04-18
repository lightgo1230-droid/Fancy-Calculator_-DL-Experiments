# YouTube Trending Video Data Analysis

## Project Overview

An exploratory data analysis (EDA) of **10,000 YouTube trending videos** from **24 countries** spanning **2020 to 2026**.  
The analysis is performed using **pandas**, **numpy**, and **matplotlib**, and automatically exports
20 PNG charts and a Word report as final outputs.

---

## License

```
Copyright 2024 Meruva Kodanda Suraj

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Dataset Attribution

| Item | Detail |
|------|--------|
| **Original Author** | Meruva Kodanda Suraj |
| **License** | Apache License 2.0 |

> This project uses the dataset originally created and published by **Meruva Kodanda Suraj**.  
> All data rights belong to the original author. Any redistribution or derivative work  
> must comply with the terms of the Apache License 2.0.

---

## Directory Structure

```
Desktop/
│
├── youtube_trending_analysis.py            ← Main analysis script
├── YouTube_Trending_Analysis_Report.docx   ← Word analysis report
├── README.md                               ← Project documentation (this file)
│
├── 새 폴더/                                ← Source CSV data directory
│   ├── yearly_trends.csv
│   ├── category_summary.csv
│   ├── country_summary.csv
│   └── trending_videos.csv
│
└── addction_results/                       ← Chart output directory (20 PNG files)
    ├── 01_yearly_views_likes.png
    ├── 02_yearly_yoy_growth.png
    ├── 03_yearly_engagement_duration.png
    ├── 04_category_avg_views.png
    ├── 05_category_bubble.png
    ├── 06_category_duration_boxplot.png
    ├── 07_country_avg_views.png
    ├── 08_country_scatter.png
    ├── 09_correlation_heatmap.png
    ├── 10_views_distribution.png
    ├── 11_views_band_pie.png
    ├── 12_days_to_trend.png
    ├── 13_dayofweek_views.png
    ├── 14_year_month_heatmap.png
    ├── 15_clickbait_distribution.png
    ├── 16_clickbait_comparison.png
    ├── 17_title_features_views.png
    ├── 18_subscriber_tier.png
    ├── 19_viral_radar.png
    └── 20_category_country_heatmap.png
```

---

## Dataset Description

| File | Rows | Description |
|------|------|-------------|
| `yearly_trends.csv` | 7 | Yearly aggregated statistics (2020–2026) |
| `category_summary.csv` | 17 | Per-category aggregated statistics |
| `country_summary.csv` | 23 | Per-country aggregated statistics |
| `trending_videos.csv` | 10,000 | Individual trending video records |

### Key Columns (`trending_videos.csv`)

| Column | Description |
|--------|-------------|
| `video_id` | Unique video identifier |
| `title` | Video title |
| `has_caps_title` | Whether the title contains all-caps words (0/1) |
| `has_emoji_title` | Whether the title contains an emoji (0/1) |
| `has_question_title` | Whether the title is phrased as a question (0/1) |
| `category` | Content category |
| `trending_country` | ISO country code where the video trended |
| `views` | Total view count |
| `likes` | Total like count |
| `comments` | Total comment count |
| `engagement_score` | Composite engagement score |
| `days_to_trend` | Days from upload to reaching trending |
| `clickbait_score` | Clickbait likelihood score (0–1) |
| `subscriber_count` | Channel subscriber count |
| `channel_verified` | Whether the channel is verified (0/1) |
| `duration_seconds` | Video length in seconds |

---

## Getting Started

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib python-docx scipy
```

### 2. Run the Script

```bash
python youtube_trending_analysis.py
```

### 3. Check Outputs

| Output | Location |
|--------|----------|
| 20 charts (PNG) | `Desktop/addction_results/` |
| Analysis report (Word) | `Desktop/YouTube_Trending_Analysis_Report.docx` |

---

## Analysis Techniques

### Descriptive Statistics
- Mean / Median / Standard Deviation / Min / Max
- **Coefficient of Variation (CV%)** — relative dispersion of distributions
- **Quantiles (Q1 / Q3 / IQR)** — basis for outlier detection

### Distribution Analysis
- Histograms (linear vs. log scale comparison)
- **KDE (Kernel Density Estimation)** — continuous distribution estimate for clickbait scores
- Binning — segmenting views and subscriber counts into discrete bands

### Time Series Analysis
- **YoY Growth Rate** — year-over-year change in views and likes
- Monthly and day-of-week pattern analysis
- Year × Month heatmap — 2D time series visualization

### Group Comparison
- GroupBy aggregation — comparisons across category, country, language, weekday, and subscriber tier
- Binary group comparison — clickbait vs. normal videos
- Verified vs. unverified channel performance analysis

### Correlation Analysis
- **Pearson Correlation Coefficient (r)** — linear relationship between engagement and other metrics
- Correlation heatmap — multi-variable correlation structure visualization

### Segmentation Analysis
- Subscriber size tiering into 5 segments
- Views long-tail distribution across 7 bands
- **Viral Top 1% profiling** — characteristic analysis of extreme outlier group

### Additional Techniques
- Cross-tabulation — category × country frequency analysis
- Derived ratio metrics — like rate, comment rate
- **Radar chart** — multi-dimensional comparison of viral vs. overall average
- Viral-baseline normalization — relative multiplier calculation

---

## Key Findings

| Item | Finding |
|------|---------|
| Total Cumulative Views | 31.17B |
| Total Cumulative Likes | 1.64B |
| Best Performing Year | 2024 (Views +75% YoY) |
| Top Category by Views | Sports / Shows (4.6M+ avg) |
| Top Countries by Views | Russia (8.4M), Turkey (8.0M) |
| Highest Engagement Country | Philippines (8.27) |
| Longest Video Category | Gaming (avg ~25 min) |
| Shortest Video Category | Shorts (avg 37 sec) |
| Best Day to Upload | Sunday (avg 5.86M views) |
| Best Month to Upload | November (avg 6.18M views) |
| Viral Top 1% Threshold | 57.1M+ views |
| Clickbait Prevalence | ~41% of all videos |
| Same-Day Trending (0 days) | 15.0% of videos |
| Trending Within 1–3 Days | 59.7% of videos |

---

## Tech Stack

| Item | Version |
|------|---------|
| Python | 3.13.9 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |
| matplotlib | 3.10.6 |
| python-docx | latest |
| OS | Windows 11 |
