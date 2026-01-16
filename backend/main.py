# ================================================
# AQI Analysis API — Continuous ANOVA Edition
# Updated Backend (Part 1/6)
# ================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import (
    spearmanr, shapiro, kstest, normaltest, ttest_ind,
    mannwhitneyu, f_oneway
)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot

import itertools
import base64
from io import BytesIO
import os
from dotenv import load_dotenv
import json

# ================================================
# FASTAPI + CORS (RENDER <-> RENDER)
# ================================================

load_dotenv()
app = FastAPI(title="AQI Analysis API", version="4.0.0")

FRONTEND = "https://aqi-new-2.onrender.com"
BACKEND = "https://aqi-new-1.onrender.com"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aqi-new-1.onrender.com",
        "https://aqi-new-2.onrender.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/{path:path}")
async def preflight_handler(path: str):
    return JSONResponse({"status": "ok"})

@app.middleware("http")
async def add_render_cors(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = FRONTEND
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ================================================
# GLOBAL DATA CACHE
# ================================================

DATA_CACHE = {}

@app.on_event("startup")
async def preload_data():
    global DATA_CACHE
    try:
        if os.path.exists('data/aqi_data_waqi.csv'):
            df = pd.read_csv('data/aqi_data_waqi.csv')
            DATA_CACHE['df'] = df
            print(f"✓ Loaded {len(df)} WAQI rows")
        elif os.path.exists('data/aqi_data_openaq.csv'):
            df = pd.read_csv('data/aqi_data_openaq.csv')
            DATA_CACHE['df'] = df
            print(f"✓ Loaded {len(df)} OpenAQ rows")
        else:
            print("⚠ No AQI dataset found at startup.")
    except Exception as e:
        print("Startup error:", e)





# ================================================
# REQUEST MODELS
# ================================================

class AnalysisRequest(BaseModel):
    cities: list
    year: int
    predict_months: int | None = None
    predict_year: int | None = None

class TTestRequest(BaseModel):
    city1: str
    city2: str
    pollutant: str
    year: int

class ANOVARequest(BaseModel):
    cities: list
    pollutant: str
    year: int

class ChiRequest(BaseModel):
    cities: list
    year: int

# ================================================
# UTILITY HELPERS
# ================================================

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def convert_np(o):
    """Convert numpy + pandas objects to native Python types."""
    if isinstance(o, dict):
        return {k: convert_np(v) for k, v in o.items()}
    if isinstance(o, list):
        return [convert_np(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o

def pollutant_to_cpcb_category(p, v):
    if pd.isna(v): return None
    p = p.lower()

    if p == "pm25":
        return (
            "Good" if v<=30 else
            "Satisfactory" if v<=60 else
            "Moderate" if v<=90 else
            "Poor" if v<=120 else
            "Very Poor" if v<=250 else
            "Severe"
        )
    if p == "pm10":
        return (
            "Good" if v<=50 else
            "Satisfactory" if v<=100 else
            "Moderate" if v<=250 else
            "Poor" if v<=350 else
            "Very Poor" if v<=430 else
            "Severe"
        )
    if p == "no2":
        return (
            "Good" if v<=40 else
            "Satisfactory" if v<=80 else
            "Moderate" if v<=180 else
            "Poor" if v<=280 else
            "Very Poor" if v<=400 else
            "Severe"
        )
    if p == "so2":
        return (
            "Good" if v<=40 else
            "Satisfactory" if v<=80 else
            "Moderate" if v<=380 else
            "Poor" if v<=800 else
            "Very Poor" if v<=1600 else
            "Severe"
        )
    if p == "o3":
        return (
            "Good" if v<=50 else
            "Satisfactory" if v<=100 else
            "Moderate" if v<=168 else
            "Poor" if v<=208 else
            "Very Poor" if v<=748 else
            "Severe"
        )
    if p == "co":
        return (
            "Good" if v<=1 else
            "Satisfactory" if v<=2 else
            "Moderate" if v<=10 else
            "Poor" if v<=17 else
            "VeryVery Poor" if v<=34 else
            "Severe"
        )
    return None
# ================================================
# DATA LOAD + FILTER SECTION
# ================================================

def load_filtered(cities, year):
    if 'df' not in DATA_CACHE:
        return pd.DataFrame()

    df = DATA_CACHE['df'].copy()
    if 'city' not in df.columns:
        return pd.DataFrame()

    df = df[df['city'].isin(cities)]

    # Timestamp normalization
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['year'] = df['timestamp'].dt.year
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year

    df = df[df['year'] == year]
    return df.reset_index(drop=True)

# ================================================
# SUMMARY + VARIABILITY
# ================================================

def calculate_summary(df):
    val = 'aqi' if 'aqi' in df.columns else 'value'
    x = df[val].dropna()
    if len(x) == 0:
        return {"count": 0}

    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
        "quartiles": {
            "Q1": float(x.quantile(0.25)),
            "Q2": float(x.quantile(0.50)),
            "Q3": float(x.quantile(0.75))
        }
    }

def calculate_variability(df):
    val = 'aqi' if 'aqi' in df.columns else 'value'
    x = df[val].dropna()
    if len(x) < 2:
        return {"std_dev":None,"iqr":None,"variance":None,"cv":None}

    mean = x.mean()
    return {
        "std_dev": float(x.std()),
        "iqr": float(x.quantile(0.75)-x.quantile(0.25)),
        "variance": float(x.var()),
        "cv": float((x.std()/mean)*100) if mean!=0 else None
    }

# ================================================
# QQ PLOTS (AQI + Pollutants)
# ================================================

from scipy.stats import probplot

def generate_qq(df):
    pollutants = ["aqi","pm25","pm10","o3","no2","so2","co"]
    qq = {}

    for p in pollutants:
        if p not in df.columns: 
            continue

        vals = df[p].dropna()
        if len(vals) < 5:
            continue

        fig = plt.figure(figsize=(4,4), dpi=100)
        ax = fig.add_subplot(111)
        probplot(vals, dist="norm", plot=ax)
        plt.tight_layout()
        qq[p] = fig_to_base64(fig)
        plt.close(fig)

    return qq

# ================================================
# CORRELATION (Spearman matrix + pairwise significance)
# ================================================

def compute_correlation(df):
    cols = ["aqi","pm25","pm10","o3","no2","so2","co"]
    use = [c for c in cols if c in df.columns]

    if len(use) < 2:
        return {"matrix":None,"pairs":None}

    data = df[use].dropna()
    if len(data) < 5:
        return {"matrix":None,"pairs":None}

    # ----- MATRIX (Spearman) -----
    corr_mat, p_mat = {}, {}
    for i in use:
        corr_mat[i] = {}
        p_mat[i] = {}
        for j in use:
            rho, p = spearmanr(data[i], data[j], nan_policy="omit")
            corr_mat[i][j] = float(rho)
            p_mat[i][j] = float(p)

    # ----- PAIRWISE TESTS -----
    pairs = {}
    for a, b in itertools.combinations(use, 2):
        d1, d2 = data[a], data[b]

        # Always Spearman here (non-parametric)
        rho, p = spearmanr(d1, d2, nan_policy="omit")
        method = "Spearman (non-parametric)"

        # A-style wording
        H0 = f"No monotonic correlation between {a.upper()} and {b.upper()}"
        H1 = f"Monotonic correlation exists between {a.upper()} and {b.upper()}"

        # Style-2 decision
        decision = (
            "Reject Null Hypothesis (H₀) → Significant correlation"
            if p < 0.05 else
            "Accept Null Hypothesis (H₀) → No significant correlation"
        )

        # p formatting
        if p < 1e-4:
            p_fmt = "< 0.0001"
        else:
            p_fmt = f"{p:.5f}"

        pairs[f"{a}-{b}"] = {
            "rho": float(rho),
            "p": float(p),
            "p_fmt": p_fmt,
            "method": method,
            "H0": H0,
            "H1": H1,
            "decision": decision,
            "significant": bool(p < 0.05),
        }

    return {"matrix": corr_mat, "pairs": pairs}


# ================================================
# VISUALIZATION SECTION (BOX, DENSITY, HIST, ETC)
# ================================================

def generate_visualizations(df):
    plots = {}
    if 'city' not in df.columns:
        return plots

    val = 'aqi' if 'aqi' in df.columns else 'value'
    df = df.dropna(subset=[val])
    if df.empty:
        return plots

    # BOX
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=df, x='city', y=val, ax=ax)
    plt.xticks(rotation=20)
    plt.title("AQI Distribution by City")
    plots['boxplot'] = fig_to_base64(fig)

    # HISTOGRAM
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df[val], bins=30, color='skyblue', edgecolor='black')
    plt.title("AQI Histogram")
    plots['histogram'] = fig_to_base64(fig)

    # DENSITY
    fig, ax = plt.subplots(figsize=(8,5))
    for c in df['city'].unique():
        sns.kdeplot(df[df['city']==c][val], fill=True, label=c, ax=ax)
    plt.title("AQI Density")
    plt.legend()
    plots['density'] = fig_to_base64(fig)

    # VIOLIN
    fig, ax = plt.subplots(figsize=(9,5))
    sns.violinplot(data=df, x='city', y=val, ax=ax)
    plt.title("AQI Violin Plot")
    plt.xticks(rotation=20)
    plots['violin'] = fig_to_base64(fig)

    # SCATTER (City timeline)
    if 'timestamp' in df.columns:
        fig, ax = plt.subplots(figsize=(9,5))
        ax.scatter(df['timestamp'], df[val], s=8)
        plt.xticks(rotation=25)
        plt.title("Scatter (Time vs AQI)")
        plots['scatter'] = fig_to_base64(fig)

    # HEXBIN
    fig, ax = plt.subplots(figsize=(7,5))
    try:
        ax.hexbin(df[val], df[val].rolling(3).mean(), gridsize=30)
        plt.title("Hexbin Plot")
        plots['hexbin'] = fig_to_base64(fig)
    except:
        pass

    # Spearman Correlation Heatmap
    use = ["aqi","pm25","pm10","o3","no2","so2","co"]
    use = [c for c in use if c in df.columns]
    if len(use)>=2:
        fig, ax = plt.subplots(figsize=(7,5))
        corr = df[use].corr(method='spearman')
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Correlation Heatmap (Spearman)")
        plots['correlation_heatmap'] = fig_to_base64(fig)

    return plots


# ================================================
# AI SUMMARY SECTION
# ================================================

def generate_ai_summary(df, predictions=None):
    stats = calculate_summary(df)
    var = calculate_variability(df)
    if stats.get("count",0)==0:
        return "No data."

    msg = f"""
Summary for {stats['count']} records:

Mean AQI: {stats['mean']:.2f}
Median AQI: {stats['median']:.2f}
Range: {stats['min']} - {stats['max']}
Std Dev: {var.get('std_dev')}
IQR: {var.get('iqr')}
"""

    if predictions and not predictions.get("error"):
        msg += "\nForecasting suggests future trend variation."

    return msg.strip()
# ================================================
# MAIN ANALYZE ROUTE
# ================================================

@app.post("/api/analyze")
async def run_analysis(req: AnalysisRequest):
    df = load_filtered(req.cities, req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    summary = calculate_summary(df)
    variability = calculate_variability(df)
    visuals = generate_visualizations(df)

    # Correlation block
    corr_out = compute_correlation(df)

    # QQ plots
    qq = generate_qq(df)

    # Normality (AQI)
    if 'aqi' in df.columns:
        sample = df['aqi'].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000)

        try:
            stat, pval = shapiro(sample)
            normality_aqi = {
                "statistic": float(stat),
                "pvalue": float(pval),
                "decision": "not_normal" if pval<0.05 else "normal"
            }
        except:
            normality_aqi = {"error":"failed"}
    else:
        normality_aqi = {"error":"missing"}

    # Normality (Pollutants)
    pollutants = ["pm25","pm10","o3","no2","so2","co"]
    normality_pollutants = {}

    for p in pollutants:
        if p not in df.columns:
            continue
        vals = df[p].dropna()
        if len(vals)==0:
            continue

        try:
            v = vals if len(vals)<5000 else vals.sample(5000)
            stat, pval = shapiro(v)
            normality_pollutants[p] = {
                "statistic": float(stat),
                "pvalue": float(pval),
                "is_normal": bool(pval>0.05)
            }
        except:
            normality_pollutants[p] = {"error":"failed"}

    # Forecast
    predictions = None
    if req.predict_months or req.predict_year:
        predictions = generate_predictions(df, req.predict_months, req.predict_year)

    # AI Summary
    ai_summary = generate_ai_summary(df, predictions)

    # ----- FINAL RETURN SHAPE (MATCH FRONTEND) -----
    return convert_np({
        "summary_stats": summary,
        "variability_metrics": variability,
        "visualizations": visuals,

        # Correlation
        "correlation_matrix": {
            "columns": list(corr_out["matrix"].keys()) if corr_out["matrix"] else None,
            "data": [list(corr_out["matrix"][r].values()) for r in corr_out["matrix"]] if corr_out["matrix"] else None
        } if corr_out["matrix"] else None,

        "correlation_pairs": corr_out["pairs"],  # pairwise rho + p + significance

        # QQ + Normality
        "qqplot_aqi": qq.get("aqi"),
        "qqplots_pollutants": {p:qq.get(p) for p in pollutants if p in qq},

        "normality_aqi": normality_aqi,
        "normality_pollutants": normality_pollutants,

        # Forecast + summary text
        "predictions": predictions,
        "ai_summary": ai_summary
    })

# ================================================
# SECTION — EFFECT SIZE
# ================================================

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    pooled = np.sqrt(((nx-1)*np.var(x,ddof=1)+(ny-1)*np.var(y,ddof=1)) / (nx+ny-2))
    if pooled == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled

# ================================================
# SECTION — CPCB HELPER (for Stacked Visuals)
# ================================================

def category_counts(df, col, cats=None):
    vc = df[col].value_counts()
    if cats:
        return {c:int(vc.get(c,0)) for c in cats}
    return vc.to_dict()

# ================================================
# SECTION — T-TEST (Continuous)
# ================================================

@app.post("/api/ttest")
async def ttest_api(req: TTestRequest):
    df = load_filtered([req.city1, req.city2], req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    pol = req.pollutant.lower()
    if pol not in df.columns:
        raise HTTPException(status_code=400, detail=f"Pollutant '{pol}' not found")

    df = df.dropna(subset=[pol])
    c1 = df[df['city']==req.city1][pol].astype(float)
    c2 = df[df['city']==req.city2][pol].astype(float)

    if len(c1)<3 or len(c2)<3:
        raise HTTPException(status_code=400, detail="Insufficient samples for test")

    # ---------- Normality ----------
    sh1, p1 = shapiro(c1) if len(c1)<5000 else (None, 0.01)
    sh2, p2 = shapiro(c2) if len(c2)<5000 else (None, 0.01)
    normal = (p1>0.05 and p2>0.05)

    # ---------- Hypotheses ----------
    H0 = f"No difference in {pol.upper()} levels between {req.city1} and {req.city2}"
    H1 = f"{pol.upper()} levels differ between {req.city1} and {req.city2}"

    # ---------- Test Selection ----------
    if normal:
        method = "Welch t-test"
        stat, p = ttest_ind(c1, c2, equal_var=False)

        # Welch degrees of freedom
        df_w = (
            (c1.var()/len(c1) + c2.var()/len(c2))**2 /
            ((c1.var()/len(c1))**2/(len(c1)-1) + (c2.var()/len(c2))**2/(len(c2)-1))
        )
        df_final = float(df_w)

    else:
        method = "Mann–Whitney U (non-parametric)"
        stat, p = mannwhitneyu(c1, c2, alternative="two-sided")
        df_final = None  # MWU has no df

    # ---------- Effect Size (Cohen's d) ----------
    effect = cohens_d(c1.values, c2.values)

    # ---------- Decision (Style-2) ----------
    decision = (
        "Reject Null Hypothesis (H₀) → Significant difference"
        if p < 0.05 else
        "Accept Null Hypothesis (H₀) → No significant difference"
    )

    # ---------- p Formatting ----------
    p_fmt = "< 0.0001" if p < 1e-4 else f"{p:.5f}"

    # ---------- STACKED VISUALS (unchanged) ----------
    cats = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    df['cat'] = df[pol].apply(lambda v: pollutant_to_cpcb_category(pol,v))
    c1_cats = category_counts(df[df['city']==req.city1], 'cat', cats)
    c2_cats = category_counts(df[df['city']==req.city2], 'cat', cats)

    # Stacked Bar
    fig, ax = plt.subplots(figsize=(8,5))
    bottom1, bottom2 = 0, 0
    for cat in cats:
        v1, v2 = c1_cats.get(cat,0), c2_cats.get(cat,0)
        ax.bar(req.city1, v1, bottom=bottom1, label=cat if bottom1==0 else "")
        ax.bar(req.city2, v2, bottom=bottom2)
        bottom1 += v1; bottom2 += v2
    ax.set_title(f"{pol.upper()} Category Counts")
    stacked_bar = fig_to_base64(fig)

    # Stacked Area
    fig, ax = plt.subplots(figsize=(8,5))
    idx = np.arange(len(cats))
    arr1 = np.array([c1_cats.get(c,0) for c in cats])
    arr2 = np.array([c2_cats.get(c,0) for c in cats])
    ax.stackplot(idx, arr1, arr2, labels=[req.city1, req.city2])
    ax.set_xticks(idx)
    ax.set_xticklabels(cats, rotation=25)
    ax.set_title(f"{pol.upper()} Category Area Plot")
    ax.legend()
    stacked_area = fig_to_base64(fig)

    # Density
    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(c1, fill=True, label=req.city1)
    sns.kdeplot(c2, fill=True, label=req.city2)
    ax.set_title(f"{pol.upper()} Density")
    ax.legend()
    density = fig_to_base64(fig)

    # Box
    fig, ax = plt.subplots(figsize=(6,5))
    ax.boxplot([c1, c2], labels=[req.city1, req.city2])
    ax.set_title(f"{pol.upper()} Box Plot")
    box = fig_to_base64(fig)

    return convert_np({
        "method": method,
        "pollutant": pol.upper(),
        "city1": req.city1,
        "city2": req.city2,

        "H0": H0,
        "H1": H1,

        "statistic": float(stat),
        "pvalue": float(p),
        "p_fmt": p_fmt,
        "df": df_final,
        "effect_size": float(effect),
        "normality": {req.city1:p1, req.city2:p2},
        "decision": decision,

        "stacked_bar": stacked_bar,
        "stacked_area": stacked_area,
        "density_plot": density,
        "box_plot": box,

        "categories_raw": {
            req.city1: c1_cats,
            req.city2: c2_cats
        }
    })
    

# ================================================
# SECTION — ANOVA (Continuous + Tukey Post-Hoc)
# ================================================

@app.post("/api/anova")
async def anova_api(req: ANOVARequest):
    df = load_filtered(req.cities, req.year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selection")

    pol = req.pollutant.lower()
    if pol not in df.columns:
        raise HTTPException(status_code=400, detail=f"Pollutant '{pol}' missing")

    df = df.dropna(subset=[pol])
    if len(df['city'].unique()) < 2:
        raise HTTPException(status_code=400, detail="Need 2+ cities for ANOVA")

    # ----- Groups -----
    groups = [df[df['city']==c][pol] for c in req.cities]
    if sum(len(g)>2 for g in groups) < 2:
        raise HTTPException(status_code=400, detail="Insufficient samples")

    # ----- ANOVA -----
    fstat, pval = f_oneway(*groups)

    # ----- DF -----
    k = len(groups)
    N = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = N - k

    # ----- H0 / H1 (A-style) -----
    H0 = f"Mean {pol.upper()} levels are equal across selected cities"
    H1 = f"At least one city differs in {pol.upper()} levels"

    # ----- Decision (Style-2) -----
    decision = (
        "Reject Null Hypothesis (H₀) → Significant difference"
        if pval < 0.05 else
        "Accept Null Hypothesis (H₀) → No significant difference"
    )

    # ----- p formatting -----
    p_fmt = "< 0.0001" if pval < 1e-4 else f"{pval:.5f}"

    # ----- Effect Size (η²) -----
    grand_mean = df[pol].mean()
    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)
    ss_total = sum((x-grand_mean)**2 for x in df[pol])
    eta_sq = ss_between/ss_total if ss_total>0 else 0.0

    # ----- Tukey -----
    tukey = pairwise_tukeyhsd(df[pol], df['city'], alpha=0.05)
    tbl = tukey.summary()
    cols = tbl.data[0]
    rows = tbl.data[1:]
    tukey_df = pd.DataFrame(rows, columns=cols)
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    # Tukey table as JSON
    tukey_table = []
    for _, r in tukey_df.iterrows():
        dec = (
            "Reject Null Hypothesis (H₀) → Significant difference"
            if r["reject"] else
            "Accept Null Hypothesis (H₀) → No difference"
        )
        tukey_table.append({
            "group1": r["group1"],
            "group2": r["group2"],
            "meandiff": float(r["meandiff"]),
            "p_adj": float(r["p-adj"]),
            "reject": bool(r["reject"]),
            "decision": dec
        })

    # ----- Tukey Heatmap -----
    labels = sorted(df['city'].unique())
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for r in tukey_table:
        if r["reject"]:
            mat.loc[r["group1"], r["group2"]] = 1
            mat.loc[r["group2"], r["group1"]] = 1

    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(mat, annot=True, cmap='Reds', cbar=False, ax=ax)
    tukey_plot = fig_to_base64(fig)
    plt.close(fig)

    # ----- Plots -----
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=df, x='city', y=pol, ax=ax)
    plt.xticks(rotation=20)
    box_plot = fig_to_base64(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5))
    sns.violinplot(data=df, x='city', y=pol, ax=ax)
    plt.xticks(rotation=20)
    violin_plot = fig_to_base64(fig)
    plt.close(fig)

    return convert_np({
        "method": "One-way ANOVA",
        "pollutant": pol.upper(),
        "cities": req.cities,

        "H0": H0,
        "H1": H1,

        "fstat": float(fstat),
        "pvalue": float(pval),
        "p_fmt": p_fmt,

        "df_between": df_between,
        "df_within": df_within,

        "effect_size_eta2": float(eta_sq),

        "decision": decision,
        "tukey_table": tukey_table,
        "tukey_plot": tukey_plot,
        "box_plot": box_plot,
        "violin_plot": violin_plot
    })


# ================================================
# SECTION — CHI-SQUARE (City × AQI Category)
# ================================================

@app.post("/api/chisquare")
async def chi_api(req: ChiRequest):
    df = load_filtered(req.cities, req.year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")

    if 'aqi' not in df.columns:
        raise HTTPException(status_code=400, detail="AQI missing")

    # ----- Categorize AQI -----
    bins = [0,50,100,200,300,400,500,9999]
    labels = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe","Hazardous"]
    df['cat'] = pd.cut(df['aqi'], bins=bins, labels=labels, include_lowest=True)

    # ----- Contingency Table -----
    tab = pd.crosstab(df['city'], df['cat'])

    chi, p, dof, exp = chi2_contingency(tab)

    # ----- Hypotheses (A-style) -----
    H0 = "AQI category distribution is the same across selected cities"
    H1 = "AQI category distribution differs across selected cities"

    # ----- Decision (Style-2) -----
    decision = (
        "Reject Null Hypothesis (H₀) → Significant association"
        if p < 0.05 else
        "Accept Null Hypothesis (H₀) → No significant association"
    )

    # ----- p formatting -----
    p_fmt = "< 0.0001" if p < 1e-4 else f"{p:.5f}"

    # ----- Cramer's V (Effect Size) -----
    N = tab.values.sum()
    r, c = tab.shape
    cramers_v = np.sqrt(chi / (N * (min(r, c)-1))) if (N>0 and min(r,c)>1) else 0.0

    # ----- Heatmap (unchanged) -----
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(tab, annot=True, cmap='Blues', fmt='d', ax=ax)
    plt.title("City × AQI Category")
    heat = fig_to_base64(fig)
    plt.close(fig)

    return convert_np({
        "method": "Chi-square Independence Test",
        "cities": req.cities,

        "H0": H0,
        "H1": H1,

        "chisq": float(chi),
        "pvalue": float(p),
        "p_fmt": p_fmt,
        "df": dof,
        "effect_size_v": float(cramers_v),

        "decision": decision,
        "plot": heat,
        "table": tab.to_dict()
    })


# ================================================
# SECTION — SUPPORT
# ================================================

@app.get("/api/pollutants")
async def list_pollutants():
    return {"pollutants": ["pm25","pm10","o3","no2","so2","co"]}

@app.get("/api/cities")
async def list_cities():
    if 'df' not in DATA_CACHE:
        return {"cities":[]}
    return {"cities": sorted(DATA_CACHE['df']['city'].dropna().unique().tolist())}

@app.get("/")
async def root():
    return {"status":"AQI API running","version":"4.0.0"}
