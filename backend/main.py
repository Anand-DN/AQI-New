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
    spearmanr, shapiro, ttest_ind,
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
        FRONTEND,
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
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

def generate_qq(df):
    pollutants = ["aqi","pm25","pm10","o3","no2","so2","co"]
    qq = {}

    for p in pollutants:
        if p not in df.columns:
            continue
        vals = df[p].dropna()
        if len(vals)==0: continue

        try:
            fig = plt.figure(figsize=(4.5,4))
            qqplot(vals, line='s')
            plt.title(f"QQ Plot — {p.upper()}")
            qq[p] = fig_to_base64(fig)
        except Exception:
            qq[p] = None

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
    if len(data)<5:
        return {"matrix":None,"pairs":None}

    # Matrix
    corr_mat, p_mat = {}, {}
    for i in use:
        corr_mat[i] = {}
        p_mat[i] = {}
        for j in use:
            rho, p = spearmanr(data[i], data[j], nan_policy="omit")
            corr_mat[i][j] = float(rho)
            p_mat[i][j] = float(p)

    # Flatten pairs
    pairs = {}
    for a,b in itertools.combinations(use,2):
        rho, p = spearmanr(data[a], data[b], nan_policy="omit")
        pairs[f"{a}-{b}"] = {
            "rho": float(rho),
            "p": float(p),
            "significant": bool(p<0.05),
            "method": "spearman"
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
# FORECAST SECTION (OPTIONAL - unchanged)
# ================================================

from statsmodels.tsa.arima.model import ARIMA

def generate_predictions(df, predict_months=None, predict_year=None):
    try:
        val = 'aqi' if 'aqi' in df.columns else 'value'
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        if date_col not in df.columns:
            return {"error":"no date column"}

        df = df[[date_col,val]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        if len(df)<10:
            return {"error":f"insufficient data {len(df)}"}

        df = df.groupby(date_col)[val].mean()
        periods = predict_months or (12 if predict_year else 6)

        model = ARIMA(df, order=(1,1,1)).fit()
        forecast = model.forecast(periods)

        return {
            "forecasted_values": forecast.tolist(),
            "method": "ARIMA(1,1,1)",
            "forecast_period": periods
        }

    except Exception as e:
        return {"error":str(e)}

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

    # Normality detection (Shapiro)
    sh1, p1 = shapiro(c1) if len(c1)<5000 else (None, 0.01)
    sh2, p2 = shapiro(c2) if len(c2)<5000 else (None, 0.01)

    normal = (p1>0.05 and p2>0.05)

    if normal:
        stat, p = ttest_ind(c1, c2, equal_var=False)
        method = "t-test (Welch)"
    else:
        stat, p = mannwhitneyu(c1, c2, alternative="two-sided")
        method = "Mann-Whitney U"

    effect = cohens_d(c1.values, c2.values)

    # CPCB category stacking
    cats = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    df['cat'] = df[pol].apply(lambda v: pollutant_to_cpcb_category(pol,v))
    c1_cats = category_counts(df[df['city']==req.city1], 'cat', cats)
    c2_cats = category_counts(df[df['city']==req.city2], 'cat', cats)

    # ========== STACKED BAR (Counts) ==========
    fig, ax = plt.subplots(figsize=(8,5))
    bottom1, bottom2 = 0, 0
    for cat in cats:
        v1, v2 = c1_cats.get(cat,0), c2_cats.get(cat,0)
        ax.bar(req.city1, v1, bottom=bottom1, label=cat if bottom1==0 else "")
        ax.bar(req.city2, v2, bottom=bottom2)
        bottom1 += v1; bottom2 += v2
    ax.set_title(f"{pol.upper()} Category Counts")
    stacked_bar = fig_to_base64(fig)

    # ========== STACKED AREA (Smooth) ==========
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

    # ========== DENSITY ==========
    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(c1, fill=True, label=req.city1)
    sns.kdeplot(c2, fill=True, label=req.city2)
    ax.set_title(f"{pol.upper()} Density")
    ax.legend()
    density = fig_to_base64(fig)

    # ========== BOX ==========
    fig, ax = plt.subplots(figsize=(6,5))
    ax.boxplot([c1, c2], labels=[req.city1, req.city2])
    ax.set_title(f"{pol.upper()} Box Plot")
    box = fig_to_base64(fig)

    return convert_np({
        "method": method,
        "statistic": stat,
        "pvalue": p,
        "effect_size": effect,
        "normality": {req.city1:p1, req.city2:p2},
        "decision": "reject H0" if p<0.05 else "fail to reject H0",

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

    # Build sample groups
    groups = [df[df['city']==c][pol] for c in req.cities]
    sizes = [len(g) for g in groups if len(g)>2]
    if len(sizes) < 2:
        raise HTTPException(status_code=400, detail="Insufficient samples")

    # --------- ANOVA ---------
    fstat, pval = f_oneway(*groups)

    # --------- Tukey HSD ---------
    tukey_res = pairwise_tukeyhsd(df[pol], df['city'], alpha=0.05)

    # Tukey table conversion
    tukey_table = []
    for i in range(len(tukey_res.meandiffs)):
        tukey_table.append({
            "group1": str(tukey_res._groupsunique[tukey_res._multicomp.pairindices[i][0]]),
            "group2": str(tukey_res._groupsunique[tukey_res._multicomp.pairindices[i][1]]),
            "meandiff": float(tukey_res.meandiffs[i]),
            "p_adj": float(tukey_res.pvalues[i]),
            "reject": bool(tukey_res.reject[i])
        })

    # --------- Box Plot ---------
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=df, x='city', y=pol, ax=ax)
    plt.title(f"ANOVA Boxplot — {pol.upper()}")
    plt.xticks(rotation=20)
    box_plot = fig_to_base64(fig)

    # --------- Violin Plot ---------
    fig, ax = plt.subplots(figsize=(9,5))
    sns.violinplot(data=df, x='city', y=pol, ax=ax)
    plt.title(f"ANOVA Violin — {pol.upper()}")
    plt.xticks(rotation=20)
    violin_plot = fig_to_base64(fig)

    # --------- Tukey Heatmap (Binary significance) ---------
    uniq = list(df['city'].unique())
    uniq.sort()
    mat = pd.DataFrame(0, index=uniq, columns=uniq)
    for row in tukey_table:
        g1, g2, rej = row['group1'], row['group2'], row['reject']
        mat.loc[g1,g2] = 1 if rej else 0
        mat.loc[g2,g1] = 1 if rej else 0

    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(mat, annot=True, cmap='Reds', cbar=False)
    plt.title("Tukey Post-Hoc (Significance)")
    tukey_heatmap = fig_to_base64(fig)

    return convert_np({
        "fstat": fstat,
        "pvalue": pval,
        "decision": "reject H0" if pval<0.05 else "fail to reject H0",

        "tukey_table": tukey_table,
        "tukey_plot": tukey_heatmap,

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

    # Categorize AQI
    bins = [0,50,100,200,300,400,500,9999]
    labels = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe","Hazardous"]
    df['cat'] = pd.cut(df['aqi'], bins=bins, labels=labels, include_lowest=True)

    tab = pd.crosstab(df['city'], df['cat'])

    from scipy.stats import chi2_contingency
    chi, p, dof, exp = chi2_contingency(tab)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(tab, annot=True, cmap='Blues', fmt='d')
    plt.title("City × AQI Category")
    heat = fig_to_base64(fig)

    return convert_np({
        "chisq": chi,
        "pvalue": p,
        "decision": "reject H0" if p<0.05 else "fail to reject H0",
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
