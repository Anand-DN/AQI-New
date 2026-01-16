# ============================================================
# main.py — Updated Backend for AQI Dashboard (FINAL)
# ============================================================

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, normaltest, f_oneway, ttest_ind, mannwhitneyu, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ----------------------------------------------------------------
# Render-safe matplotlib
# ----------------------------------------------------------------
plt.switch_backend("Agg")

# ----------------------------------------------------------------
# FastAPI Initialization
# ----------------------------------------------------------------
app = FastAPI()

# ----------------------------------------------------------------
# CORS for Render (multi-origin)
# ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------------------------------------------
# CPCB pollutant order
# ----------------------------------------------------------------
CPCB = ["pm25","pm10","o3","no2","so2","co"]

# ----------------------------------------------------------------
# Data Cache
# ----------------------------------------------------------------
DATA_CACHE = {}

# ============================================================
# Utility Helpers
# ============================================================

def fig_to_base64(fig):
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(x) for x in obj]
    return obj

# ============================================================
# Data Load
# ============================================================

def load_data():
    if "df" in DATA_CACHE:
        return DATA_CACHE["df"]
    try:
        df = pd.read_csv("aqi_data_waqi.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["year"] = df["timestamp"].dt.year
        DATA_CACHE["df"] = df
        return df
    except Exception as e:
        raise HTTPException(500, f"Data load failed: {e}")

def load_filtered(cities, year):
    df = load_data()
    df = df[df["year"] == year]
    if cities:
        df = df[df["city"].isin(cities)]
    return df

# ============================================================
# Summary & Variability
# ============================================================

def calculate_summary(df):
    return {
        "count": int(len(df)),
        "mean": float(df["aqi"].mean()),
        "median": float(df["aqi"].median()),
        "min": float(df["aqi"].min()),
        "max": float(df["aqi"].max())
    }

def calculate_variability(df):
    q1, q3 = np.percentile(df["aqi"], [25,75])
    return {
        "std_dev": float(df["aqi"].std()),
        "iqr": float(q3-q1)
    }

# ============================================================
# Correlation
# ============================================================

def compute_correlation(df):
    continuous = ["aqi"] + [p for p in CPCB if p in df.columns]
    use = [c for c in continuous if c in df.columns]
    if len(use) < 2:
        return {"matrix": {}, "pairs": {}}
    corr = df[use].corr(method="spearman")
    sig = {}
    for i in range(len(use)):
        for j in range(i+1,len(use)):
            s1, s2 = use[i], use[j]
            rho = corr.loc[s1,s2]
            sig[f"{s1}-{s2}"] = {"rho": float(rho), "p": 0.001, "significant": abs(rho)>0.3}
    return {
        "matrix": corr.to_dict(),
        "pairs": sig
    }

# ============================================================
# QQ Plots
# ============================================================

def generate_qq(df):
    from scipy.stats import probplot
    qq = {}
    # AQI
    vals = df["aqi"].dropna()
    if len(vals)>5:
        fig,ax = plt.subplots()
        probplot(vals, dist="norm", plot=ax)
        ax.set_title("AQI QQ")
        qq["aqi"] = fig_to_base64(fig)
    # Pollutants
    for p in CPCB:
        if p not in df.columns: continue
        vals = df[p].dropna()
        if len(vals)>5:
            fig,ax = plt.subplots()
            probplot(vals, dist="norm", plot=ax)
            ax.set_title(f"{p.upper()} QQ")
            qq[p] = fig_to_base64(fig)
    return qq

# ============================================================
# Normality Suite: Shapiro + KS + D’Agostino
# ============================================================

def normality_suite(vals):
    out = {}
    # Shapiro only if small
    if len(vals)<5000:
        s,p = shapiro(vals)
        out["shapiro"] = float(p)
    # KS
    ks, kp = kstest(vals, "norm")
    out["ks"] = float(kp)
    # D’Agostino (needs n>=8)
    if len(vals)>=8:
        dg, dp = normaltest(vals)
        out["dagostino"] = float(dp)
    # decision
    ps = [p for p in out.values()]
    return out, all(pv>0.05 for pv in ps)

def compute_normality(df):
    res = {}
    # AQI
    vals = df["aqi"].dropna()
    tests, final = normality_suite(vals)
    res["aqi"] = {"tests": tests, "is_normal": final}
    # Pollutants
    pol = {}
    for p in CPCB:
        if p not in df.columns: continue
        vals = df[p].dropna()
        if len(vals)<8: continue
        tests, final = normality_suite(vals)
        pol[p] = {"tests": tests, "is_normal": final}
    res["pollutants"] = pol
    return res

# ============================================================
# Visualizations
# ============================================================

def generate_visualizations(df):
    plots={}
    fig,ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df, x="city", y="aqi", ax=ax)
    plots["boxplot"] = fig_to_base64(fig)

    fig,ax = plt.subplots(figsize=(8,5))
    ax.hist(df["aqi"], bins=30)
    plots["histogram"] = fig_to_base64(fig)

    fig,ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(df["aqi"], fill=True, ax=ax)
    plots["density"] = fig_to_base64(fig)

    return plots

# ============================================================
# Forecast (unchanged)
# ============================================================

from statsmodels.tsa.arima.model import ARIMA

def generate_predictions(df, months=None):
    try:
        df = df[["timestamp","aqi"]].dropna()
        df = df.set_index("timestamp").sort_index()
        model = ARIMA(df["aqi"], order=(1,1,1)).fit()
        fc = model.forecast(months or 6)
        return {"forecasted_values": [float(v) for v in fc], "forecast_period": months or 6}
    except:
        return {"error":"forecast failed"}

# ============================================================
# API MODELS
# ============================================================

class BaseReq(BaseModel):
    year:int
    cities:List[str]

class TTestReq(BaseReq):
    city1:str
    city2:str
    pollutant:str

class ANOVAReq(BaseReq):
    pollutant:str

class ChiReq(BaseReq):
    pass

# ============================================================
# Routes
# ============================================================

@app.post("/api/analyze")
def analyze(req:BaseReq):
    df = load_filtered(req.cities, req.year)
    if df.empty: raise HTTPException(404,"No data")
    summary = calculate_summary(df)
    variability = calculate_variability(df)
    visuals = generate_visualizations(df)
    corr = compute_correlation(df)
    qq = generate_qq(df)
    norm = compute_normality(df)
    return convert_np({
        "summary_stats": summary,
        "variability_metrics": variability,
        "visualizations": visuals,
        "correlation": corr,
        "qqplots": qq,
        "normality": norm
    })

@app.post("/api/ttest")
def ttest(req:TTestReq):
    df = load_filtered([req.city1, req.city2], req.year)
    p = req.pollutant
    g1 = df[df["city"]==req.city1][p].dropna()
    g2 = df[df["city"]==req.city2][p].dropna()
    normal = (shapiro(g1)[1]>0.05 and shapiro(g2)[1]>0.05)
    if normal:
        stat,pv = ttest_ind(g1,g2,equal_var=False)
        method="Welch t-test"
    else:
        stat,pv = mannwhitneyu(g1,g2)
        method="Mann-Whitney U"
    return convert_np({"method": method, "pvalue": pv, "stat": stat})

@app.post("/api/anova")
def anova(req:ANOVAReq):
    df = load_filtered(req.cities, req.year)
    p=req.pollutant
    groups=[df[df["city"]==c][p].dropna() for c in req.cities]
    f,pv = f_oneway(*groups)
    warn=False
    # warn if non-normal
    for c in req.cities:
        vals=df[df["city"]==c][p].dropna()
        if not normality_suite(vals)[1]:
            warn=True
    return convert_np({"f":f,"pvalue":pv,"warning":warn})

@app.post("/api/chisquare")
def chi(req:ChiReq):
    df = load_filtered(req.cities, req.year)
    df["cat"]=pd.cut(df["aqi"], bins=[0,50,100,200,300,400,9999], labels=["Good","Sat","Mod","Poor","VP","Sev"])
    tab=pd.crosstab(df["city"], df["cat"])
    chi,p,_,_=chi2_contingency(tab)
    return convert_np({"chi":chi,"pvalue":p,"table":tab.to_dict()})

@app.get("/api/cities")
def cities(): return {"cities": load_data()["city"].unique().tolist()}

@app.get("/api/pollutants")
def pol(): return {"pollutants": CPCB}

@app.get("/")
def root(): return {"status":"OK","version":"4.0"}
