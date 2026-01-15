# ==========================
# SECTION 1 — IMPORTS & SETUP
# ==========================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
from scipy.stats import shapiro, kstest, mannwhitneyu, ttest_ind, chi2_contingency
from statsmodels.stats.multitest import multipletests
from scipy.stats import probplot
import itertools
from scipy.stats import spearmanr
# Optional forecasting imports (kept from your version)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot

# ==========================
# INIT APP + CORS
# ==========================

load_dotenv()
app = FastAPI(title="AQI Analysis API", version="2.0.0")

origins = [
    "https://aqi-new-2.onrender.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,  # IMPORTANT (or add cookie rules)
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ==========================
# GLOBAL CACHE
# ==========================

DATA_CACHE = {}

@app.on_event("startup")
async def preload_data():
    global DATA_CACHE
    try:
        if os.path.exists('data/aqi_data_waqi.csv'):
            df = pd.read_csv('data/aqi_data_waqi.csv')
            DATA_CACHE['df'] = df
            print(f"✓ Preloaded {len(df)} records from aqi_data_waqi.csv")
        elif os.path.exists('data/aqi_data_openaq.csv'):
            df = pd.read_csv('data/aqi_data_openaq.csv')
            DATA_CACHE['df'] = df
            print(f"✓ Preloaded {len(df)} records from aqi_data_openaq.csv")
        else:
            print("⚠ No AQI CSV found at startup.")
    except Exception as e:
        print(f"Error during startup: {e}")

# ==========================
# REQUEST MODELS
# ==========================

class AnalysisRequest(BaseModel):
    cities: List[str]
    year: int
    predict_months: Optional[int] = None
    predict_year: Optional[int] = None

class TTestRequest(BaseModel):
    city1: str
    city2: str
    pollutant: str
    year: int

class ANOVARequest(BaseModel):
    cities: List[str]
    pollutant: str
    year: int

class ChiRequest(BaseModel):
    cities: List[str]
    year: int

# ==========================
# BASIC ROUTES
# ==========================

@app.get("/")
async def root():
    return {"message": "AQI Analysis API Running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/cities")
async def get_cities():
    try:
        df = DATA_CACHE.get('df')
        if df is not None and 'city' in df.columns:
            return {"cities": sorted(df['city'].dropna().unique().tolist())}
        return {"cities": []}
    except:
        return {"cities": []}

# ==========================
# SECTION 2 — UTILITY HELPERS
# ==========================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img


def convert_np(o):
    """Recursively convert numpy types to python primitives for JSON."""
    import numpy as np
    if isinstance(o, dict):
        return {k: convert_np(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [convert_np(i) for i in o]
    elif isinstance(o, np.generic):
        return o.item()
    return o


def fill_missing_with_mean(df, pollutant):
    """Mean-impute missing pollutant values."""
    if pollutant in df.columns:
        df[pollutant] = df[pollutant].fillna(df[pollutant].mean())
    return df


# ==========================
# CPCB CATEGORY MAPPING
# ==========================

def pollutant_to_cpcb_category(pollutant, value):
    """CPCB pollutant category mapping using value in µg/m³."""
    if pd.isna(value):
        return None

    # PM2.5
    if pollutant.lower() == "pm25":
        if value <= 30: return "Good"
        elif value <= 60: return "Satisfactory"
        elif value <= 90: return "Moderate"
        elif value <= 120: return "Poor"
        elif value <= 250: return "Very Poor"
        else: return "Severe"

    # PM10
    if pollutant.lower() == "pm10":
        if value <= 50: return "Good"
        elif value <= 100: return "Satisfactory"
        elif value <= 250: return "Moderate"
        elif value <= 350: return "Poor"
        elif value <= 430: return "Very Poor"
        else: return "Severe"

    # NO2
    if pollutant.lower() == "no2":
        if value <= 40: return "Good"
        elif value <= 80: return "Satisfactory"
        elif value <= 180: return "Moderate"
        elif value <= 280: return "Poor"
        elif value <= 400: return "Very Poor"
        else: return "Severe"

    # SO2
    if pollutant.lower() == "so2":
        if value <= 40: return "Good"
        elif value <= 80: return "Satisfactory"
        elif value <= 380: return "Moderate"
        elif value <= 800: return "Poor"
        elif value <= 1600: return "Very Poor"
        else: return "Severe"

    # O3
    if pollutant.lower() == "o3":
        if value <= 50: return "Good"
        elif value <= 100: return "Satisfactory"
        elif value <= 168: return "Moderate"
        elif value <= 208: return "Poor"
        elif value <= 748: return "Very Poor"
        else: return "Severe"

    # CO (mg/m³)
    if pollutant.lower() == "co":
        if value <= 1: return "Good"
        elif value <= 2: return "Satisfactory"
        elif value <= 10: return "Moderate"
        elif value <= 17: return "Poor"
        elif value <= 34: return "Very Poor"
        else: return "Severe"

    return None


def aqi_to_category(aqi):
    """CPCB AQI category mapping."""
    if pd.isna(aqi):
        return None
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"


# ==========================
# EFFECT SIZE METRICS
# ==========================

def cohens_d(x, y):
    """Cohen's d for T-test numeric effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return None
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled == 0: return 0
    return (np.mean(x) - np.mean(y)) / pooled


def cramers_v(contingency):
    """Cramér’s V for categorical associations."""
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n = contingency.sum().sum()
    r, c = contingency.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


# ==========================
# CATEGORY DISTRIBUTION HELPERS
# ==========================

def category_counts(df, col, categories=None):
    """Return raw counts for target categories."""
    counts = df[col].value_counts()
    if categories:
        return {cat: int(counts.get(cat, 0)) for cat in categories}
    return counts.to_dict()


def category_percent(df, col, categories=None):
    """Return percent distribution."""
    total = len(df)
    if total == 0:
        return {cat: 0 for cat in (categories or [])}
    counts = df[col].value_counts() / total * 100
    if categories:
        return {cat: round(counts.get(cat, 0.0), 2) for cat in categories}
    return {k: round(v,2) for k,v in counts.items()}


# ==========================
# DATA FILTERING + IMPUTATION
# ==========================

def load_filtered(cities, year):
    """Filter global df by selected cities + year."""
    if 'df' not in DATA_CACHE:
        return pd.DataFrame()

    df = DATA_CACHE['df'].copy()

    if 'city' not in df.columns:
        return pd.DataFrame()

    df = df[df['city'].isin(cities)]

    # timestamp / date handling
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['year'] = df['timestamp'].dt.year
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year

    df = df[df['year'] == year]
    return df.reset_index(drop=True)

# ==========================
# SECTION 3 — NORMALITY & CORRELATION
# ==========================

def compute_normality(values):
    """Return Shapiro + KS + QQ plot + decision."""
    values = pd.Series(values).dropna()
    if len(values) < 10:
        return {
            "error": "Insufficient data for normality tests",
            "count": len(values)
        }

    # ---- Shapiro-Wilk Test ----
    sh_stat, sh_p = shapiro(values)

    # ---- KS Test vs Normal Distribution ----
    mean, std = values.mean(), values.std()
    if std > 0:
        ks_stat, ks_p = kstest(values, 'norm', args=(mean, std))
    else:
        ks_stat, ks_p = None, None

    # ---- QQ Plot ----
    fig = plt.figure(figsize=(6, 6))
    probplot(values, dist="norm", plot=plt)
    plt.title("QQ Plot (Normality Check)")
    qq_base64 = fig_to_base64(fig)

    # ---- Decision Logic ----
    normality = "non-normal"
    if sh_p > 0.05 and (ks_p is None or ks_p > 0.05):
        normality = "normal"
    elif sh_p > 0.05 or (ks_p and ks_p > 0.05):
        normality = "borderline"

    return convert_np({
        "count": len(values),
        "shapiro": {"stat": sh_stat, "p": sh_p},
        "ks": {"stat": ks_stat, "p": ks_p},
        "qqplot": qq_base64,
        "decision": normality
    })




def compute_correlation(df):
    pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    results = {}

    for p in pollutants:
        if p not in df.columns:
            continue
        
        x = df["aqi"].astype(float)
        y = df[p].astype(float)

        corr, pval = spearmanr(x, y, nan_policy="omit")

        results[f"aqi_vs_{p}"] = {
            "corr": float(corr),
            "p": float(pval),
            "method": "spearman",
            "significant": bool(pval < 0.05)
        }
    
    return results


# ==========================
# SECTION 4 — T-TEST MODULE
# ==========================

@app.post("/api/ttest")
async def ttest_api(req: TTestRequest):
    df = load_filtered([req.city1, req.city2], req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    pollutant = req.pollutant.lower()

    if pollutant not in df.columns:
        raise HTTPException(status_code=400, detail=f"Pollutant '{pollutant}' not available")

    # ---- Filter by city ----
    df = fill_missing_with_mean(df, pollutant)
    c1 = df[df['city'] == req.city1][pollutant].dropna()
    c2 = df[df['city'] == req.city2][pollutant].dropna()

    if len(c1) < 5 or len(c2) < 5:
        raise HTTPException(status_code=400, detail="Insufficient samples for T-Test")

    # ---- NORMALITY CHECK ----
    from scipy.stats import shapiro, mannwhitneyu, ttest_ind

    sh1, p1 = shapiro(c1) if len(c1) < 5000 else (None, 0.01)
    sh2, p2 = shapiro(c2) if len(c2) < 5000 else (None, 0.01)

    normal = (p1 > 0.05 and p2 > 0.05)

    # ---- TEST SELECTION ----
    if normal:
        stat, p = ttest_ind(c1, c2, equal_var=False)
        method = "t-test (independent)"
    else:
        stat, p = mannwhitneyu(c1, c2, alternative='two-sided')
        method = "Mann-Whitney U (non-parametric)"

    # ---- EFFECT SIZE ----
    effect = cohens_d(c1.values, c2.values)

    # ---- CATEGORY MAPPING ----
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

    df['cat'] = df[pollutant].apply(lambda v: pollutant_to_cpcb_category(pollutant, v))
    c1_cats = category_counts(df[df['city'] == req.city1], 'cat', cats)
    c2_cats = category_counts(df[df['city'] == req.city2], 'cat', cats)

    # ---- STACKED BAR RAW COUNTS ----
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom1, bottom2 = 0, 0

    for cat in cats:
        val1 = c1_cats.get(cat, 0)
        val2 = c2_cats.get(cat, 0)
        ax.bar(req.city1, val1, bottom=bottom1, label=cat if bottom1==0 else "")
        ax.bar(req.city2, val2, bottom=bottom2)
        bottom1 += val1
        bottom2 += val2

    ax.set_title(f"{pollutant.upper()} Category Distribution (Raw Counts)")
    ax.set_ylabel("Count")
    ax.legend()
    stacked_b64 = fig_to_base64(fig)

    # ---- DENSITY PLOT ----
    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(c1, fill=True, label=req.city1)
    sns.kdeplot(c2, fill=True, label=req.city2)
    ax.set_title(f"{pollutant.upper()} Density Plot")
    ax.legend()
    density_b64 = fig_to_base64(fig)

    # ---- BOX PLOT ----
    fig, ax = plt.subplots(figsize=(6,5))
    ax.boxplot([c1, c2], labels=[req.city1, req.city2])
    ax.set_title(f"{pollutant.upper()} Box Plot")
    box_b64 = fig_to_base64(fig)

    return convert_np({
        "method": method,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect,
        "normality": {
            req.city1: {"p": p1},
            req.city2: {"p": p2}
        },
        "decision": "reject Null Hypothesis H0" if p < 0.05 else "Accept Null Hypothesis H0",
        "stacked_bar": stacked_b64,
        "density_plot": density_b64,
        "box_plot": box_b64,
        "categories_raw": {req.city1: c1_cats, req.city2: c2_cats}
    })

# ==========================
# SECTION 5 — CATEGORICAL ANOVA MODULE
# ==========================

@app.post("/api/anova")
async def anova_api(req: ANOVARequest):
    df = load_filtered(req.cities, req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    pollutant = req.pollutant.lower()

    if pollutant not in df.columns:
        raise HTTPException(status_code=400, detail=f"Pollutant '{pollutant}' not available")

    df = fill_missing_with_mean(df, pollutant)

    # ---- CPCB category mapping ----
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    df['cat'] = df[pollutant].apply(lambda v: pollutant_to_cpcb_category(pollutant, v))

    # ---- Build contingency table for Chi k×m ----
    contingency = pd.crosstab(df['city'], df['cat']).reindex(columns=cats, fill_value=0)

    # ---- Overall Chi-square test ----
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    effect = cramers_v(contingency)

    # ---- Post-hoc Pairwise Chi-square ----
    pairs = list(itertools.combinations(req.cities, 2))
    posthoc = []

    for c1, c2 in pairs:
        sub = contingency.loc[[c1, c2]]
        chi2_, p_, _, _ = chi2_contingency(sub, correction=False)
        posthoc.append({"pair": f"{c1} vs {c2}", "p": p_})

    # ---- Multiple correction Holm-Bonferroni ----
    if len(posthoc) > 1:
        pvals = [x['p'] for x in posthoc]
        _, pvals_corr, _, _ = multipletests(pvals, method="holm")
        for i, v in enumerate(pvals_corr):
            posthoc[i]['p_corrected'] = float(v)
            posthoc[i]['significant'] = bool(v < 0.05)
    else:
        for x in posthoc:
            x['p_corrected'] = x['p']
            x['significant'] = x['p'] < 0.05

    # ---- Stacked PERCENT Bar Plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom_map = {city: 0 for city in req.cities}

    perc_dict = {}
    for city in req.cities:
        sub = df[df['city'] == city]
        perc = category_percent(sub, 'cat', cats)
        perc_dict[city] = perc

    for cat in cats:
        vals = [perc_dict[city][cat] for city in req.cities]
        ax.bar(req.cities, vals, bottom=[bottom_map[c] for c in req.cities], label=cat)
        for i, city in enumerate(req.cities):
            bottom_map[city] += vals[i]

    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"{pollutant.upper()} Category Distribution by City (ANOVA - Categorical)")
    ax.legend(title="CPCB Category", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=20)
    stacked_b64 = fig_to_base64(fig)

    # ---- Post-hoc Heatmap ----
    size = len(req.cities)
    heat = np.ones((size, size)) * np.nan

    for entry in posthoc:
        c1, c2 = entry['pair'].split(" vs ")
        i, j = req.cities.index(c1), req.cities.index(c2)
        heat[i, j] = entry['p_corrected']
        heat[j, i] = entry['p_corrected']

    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(heat, annot=True, cmap="viridis_r", xticklabels=req.cities, yticklabels=req.cities)
    plt.title("Post-hoc Pairwise (Holm-corrected P-values)")
    heatmap_b64 = fig_to_base64(fig)

    return convert_np({
        "method": "Chi-square k×m (categorical ANOVA)",
        "chi_square": chi2,
        "dof": dof,
        "p_value": p,
        "effect_size": effect,
        "decision": "reject Null Hypothesis H0" if p < 0.05 else "Accept Null Hypothesis H0",
        "contingency": contingency.to_dict(),
        "percent_distribution": perc_dict,
        "stacked_percent": stacked_b64,
        "posthoc": posthoc,
        "posthoc_heatmap": heatmap_b64
    })

# ==========================
# SECTION 6 — CHI-SQUARE (AQI × CITY)
# ==========================

@app.post("/api/chisquare")
async def chi_api(req: ChiRequest):
    df = load_filtered(req.cities, req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    if 'aqi' not in df.columns:
        raise HTTPException(status_code=400, detail="AQI column not found in dataset")

    # ---- CPCB AQI Category ----
    df['aqi_cat'] = df['aqi'].apply(aqi_to_category)

    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

    # ---- Build contingency table ----
    contingency = pd.crosstab(df['city'], df['aqi_cat']).reindex(columns=cats, fill_value=0)

    # ---- Chi-square test ----
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    effect = cramers_v(contingency)

    # ---- Heatmap ----
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="YlOrRd", cbar=True)
    ax.set_title("City × AQI Category (Counts)")
    ax.set_ylabel("City")
    ax.set_xlabel("AQI Category")
    heatmap_b64 = fig_to_base64(fig)

    return convert_np({
        "method": "Chi-square (City × AQI Category)",
        "chi_square": chi2,
        "p_value": p,
        "dof": dof,
        "effect_size": effect,
        "decision": "reject Null Hypothesis H0" if p < 0.05 else "Accept Null Hypothesis H0",
        "categories": cats,
        "contingency": contingency.to_dict(),
        "heatmap": heatmap_b64
    })

# ==========================
# SECTION 7 — SUMMARY & VISUALIZATION & FORECASTING
# ==========================

def calculate_summary_statistics(df):
    value_col = 'aqi' if 'aqi' in df.columns else 'value'
    values = df[value_col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) == 0:
        return {"count": 0}

    return convert_np({
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quartiles": {
            "Q1": float(values.quantile(0.25)),
            "Q2": float(values.quantile(0.50)),
            "Q3": float(values.quantile(0.75)),
        }
    })


def calculate_variability_metrics(df):
    value_col = 'aqi' if 'aqi' in df.columns else 'value'
    values = df[value_col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 2:
        return convert_np({
            "std_dev": None,
            "iqr": None,
            "variance": None,
            "cv": None
        })

    mean = values.mean()
    return convert_np({
        "std_dev": float(values.std()),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "variance": float(values.var()),
        "cv": float((values.std() / mean) * 100) if mean != 0 else None
    })

def generate_qq(df):
    pollutants = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
    qq_plots = {}

    for p in pollutants:
        if p not in df.columns:
            continue
        
        fig = plt.figure(figsize=(5,4))
        stats.probplot(df[p].dropna(), dist="norm", plot=plt)
        plt.title(f"QQ Plot - {p}")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        qq_plots[p] = base64.b64encode(buf.read()).decode("utf-8")

    return qq_plots


def generate_visualizations(df):
    plots = {}
    value_col = 'aqi' if 'aqi' in df.columns else 'value'
    df = df.dropna(subset=[value_col])

    if df.empty:
        return {}

    # ---- Box Plot ----
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=df, x='city', y=value_col, ax=ax)
    plt.title("AQI Distribution by City")
    plt.xticks(rotation=25)
    plots['boxplot'] = fig_to_base64(fig)

    # ---- Histogram ----
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df[value_col], bins=30, color='skyblue', edgecolor='black')
    plt.title("AQI Histogram")
    plots['histogram'] = fig_to_base64(fig)

    # ---- Density ----
    fig, ax = plt.subplots(figsize=(8,5))
    for city in df['city'].unique():
        sns.kdeplot(df[df['city']==city][value_col], fill=True, label=city, ax=ax)
    plt.title("AQI Density")
    plt.legend()
    plots['density'] = fig_to_base64(fig)

    return plots


# ==========================
# FORECASTING (ARIMA + FALLBACK)
# ==========================

def generate_predictions(df, predict_months=None, predict_year=None):
    try:
        value_col = 'aqi' if 'aqi' in df.columns else 'value'
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'

        if date_col not in df.columns:
            return {"error": "No timestamp column for forecasting"}

        df = df[[date_col, value_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        if len(df) < 10:
            return {"error": f"Insufficient data ({len(df)}) for forecasting"}

        df = df.groupby(date_col)[value_col].mean()
        periods = predict_months or (12 if predict_year else 6)

        model = ARIMA(df, order=(1,1,1)).fit()
        forecast = model.forecast(periods)

        return convert_np({
            "forecasted_values": forecast.tolist(),
            "method": "ARIMA (1,1,1)"
        })

    except Exception as e:
        return {"error": str(e)}


# ==========================
# AI SUMMARY (NARRATIVE)
# ==========================

def generate_ai_summary(df, predictions=None):
    stats = calculate_summary_statistics(df)
    var = calculate_variability_metrics(df)

    if stats.get("count", 0) == 0:
        return "No data available."

    msg = f"""
Summary for {stats['count']} records:

Mean AQI: {stats['mean']:.2f}
Median AQI: {stats['median']:.2f}
Range: {stats['min']} - {stats['max']}
Std Dev: {var.get('std_dev')}
IQR: {var.get('iqr')}
"""

    if predictions and not predictions.get("error"):
        msg += "\nForecasting indicates possible AQI trend shifts."

    return msg.strip()

# ==========================
# SECTION 8 — ANALYZE (MAIN ROUTE)
# ==========================

@app.post("/api/analyze")
async def run_analysis(req: AnalysisRequest):
    df = load_filtered(req.cities, req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {req.cities} in {req.year}")

    # ---- SUMMARY ----
    summary = calculate_summary_statistics(df)
    variability = calculate_variability_metrics(df)

    # ---- VISUALS ----
    visuals = generate_visualizations(df)

    # ---- CORRELATION ----
    correlation = compute_correlation(df)
    qqplot=generate_qq(df)

    # ---- NORMALITY (AQI only) ----
    if 'aqi' in df.columns:
        aqi_series = df['aqi'].dropna()

    # Shapiro Limit Handling
    sample = aqi_series
    if len(sample) > 5000:
        sample = sample.sample(5000)

    try:
        stat, pval = shapiro(sample)
        normality = {
            "statistic": float(stat),
            "pvalue": float(pval),
            "decision": "not_normal" if pval < 0.05 else "normal"
        }
    except Exception as e:
        normality = {"error": str(e)}
    else:
        normality = {"error": "AQI missing for normality"}

# ---- QQ Plot (AQI) ----
    qq_base64 = None
    if 'aqi' in df.columns:
        try:
            fig = plt.figure(figsize=(6, 6))
            qqplot(df['aqi'].dropna(), line='s', ax=plt.gca())
            qq_base64 = fig_to_base64(fig)
        except Exception as e:
            qq_base64 = None


    # ---- FORECASTING (Optional) ----
    predictions = None
    if req.predict_months or req.predict_year:
        predictions = generate_predictions(
            df,
            predict_months=req.predict_months,
            predict_year=req.predict_year
        )

    # ---- AI SUMMARY ----
    ai_summary = generate_ai_summary(df, predictions)

    return convert_np({
         "summary_stats": summary,
         "variability_metrics": variability,
         "visualizations": visuals,
         "correlation_pairs": correlation,
         "qqplots": qqplot,
         "normality_test": normality,
         "predictions": predictions,
         "ai_summary": ai_summary
})


@app.get("/api/pollutants")
async def get_pollutants():
    pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    df = DATA_CACHE.get('df')
    if df is None:
        return {"pollutants": pollutant_cols}

    available = [p for p in pollutant_cols if p in df.columns]
    return {"pollutants": available or pollutant_cols}


# ==========================
# SECTION 9 — ENTRYPOINT
# ==========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )







