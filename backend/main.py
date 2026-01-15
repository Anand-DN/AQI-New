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
from io import BytesIO
import base64
import json

# stats
from scipy.stats import shapiro, kstest, mannwhitneyu, ttest_ind, chi2_contingency, spearmanr, probplot
from scipy import stats
import itertools

# Optional forecasting imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.multitest import multipletests

# ==========================
# INIT APP + CORS
# ==========================

load_dotenv()
app = FastAPI(title="AQI Analysis API", version="3.0.0")

origins = [
    "https://aqi-new-2.onrender.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
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
            print("⚠ No AQI dataset found at startup")
    except Exception as e:
        print(f"Startup error: {e}")

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
    return {"message": "AQI API Running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/cities")
async def get_cities():
    try:
        df = DATA_CACHE.get('df')
        if df is None: return {"cities":[]}
        if 'city' not in df.columns: return {"cities":[]}
        return {"cities": sorted(df['city'].dropna().unique().tolist())}
    except:
        return {"cities":[]}

# ==========================
# UTILITY HELPERS
# ==========================

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img

def convert_np(o):
    """Convert numpy to python primitives recursively for JSON."""
    if isinstance(o, dict):
        return {k: convert_np(v) for k,v in o.items()}
    if isinstance(o, list):
        return [convert_np(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o

def fill_missing_with_mean(df, pollutant):
    if pollutant in df.columns:
        df[pollutant] = df[pollutant].fillna(df[pollutant].mean())
    return df

# ==========================
# CPCB CATEGORY MAPPING
# ==========================

def pollutant_to_cpcb_category(pollutant, value):
    if pd.isna(value): return None
    p = pollutant.lower()

    if p == "pm25":
        if value<=30: return "Good"
        elif value<=60: return "Satisfactory"
        elif value<=90: return "Moderate"
        elif value<=120:return "Poor"
        elif value<=250:return "Very Poor"
        else:return "Severe"

    if p == "pm10":
        if value<=50: return "Good"
        elif value<=100:return "Satisfactory"
        elif value<=250:return "Moderate"
        elif value<=350:return "Poor"
        elif value<=430:return "Very Poor"
        else:return "Severe"

    if p == "no2":
        if value<=40:return "Good"
        elif value<=80:return "Satisfactory"
        elif value<=180:return "Moderate"
        elif value<=280:return "Poor"
        elif value<=400:return "Very Poor"
        else:return "Severe"

    if p == "so2":
        if value<=40:return "Good"
        elif value<=80:return "Satisfactory"
        elif value<=380:return "Moderate"
        elif value<=800:return "Poor"
        elif value<=1600:return "Very Poor"
        else:return "Severe"

    if p == "o3":
        if value<=50:return "Good"
        elif value<=100:return "Satisfactory"
        elif value<=168:return "Moderate"
        elif value<=208:return "Poor"
        elif value<=748:return "Very Poor"
        else:return "Severe"

    if p == "co":
        if value<=1:return "Good"
        elif value<=2:return "Satisfactory"
        elif value<=10:return "Moderate"
        elif value<=17:return "Poor"
        elif value<=34:return "Very Poor"
        else:return "Severe"

    return None

def aqi_to_category(aqi):
    if pd.isna(aqi): return None
    if aqi<=50: return "Good"
    elif aqi<=100: return "Satisfactory"
    elif aqi<=200: return "Moderate"
    elif aqi<=300: return "Poor"
    elif aqi<=400: return "Very Poor"
    else: return "Severe"

# ==========================
# EFFECT SIZE + CATEGORY DISTRIBUTION
# ==========================

def cohens_d(x,y):
    nx,ny=len(x),len(y)
    if nx<2 or ny<2:return None
    pooled=np.sqrt(((nx-1)*np.var(x,ddof=1)+(ny-1)*np.var(y,ddof=1))/(nx+ny-2))
    if pooled==0:return 0
    return (np.mean(x)-np.mean(y))/pooled

def cramers_v(contingency):
    chi2,_,_,_=chi2_contingency(contingency,correction=False)
    n=contingency.sum().sum()
    r,c=contingency.shape
    return np.sqrt(chi2/(n*(min(r,c)-1)))

def category_counts(df,col,cats=None):
    vc=df[col].value_counts()
    if cats: return {c:int(vc.get(c,0)) for c in cats}
    return vc.to_dict()

def category_percent(df,col,cats=None):
    n=len(df)
    if n==0: return {c:0 for c in (cats or [])}
    vc=df[col].value_counts()/n*100
    if cats: return {c:round(vc.get(c,0.0),2) for c in cats}
    return {k:round(v,2) for k,v in vc.items()}

# ==========================
# LOAD + FILTER
# ==========================

def load_filtered(cities,year):
    if 'df' not in DATA_CACHE: return pd.DataFrame()
    df=DATA_CACHE['df'].copy()

    if 'city' not in df.columns: return pd.DataFrame()
    df=df[df['city'].isin(cities)]

    if 'timestamp' in df.columns:
        df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
        df['year']=df['timestamp'].dt.year
    elif 'date' in df.columns:
        df['date']=pd.to_datetime(df['date'],errors='coerce')
        df['year']=df['date'].dt.year

    df=df[df['year']==year]
    return df.reset_index(drop=True)

# ==========================
# CORRELATION (STYLE C HEATMAP-LIKE)
# ==========================

def compute_correlation(df):
    pollutants=["pm25","pm10","o3","no2","so2","co"]
    results={}
    x=df["aqi"].astype(float)

    for p in pollutants:
        if p not in df.columns: continue
        y=df[p].astype(float)
        corr,pval=spearmanr(x,y,nan_policy="omit")

        results[p.upper()] = {
            "corr": float(corr),
            "p": float(pval),
            "significant": bool(pval<0.05),
            "method": "spearman"
        }

    return results


# ==========================
# SECTION — T-TEST MODULE
# ==========================

@app.post("/api/ttest")
async def ttest_api(req: TTestRequest):
    df = load_filtered([req.city1, req.city2], req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    pollutant = req.pollutant.lower()
    if pollutant not in df.columns:
        raise HTTPException(status_code=400, detail=f"Pollutant '{pollutant}' not found")

    df = fill_missing_with_mean(df, pollutant)
    c1 = df[df['city']==req.city1][pollutant].dropna()
    c2 = df[df['city']==req.city2][pollutant].dropna()

    if len(c1)<5 or len(c2)<5:
        raise HTTPException(status_code=400, detail="Insufficient samples for T-Test")

    sh1, p1 = shapiro(c1) if len(c1)<5000 else (None, 0.01)
    sh2, p2 = shapiro(c2) if len(c2)<5000 else (None, 0.01)
    normal = (p1>0.05 and p2>0.05)

    if normal:
        stat, p = ttest_ind(c1, c2, equal_var=False)
        method = "t-test"
    else:
        stat, p = mannwhitneyu(c1, c2, alternative="two-sided")
        method = "Mann-Whitney U"

    effect = cohens_d(c1.values, c2.values)

    cats=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    df['cat'] = df[pollutant].apply(lambda v: pollutant_to_cpcb_category(pollutant,v))
    c1_cats = category_counts(df[df['city']==req.city1], 'cat', cats)
    c2_cats = category_counts(df[df['city']==req.city2], 'cat', cats)

    fig, ax = plt.subplots(figsize=(8,5))
    bottom1, bottom2 = 0,0
    for cat in cats:
        v1=c1_cats.get(cat,0); v2=c2_cats.get(cat,0)
        ax.bar(req.city1,v1,bottom=bottom1,label=cat if bottom1==0 else "")
        ax.bar(req.city2,v2,bottom=bottom2)
        bottom1+=v1; bottom2+=v2
    ax.set_title(f"{pollutant.upper()} Category Counts")
    stacked = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(c1, fill=True, label=req.city1)
    sns.kdeplot(c2, fill=True, label=req.city2)
    ax.set_title(f"{pollutant.upper()} Density")
    density = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.boxplot([c1,c2],labels=[req.city1,req.city2])
    ax.set_title(f"{pollutant.upper()} Box")
    box = fig_to_base64(fig)

    return convert_np({
        "method": method,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect,
        "normality": {req.city1:p1, req.city2:p2},
        "decision": "reject H0" if p<0.05 else "fail to reject H0",
        "stacked_bar": stacked,
        "density_plot": density,
        "box_plot": box,
        "categories_raw": {req.city1:c1_cats, req.city2:c2_cats}
    })


# ==========================
# SECTION — CATEGORICAL ANOVA (CHI)
# ==========================

@app.post("/api/anova")
async def anova_api(req: ANOVARequest):
    df = load_filtered(req.cities, req.year)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for year/cities")

    pollutant=req.pollutant.lower()
    if pollutant not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid pollutant")

    df=fill_missing_with_mean(df,pollutant)
    cats=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    df['cat']=df[pollutant].apply(lambda v: pollutant_to_cpcb_category(pollutant,v))

    contingency=pd.crosstab(df['city'],df['cat']).reindex(columns=cats,fill_value=0)
    chi2,p,dof,exp=chi2_contingency(contingency,correction=False)
    effect=cramers_v(contingency)

    pairs=list(itertools.combinations(req.cities,2))
    post=[]
    for c1,c2 in pairs:
        sub=contingency.loc[[c1,c2]]
        chi_,p_,_,_=chi2_contingency(sub,correction=False)
        post.append({"pair":f"{c1} vs {c2}","p":p_})

    if len(post)>1:
        pvals=[r['p'] for r in post]
        _,pc,_,_=multipletests(pvals,method="holm")
        for i,v in enumerate(pc):
            post[i]['p_corrected']=float(v)
            post[i]['significant']=bool(v<0.05)
    else:
        for r in post:
            r['p_corrected']=r['p']
            r['significant']=r['p']<0.05

    fig,ax=plt.subplots(figsize=(9,6))
    bottom={c:0 for c in req.cities}
    perc={}
    for c in req.cities:
        sub=df[df['city']==c]
        perc[c]=category_percent(sub,'cat',cats)
    for cat in cats:
        vals=[perc[c][cat] for c in req.cities]
        ax.bar(req.cities,vals,bottom=[bottom[c] for c in req.cities],label=cat)
        for i,city in enumerate(req.cities): bottom[city]+=vals[i]
    ax.set_ylabel("%")
    ax.set_title(f"{pollutant.upper()} CPCB Distribution")
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left')
    stacked=fig_to_base64(fig)

    size=len(req.cities)
    heat=np.ones((size,size))*np.nan
    for r in post:
        c1,c2=r['pair'].split(" vs ")
        i=req.cities.index(c1); j=req.cities.index(c2)
        heat[i,j]=r['p_corrected']; heat[j,i]=r['p_corrected']
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(heat,annot=True,cmap="viridis_r",xticklabels=req.cities,yticklabels=req.cities)
    plt.title("Post-hoc (Holm)")
    heatmap=fig_to_base64(fig)

    return convert_np({
        "method":"Chi k×m",
        "chi_square":chi2,
        "p_value":p,
        "dof":dof,
        "effect_size":effect,
        "decision":"reject H0" if p<0.05 else "fail to reject H0",
        "contingency":contingency.to_dict(),
        "percent_distribution":perc,
        "stacked_percent":stacked,
        "posthoc":post,
        "posthoc_heatmap":heatmap
    })


# ==========================
# SECTION — CHI-SQUARE (AQI × CITY)
# ==========================

@app.post("/api/chisquare")
async def chi_api(req: ChiRequest):
    df=load_filtered(req.cities, req.year)
    if df.empty:
        raise HTTPException(status_code=404,detail="No data")

    if 'aqi' not in df.columns:
        raise HTTPException(status_code=400,detail="AQI missing")

    df['aqi_cat']=df['aqi'].apply(aqi_to_category)
    cats=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    contingency=pd.crosstab(df['city'],df['aqi_cat']).reindex(columns=cats,fill_value=0)

    chi2,p,dof,_=chi2_contingency(contingency,correction=False)
    effect=cramers_v(contingency)

    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(contingency,annot=True,fmt="d",cmap="YlOrRd")
    ax.set_title("City × AQI Category")
    heatmap=fig_to_base64(fig)

    return convert_np({
        "method":"Chi",
        "chi_square":chi2,
        "p_value":p,
        "dof":dof,
        "effect_size":effect,
        "decision":"reject H0" if p<0.05 else "fail to reject H0",
        "categories":cats,
        "contingency":contingency.to_dict(),
        "heatmap":heatmap
    })


# ==========================
# SECTION — SUMMARY + VISUALS + FORECAST
# ==========================

def calculate_summary_statistics(df):
    val='aqi' if 'aqi' in df.columns else 'value'
    x=df[val].replace([np.inf,-np.inf],np.nan).dropna()
    if len(x)==0: return {"count":0}
    return convert_np({
        "count":int(len(x)),
        "mean":float(x.mean()),
        "median":float(x.median()),
        "min":float(x.min()),
        "max":float(x.max()),
        "quartiles":{
            "Q1":float(x.quantile(0.25)),
            "Q2":float(x.quantile(0.50)),
            "Q3":float(x.quantile(0.75))
        }
    })

def calculate_variability_metrics(df):
    val='aqi' if 'aqi' in df.columns else 'value'
    x=df[val].replace([np.inf,-np.inf],np.nan).dropna()
    if len(x)<2:
        return {"std_dev":None,"iqr":None,"variance":None,"cv":None}
    mean=x.mean()
    return convert_np({
        "std_dev":float(x.std()),
        "iqr":float(x.quantile(0.75)-x.quantile(0.25)),
        "variance":float(x.var()),
        "cv":float((x.std()/mean)*100) if mean!=0 else None
    })


# ==========================
# SECTION — QQ PLOTS
# ==========================

def generate_qq(df):
    pollutants = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
    qq_plots = {}

    for p in pollutants:
        if p not in df.columns:
            continue

        try:
            fig = plt.figure(figsize=(5, 4))
            stats.probplot(df[p].dropna(), dist="norm", plot=plt)
            plt.title(f"QQ Plot - {p.upper()}")

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            qq_plots[p] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        except Exception:
            qq_plots[p] = None

    return qq_plots


# ==========================
# SECTION — VISUALIZATIONS
# ==========================

def generate_visualizations(df):
    plots = {}
    if 'city' not in df.columns:
        return {}

    value_col = 'aqi' if 'aqi' in df.columns else 'value'
    df = df.dropna(subset=[value_col])

    if df.empty:
        return {}

    # BOX PLOT
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x='city', y=value_col, ax=ax)
    plt.xticks(rotation=25)
    plt.title("AQI Distribution by City")
    plots['boxplot'] = fig_to_base64(fig)

    # HISTOGRAM
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[value_col], bins=30, color='skyblue', edgecolor='black')
    plt.title("AQI Histogram")
    plots['histogram'] = fig_to_base64(fig)

    # DENSITY
    fig, ax = plt.subplots(figsize=(8, 5))
    for city in df['city'].unique():
        sns.kdeplot(df[df['city'] == city][value_col], fill=True, label=city, ax=ax)
    plt.title("AQI Density")
    plt.legend()
    plots['density'] = fig_to_base64(fig)

     # Violin
    fig, ax = plt.subplots(figsize=(9,5))
    sns.violinplot(data=df, x='city', y=value_col, ax=ax)
    plots['violin'] = fig_to_base64(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df['timestamp'], df[value_col], s=8)
    plt.xticks(rotation=20)
    plots['scatter'] = fig_to_base64(fig)

    # Hexbin
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hexbin(df[value_col], df[value_col].rolling(3).mean(), gridsize=30)
    plots['hexbin'] = fig_to_base64(fig)

    # Correlation Heatmap
    corr = df[['aqi','pm25','pm10','o3','no2','so2','co']].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plots['correlation_heatmap'] = fig_to_base64(fig)


    return plots


# ==========================
# SECTION — FORECASTING
# ==========================

def generate_predictions(df, predict_months=None, predict_year=None):
    try:
        val = 'aqi' if 'aqi' in df.columns else 'value'
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'

        if date_col not in df.columns:
            return {"error": "No timestamp column"}

        df = df[[date_col, val]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        if len(df) < 10:
            return {"error": f"Not enough data ({len(df)})"}

        df = df.groupby(date_col)[val].mean()

        periods = predict_months or (12 if predict_year else 6)

        model = ARIMA(df, order=(1, 1, 1)).fit()
        forecast = model.forecast(periods)

        return {
            "forecasted_values": forecast.tolist(),
            "method": "ARIMA (1,1,1)"
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================
# SECTION — AI SUMMARY
# ==========================

def generate_ai_summary(df, predictions=None):
    stats_ = calculate_summary_statistics(df)
    var = calculate_variability_metrics(df)

    if stats_.get("count", 0) == 0:
        return "No data."

    msg = f"""
Summary for {stats_['count']} records:

Mean AQI: {stats_['mean']:.2f}
Median AQI: {stats_['median']:.2f}
Range: {stats_['min']} - {stats_['max']}
Std Dev: {var.get('std_dev')}
IQR: {var.get('iqr')}
"""

    if predictions and not predictions.get("error"):
        msg += "\nForecasting suggests trend variation."

    return msg.strip()


# ==========================
# SECTION — MAIN ANALYZE ROUTE
# ==========================

@app.post("/api/analyze")
async def run_analysis(req: AnalysisRequest):
    df = load_filtered(req.cities, req.year)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for selected cities/year")

    summary = calculate_summary_statistics(df)
    variability = calculate_variability_metrics(df)
    visuals = generate_visualizations(df)

    correlation = compute_correlation(df)
    qqplots = generate_qq(df)

    # NORMALITY (AQI)
    if 'aqi' in df.columns:
        sample = df['aqi'].dropna()
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
        normality = {"error": "AQI missing"}

    # FORECAST
    predictions = None
    if req.predict_months or req.predict_year:
        predictions = generate_predictions(df, req.predict_months, req.predict_year)

    # SUMMARY TEXT
    ai_summary = generate_ai_summary(df, predictions)

    return convert_np({
        "summary_stats": summary,
        "variability_metrics": variability,
        "visualizations": visuals,
        "correlation_pairs": correlation,   # STYLE C OUTPUT
        "qqplots": qqplots,
        "normality_test": normality,
        "predictions": predictions,
        "ai_summary": ai_summary
    })


# ==========================
# POLLUTANTS ENDPOINT
# ==========================

@app.get("/api/pollutants")
async def get_pollutants():
    base = ['pm25','pm10','o3','no2','so2','co']
    df = DATA_CACHE.get('df')
    if df is None:
        return {"pollutants": base}
    available = [p for p in base if p in df.columns]
    return {"pollutants": available or base}


# ==========================
# ENTRYPOINT (Render uses gunicorn/uvicorn internally)
# ==========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )

