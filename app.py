# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG ----------------- #
st.set_page_config(page_title="Earthquake & Tsunami Dashboard (1995–2023)", layout="wide")
sns.set(style="whitegrid")

# --------- STYLES / HEADINGS --------- #
st.markdown("""
<style>
h1 { font-size: 38px !important; font-weight: 800 !important; }
h2 { font-size: 30px !important; font-weight: 700 !important; }
h3, .stSubheader { font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)

# ---------- HELPER ---------- #
def show_plot(fig):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ----------------- LOAD DATA ----------------- #
@st.cache_data
def load_data(path="earthquake_1995-2023.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    df["date_time"] = pd.to_datetime(df.get("date_time"), errors="coerce")
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month

    for col in ["magnitude","depth","sig","cdi","mmi","tsunami","latitude","longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

# ---------- NUMERIC COLUMNS (UNCHANGED) ---------- #
numeric_cols_all = [
    "magnitude","cdi","mmi","tsunami","sig",
    "nst","dmin","gap","depth","latitude","longitude"
]
numeric_cols = [c for c in numeric_cols_all if c in df.columns and not df[c].isna().all()]

# ---------- MEANINGFUL COLUMNS ---------- #
meaningful_cols = [c for c in ["magnitude","depth","sig","cdi","mmi","tsunami"] if c in numeric_cols]

# ---------- CATEGORY HELPERS ---------- #
def mag_category(m):
    if pd.isna(m): return np.nan
    if m < 2: return "Micro (<2.0)"
    elif m < 4: return "Minor (2.0–3.9)"
    elif m < 5: return "Light (4.0–4.9)"
    elif m < 6: return "Moderate (5.0–5.9)"
    elif m < 7: return "Strong (6.0–6.9)"
    elif m < 8: return "Major (7.0–7.9)"
    else: return "Great (8.0+)"

def depth_category(d):
    if pd.isna(d): return np.nan
    if d < 70: return "Shallow (0–70 km)"
    elif d < 300: return "Intermediate (70–300 km)"
    else: return "Deep (300+ km)"

df["mag_category"] = df["magnitude"].apply(mag_category)
df["depth_category"] = df["depth"].apply(depth_category)

# ---------- REGION ---------- #
def classify_region(lat, lon):
    if pd.isna(lat) or pd.isna(lon): return "Unknown"
    if -170 <= lon <= -30 and 5 <= lat <= 70: return "North America"
    if -90 <= lon <= -30 and -60 <= lat <= 15: return "South America"
    if -25 <= lon <= 60 and 35 <= lat <= 70: return "Europe"
    if -20 <= lon <= 55 and -40 <= lat <= 35: return "Africa"
    if 60 <= lon <= 150 and 5 <= lat <= 80: return "Asia"
    if 110 <= lon <= 180 and -50 <= lat <= 0: return "Oceania"
    if lat > 70 or lat < -70: return "Polar Regions"
    return "Unknown"

df["region"] = df.apply(lambda r: classify_region(r["latitude"], r["longitude"]), axis=1)

# ---------- TITLE ---------- #
st.markdown("<h1>Earthquake & Tsunami Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =========================================================
# OVERVIEW DASHBOARD
# =========================================================
st.markdown("<h2>➤ Overview Dashboards</h2>", unsafe_allow_html=True)

yearly = (
    df.groupby("year")
    .agg(
        earthquake_count=("magnitude","count"),
        tsunami_count=("tsunami", lambda x: (x==1).sum())
    )
    .dropna()
    .reset_index()
)

choice = st.selectbox(
    "Select view:",
    ["Earthquake count per year","Tsunami count per year","Both"]
)

fig, ax = plt.subplots(figsize=(5,3))
if choice=="Earthquake count per year":
    ax.plot(yearly["year"],yearly["earthquake_count"],marker="o")
elif choice=="Tsunami count per year":
    ax.plot(yearly["year"],yearly["tsunami_count"],marker="o",color="red")
else:
    ax.plot(yearly["year"],yearly["earthquake_count"],marker="o",label="Earthquakes")
    ax.plot(yearly["year"],yearly["tsunami_count"],marker="o",color="red",label="Tsunamis")
    ax.legend()
show_plot(fig)

st.info("Conclusion: Earthquake and tsunami occurrences vary across years.")

# =========================================================
# REGION ANALYSIS
# =========================================================
st.subheader("Region-Based Analysis")

regions = sorted(df["region"].dropna().unique())
selected_region = st.selectbox("Select region:", regions)
df_r = df[df["region"]==selected_region]

st.markdown(f"**Total earthquakes in {selected_region}: {len(df_r)}**")

fig_r, ax_r = plt.subplots(figsize=(5,3))
sns.histplot(df_r["magnitude"].dropna(), bins=20, kde=True, ax=ax_r)
show_plot(fig_r)

st.info("Conclusion: Magnitude distribution differs by geographic region.")

# =========================================================
# SUMMARY STATISTICS (MEANINGFUL ONLY)
# =========================================================
st.markdown("<h2>➤ Summary Statistics</h2>", unsafe_allow_html=True)

summary=[]
for col in meaningful_cols:
    s=df[col].dropna()
    summary.append({
        "Variable":col,
        "Mean":s.mean(),
        "Median":s.median(),
        "Std Dev":s.std(),
        "MAD":np.median(np.abs(s-s.median())),
        "IQR":s.quantile(0.75)-s.quantile(0.25)
    })

st.dataframe(pd.DataFrame(summary).set_index("Variable"))
st.info("Conclusion: High variability and skewness are observed in earthquake data.")

# =========================================================
# BOXPLOTS
# =========================================================
st.markdown("<h2>➤ Box Plots</h2>", unsafe_allow_html=True)

cA,cB=st.columns(2)
y1=st.selectbox("Boxplot Y-axis 1:",numeric_cols,index=0)
y2=st.selectbox("Boxplot Y-axis 2:",numeric_cols,index=1)

fig_b,axes=plt.subplots(1,2,figsize=(7,3))
sns.boxplot(y=df[y1].dropna(),ax=axes[0])
sns.boxplot(y=df[y2].dropna(),ax=axes[1])
show_plot(fig_b)

st.info("Conclusion: Presence of outliers confirms non-normal distributions.")

# =========================================================
# HISTOGRAM & DENSITY
# =========================================================
st.markdown("<h2>➤ Histogram & Density Plot</h2>", unsafe_allow_html=True)

hvar=st.selectbox("Select variable:",numeric_cols)

fig_h,ax_h=plt.subplots(figsize=(5,3))
sns.histplot(df[hvar].dropna(),bins=30,ax=ax_h)
show_plot(fig_h)

fig_k,ax_k=plt.subplots(figsize=(5,3))
sns.kdeplot(df[hvar].dropna(),fill=True,ax=ax_k)
show_plot(fig_k)

st.info("Conclusion: Data distributions are skewed and non-normal.")

# =========================================================
# CORRELATION (SPEARMAN)
# =========================================================
st.markdown("<h2>➤ Spearman Correlation</h2>", unsafe_allow_html=True)

data=df[meaningful_cols].dropna()
corr=data.corr(method="spearman")

fig_c,ax_c=plt.subplots(figsize=(6,4))
sns.heatmap(corr,annot=True,cmap="coolwarm",vmin=-1,vmax=1,ax=ax_c)
show_plot(fig_c)

st.info("Conclusion: Magnitude shows a strong monotonic relationship with significance.")

# =========================================================
# RELATIONSHIP PLOTS (UNCHANGED)
# =========================================================
st.markdown("<h2>➤ Relationship Plots</h2>", unsafe_allow_html=True)

x_sel=st.selectbox("X-axis:",numeric_cols,index=0)
y_sel=st.selectbox("Y-axis:",numeric_cols,index=1)

# Scatter
st.subheader("Scatter Plot")
fig_sc,ax_sc=plt.subplots(figsize=(5,3))
sns.scatterplot(x=df[x_sel],y=df[y_sel],hue=df["tsunami"],ax=ax_sc,s=20)
show_plot(fig_sc)

# Hexbin
st.subheader("Hexbin Plot")
fig_hb,ax_hb=plt.subplots(figsize=(5,3))
hb=ax_hb.hexbin(df[x_sel],df[y_sel],gridsize=25)
fig_hb.colorbar(hb)
show_plot(fig_hb)

# Contour
st.subheader("Contour Plot")
fig_ct,ax_ct=plt.subplots(figsize=(5,3))
sns.kdeplot(x=df[x_sel].dropna(),y=df[y_sel].dropna(),fill=True,ax=ax_ct)
show_plot(fig_ct)

# Violin
st.subheader("Violin Plot")
cat_cols=[c for c in ["mag_category","depth_category","region"] if c in df.columns]
vcat=st.selectbox("Category:",cat_cols)
vy=st.selectbox("Numeric:",numeric_cols)

fig_v,ax_v=plt.subplots(figsize=(5,3))
sns.violinplot(data=df,x=vcat,y=vy,ax=ax_v)
plt.xticks(rotation=25)
show_plot(fig_v)

st.info("Conclusion: Relationship plots show clustering and non-linear patterns.")






