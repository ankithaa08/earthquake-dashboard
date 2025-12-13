# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ----------------- PAGE CONFIG ----------------- #
st.set_page_config(
    page_title="Earthquake & Tsunami Dashboard (1995–2023)",
    layout="wide"
)
sns.set(style="whitegrid")

# --------- STYLES / HEADINGS --------- #
st.markdown(
    """
    <style>
    h1 { font-size: 38px !important; font-weight: 800 !important; }
    h2 { font-size: 30px !important; font-weight: 700 !important; }
    h3, .stSubheader { font-size: 22px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HELPER: CENTER SMALL FIGURES ---------- #
def show_plot(fig):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ----------------- LOAD DATA ----------------- #
@st.cache_data
def load_data(path="earthquake_1995-2023.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    date_col = find_col(["date_time", "datetime", "date"])
    mag_col = find_col(["magnitude", "mag"])
    depth_col = find_col(["depth"])
    tsu_col = find_col(["tsunami"])
    lat_col = find_col(["latitude", "lat"])
    lon_col = find_col(["longitude", "lon"])

    df["date_time"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["magnitude"] = pd.to_numeric(df[mag_col], errors="coerce") if mag_col else np.nan
    df["depth"] = pd.to_numeric(df[depth_col], errors="coerce") if depth_col else np.nan
    df["tsunami"] = pd.to_numeric(df[tsu_col], errors="coerce") if tsu_col else np.nan
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan

    for col in ["cdi", "mmi", "sig"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = df["date_time"].dt.year
    return df

df = load_data()

# ----------------- MEANINGFUL COLUMNS ONLY ----------------- #
meaningful_cols = [
    c for c in ["magnitude", "depth", "sig", "cdi", "mmi", "tsunami"]
    if c in df.columns and not df[c].isna().all()
]

# ----------------- TITLE ----------------- #
st.markdown("<h1>Earthquake & Tsunami Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =========================================================
# DATA INCONSISTENCY
# =========================================================
st.markdown("<h2>➤ Check the Data for Inconsistency</h2>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Missing Values")
    st.dataframe(df[meaningful_cols].isna().sum().to_frame("Missing"))
with c2:
    st.subheader("Duplicate Rows")
    st.metric("Duplicates", int(df.duplicated().sum()))

# =========================================================
# SUMMARY STATISTICS (MEANINGFUL ONLY)
# =========================================================
st.markdown("<h2>➤ Summary Statistics – Meaningful Variables</h2>", unsafe_allow_html=True)

summary = []
for col in meaningful_cols:
    s = df[col].dropna()
    summary.append({
        "Column": col,
        "Mean": s.mean(),
        "Median": s.median(),
        "Std Dev": s.std(),
        "Q1": s.quantile(0.25),
        "Q3": s.quantile(0.75),
        "IQR": s.quantile(0.75) - s.quantile(0.25)
    })

st.dataframe(pd.DataFrame(summary).set_index("Column"))

# =========================================================
# HISTOGRAM & DENSITY
# =========================================================
st.markdown("<h2>➤ Histogram & Density Plot</h2>", unsafe_allow_html=True)

hvar = st.selectbox("Select variable:", meaningful_cols)
fig_h, ax_h = plt.subplots(figsize=(5, 3))
sns.histplot(df[hvar].dropna(), bins=30, ax=ax_h)
show_plot(fig_h)

fig_k, ax_k = plt.subplots(figsize=(5, 3))
sns.kdeplot(df[hvar].dropna(), fill=True, ax=ax_k)
show_plot(fig_k)

# =========================================================
# SPEARMAN CORRELATION WITH SIGNIFICANCE HIGHLIGHT
# =========================================================
st.markdown("<h2>➤ Spearman Correlation (ρ) with Statistical Significance</h2>", unsafe_allow_html=True)

data = df[meaningful_cols].dropna()

corr = pd.DataFrame(index=meaningful_cols, columns=meaningful_cols, dtype=float)
pval = corr.copy()

for i in meaningful_cols:
    for j in meaningful_cols:
        rho, p = spearmanr(data[i], data[j])
        corr.loc[i, j] = rho
        pval.loc[i, j] = p

st.subheader("Spearman Correlation Coefficient (ρ)")
st.dataframe(corr.round(3))

st.subheader("P-Value Matrix")
st.dataframe(pval.round(4))

# ---------- SIGNIFICANCE MASK ---------- #
significant_mask = pval < 0.05

fig_c, ax_c = plt.subplots(figsize=(6, 4))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    mask=~significant_mask,
    ax=ax_c
)
ax_c.set_title("Highlighted cells are statistically significant (p < 0.05)")
show_plot(fig_c)

# =========================================================
# RELATIONSHIP PLOTS
# =========================================================
st.markdown("<h2>➤ Relationship Plots</h2>", unsafe_allow_html=True)

x_sel = st.selectbox("X-axis:", meaningful_cols, index=0)
y_sel = st.selectbox("Y-axis:", meaningful_cols, index=1)

# Scatter
st.subheader("Scatter Plot")
fig_s, ax_s = plt.subplots(figsize=(5, 3))
sns.scatterplot(x=df[x_sel], y=df[y_sel], ax=ax_s, s=20)
show_plot(fig_s)

# Hexbin
st.subheader("Hexbin Plot")
fig_hb, ax_hb = plt.subplots(figsize=(5, 3))
hb = ax_hb.hexbin(df[x_sel], df[y_sel], gridsize=25)
fig_hb.colorbar(hb)
show_plot(fig_hb)

# Contour
st.subheader("Contour Plot")
fig_ct, ax_ct = plt.subplots(figsize=(5, 3))
sns.kdeplot(x=df[x_sel].dropna(), y=df[y_sel].dropna(), fill=True, ax=ax_ct)
show_plot(fig_ct)

# Violin
st.subheader("Violin Plot")
fig_v, ax_v = plt.subplots(figsize=(5, 3))
sns.violinplot(x="tsunami", y=y_sel, data=df, ax=ax_v)
show_plot(fig_v)


