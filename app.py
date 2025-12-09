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
    """Center a Matplotlib figure and prevent it from expanding full width."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ----------------- LOAD & NORMALIZE DATA ----------------- #
@st.cache_data
def load_data(path="earthquake_1995-2023.csv"):
    df = pd.read_csv(path)
    # Normalize column names: strip and lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # helper to find first match from candidates
    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    # Common candidate names
    date_cands = ["date_time", "datetime", "date", "time", "timestamp"]
    mag_cands = ["magnitude", "mag"]
    tsunami_cands = ["tsunami", "tsun"]
    depth_cands = ["depth"]
    lat_cands = ["latitude", "lat", "y"]
    lon_cands = ["longitude", "lon", "lng", "long", "x"]
    other_cands = ["cdi", "mmi", "sig", "nst", "dmin", "gap", "latitude", "longitude"]

    # find columns
    date_col = find_col(date_cands)
    mag_col = find_col(mag_cands)
    tsu_col = find_col(tsunami_cands)
    depth_col = find_col(depth_cands)
    lat_col = find_col(lat_cands)
    lon_col = find_col(lon_cands)

    # create canonical columns if present
    if date_col:
        # try common formats; allow coercion to NaT
        df["date_time"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        df["date_time"] = pd.NaT

    if mag_col:
        df["magnitude"] = pd.to_numeric(df[mag_col], errors="coerce")
    else:
        df["magnitude"] = np.nan

    if tsu_col:
        # standardize tsunami to 0/1 when possible
        df["tsunami"] = pd.to_numeric(df[tsu_col], errors="coerce")
        # if values are boolean/string like 'yes', try to coerce
        df["tsunami"] = df["tsunami"].fillna(df[tsu_col].map(lambda v: 1 if str(v).strip().lower() in ("1", "yes", "true", "y", "t") else 0 if str(v).strip().lower() in ("0","no","false","n","f") else np.nan))
        # finally, convert to int where possible
        try:
            df["tsunami"] = df["tsunami"].astype("Int64")
        except Exception:
            pass
    else:
        df["tsunami"] = np.nan

    if depth_col:
        df["depth"] = pd.to_numeric(df[depth_col], errors="coerce")
    else:
        df["depth"] = np.nan

    if lat_col:
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        df["latitude"] = np.nan

    if lon_col:
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        df["longitude"] = np.nan

    # keep useful other numeric columns if present
    for col in ["cdi", "mmi", "sig", "nst", "dmin", "gap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # year / month extracted from date_time
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month

    return df, {
        "date_col": date_col,
        "mag_col": mag_col,
        "tsu_col": tsu_col,
        "depth_col": depth_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }

df, detected_cols = load_data()

# show a small debug summary so you know what was detected (safe to remove later)
st.markdown("<h3>Data detection (auto-mapping)</h3>", unsafe_allow_html=True)
st.write(detected_cols)

# ---------- numeric columns list (only those present) ---------- #
numeric_cols_all = ["magnitude", "cdi", "mmi", "tsunami", "sig", "nst", "dmin", "gap", "depth", "latitude", "longitude"]
numeric_cols = [c for c in numeric_cols_all if c in df.columns and not df[c].isna().all()]

# ---------- category helpers ---------- #
def mag_category(m):
    if pd.isna(m):
        return np.nan
    if m < 2:
        return "Micro (<2.0)"
    elif m < 4:
        return "Minor (2.0–3.9)"
    elif m < 5:
        return "Light (4.0–4.9)"
    elif m < 6:
        return "Moderate (5.0–5.9)"
    elif m < 7:
        return "Strong (6.0–6.9)"
    elif m < 8:
        return "Major (7.0–7.9)"
    else:
        return "Great (8.0+)"

def depth_category(d):
    if pd.isna(d):
        return np.nan
    if d < 70:
        return "Shallow (0–70 km)"
    elif d < 300:
        return "Intermediate (70–300 km)"
    else:
        return "Deep (300+ km)"

df["mag_category"] = df["magnitude"].apply(mag_category)
df["depth_category"] = df["depth"].apply(depth_category)

# ----------------- REGION (robust) ----------------- #
def classify_region(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    if -170 <= lon <= -30 and 5 <= lat <= 70:
        return "North America"
    if -90 <= lon <= -30 and -60 <= lat <= 15:
        return "South America"
    if -25 <= lon <= 60 and 35 <= lat <= 70:
        return "Europe"
    if -20 <= lon <= 55 and -40 <= lat <= 35:
        return "Africa"
    if 60 <= lon <= 150 and 5 <= lat <= 80:
        return "Asia"
    if 110 <= lon <= 180 and -50 <= lat <= 0:
        return "Oceania"
    if lat > 70 or lat < -70:
        return "Polar Regions"
    return "Unknown"

# If latitude & longitude exist, create region; otherwise Unknown
if "latitude" in df.columns and "longitude" in df.columns:
    df["region"] = df.apply(lambda r: classify_region(r["latitude"], r["longitude"]), axis=1)
else:
    df["region"] = "Unknown"

df["region"] = df["region"].astype(str)

# ---------- TITLE ---------- #
st.markdown("<h1>Earthquake & Tsunami Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =========================================================
#                    ➤ OVERVIEW DASHBOARDS
# =========================================================
st.markdown("<h2>➤ Overview Dashboards</h2>", unsafe_allow_html=True)

# ---------- YEARLY LINE CHART ---------- #
st.subheader("Yearwise Earthquake & Tsunami Trend")

# safe grouping - if 'year' missing will produce NaN group but that's fine
yearly = (
    df.groupby("year")
    .agg(
        earthquake_count=("magnitude", "count"),
        tsunami_count=("tsunami", lambda x: (x == 1).sum() if x.notna().any() else 0)
    )
    .reset_index()
)

# if year is NaN rows exist, drop NaN for plotting
yearly = yearly.dropna(subset=["year"])

choice = st.selectbox("Select view:", ["Earthquake count per year", "Tsunami count per year", "Both"])

fig, ax = plt.subplots(figsize=(5, 3))
if choice == "Earthquake count per year":
    ax.plot(yearly["year"], yearly["earthquake_count"], marker="o")
elif choice == "Tsunami count per year":
    ax.plot(yearly["year"], yearly["tsunami_count"], marker="o", color="red")
else:
    ax.plot(yearly["year"], yearly["earthquake_count"], marker="o", label="Earthquakes")
    ax.plot(yearly["year"], yearly["tsunami_count"], marker="o", label="Tsunamis", color="red")
    ax.legend()

ax.set_xlabel("Year")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.3)
show_plot(fig)

# =========================================================
# REGION ANALYSIS
# =========================================================
st.subheader("Region-Based Analysis")

regions = sorted(df["region"].dropna().unique().tolist())
if len(regions) == 0:
    regions = ["Unknown"]

selected_region = st.selectbox("Select region:", regions)
df_r = df[df["region"] == selected_region]

st.markdown(f"<h3>{selected_region}: {len(df_r)} earthquakes</h3>", unsafe_allow_html=True)

fig_r, ax_r = plt.subplots(figsize=(5, 3))
# handle missing magnitude gracefully
if "magnitude" in df_r.columns and not df_r["magnitude"].dropna().empty:
    sns.histplot(df_r["magnitude"], bins=20, kde=True, ax=ax_r)
    ax_r.set_xlabel("Magnitude")
else:
    ax_r.text(0.5, 0.5, "No magnitude data", ha="center", va="center")
show_plot(fig_r)

# =========================================================
#                 ➤ DATA INCONSISTENCY CHECK
# =========================================================
st.markdown("<h2>➤ Check the Data for Inconsistency</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Missing Values")
    st.dataframe(df.isna().sum().to_frame("Missing"))
with col2:
    st.subheader("Duplicate Rows")
    st.metric("Duplicates", int(df.duplicated().sum()))

# =========================================================
#                 ➤ SUMMARY STATISTICS
# =========================================================
st.markdown("<h2>➤ Summary Statistics – Mean, Median, Quartiles</h2>", unsafe_allow_html=True)

summary = []
for col in numeric_cols:
    s = df[col].dropna()
    summary.append({
        "Column": col,
        "Mean": s.mean() if not s.empty else np.nan,
        "Median": s.median() if not s.empty else np.nan,
        "Min": s.min() if not s.empty else np.nan,
        "Max": s.max() if not s.empty else np.nan,
        "Q1 (25%)": s.quantile(0.25) if not s.empty else np.nan,
        "Q3 (75%)": s.quantile(0.75) if not s.empty else np.nan,
    })

summary_df = pd.DataFrame(summary).set_index("Column")
st.dataframe(summary_df)

# =========================================================
#                     ➤ BOXPLOTS
# =========================================================
st.markdown("<h2>➤ Box Plots</h2>", unsafe_allow_html=True)

if len(numeric_cols) >= 2:
    cA, cB = st.columns(2)
    with cA:
        y1 = st.selectbox("Boxplot Y-axis 1:", numeric_cols, index=0)
    with cB:
        y2 = st.selectbox("Boxplot Y-axis 2:", numeric_cols, index=1)
    fig_b, axes = plt.subplots(1, 2, figsize=(7, 3))
    sns.boxplot(y=df[y1].dropna(), ax=axes[0])
    sns.boxplot(y=df[y2].dropna(), ax=axes[1])
    axes[0].set_title(y1)
    axes[1].set_title(y2)
    show_plot(fig_b)
else:
    st.info("Not enough numeric columns available for boxplots.")

# =========================================================
#             ➤ VARIABILITY METRICS
# =========================================================
st.markdown("<h2>➤ Variability Metrics – Std Dev, MAD, IQR</h2>", unsafe_allow_html=True)

def MAD(s):
    med = s.median()
    return np.median(np.abs(s - med))

metrics = []
for col in numeric_cols:
    s = df[col].dropna()
    metrics.append({
        "Column": col,
        "Std Dev": float(s.std()) if not s.empty else np.nan,
        "MAD": float(MAD(s)) if not s.empty else np.nan,
        "IQR": float(s.quantile(0.75) - s.quantile(0.25)) if not s.empty else np.nan
    })
st.dataframe(pd.DataFrame(metrics).set_index("Column"))

# =========================================================
#         ➤ HISTOGRAM & DENSITY
# =========================================================
st.markdown("<h2>➤ Histogram & Density Plot</h2>", unsafe_allow_html=True)

if len(numeric_cols) > 0:
    hvar = st.selectbox("Select variable:", numeric_cols)
    fig_h, ax_h = plt.subplots(figsize=(5, 3))
    sns.histplot(df[hvar].dropna(), bins=30, kde=False, ax=ax_h)
    ax_h.set_xlabel(hvar)
    show_plot(fig_h)

    fig_k, ax_k = plt.subplots(figsize=(5, 3))
    sns.kdeplot(df[hvar].dropna(), fill=True, ax=ax_k)
    ax_k.set_xlabel(hvar)
    show_plot(fig_k)
else:
    st.info("No numeric columns available for histogram/density.")

# =========================================================
#                ➤ CORRELATION MATRIX
# =========================================================
st.markdown("<h2>➤ Correlation Matrix</h2>", unsafe_allow_html=True)

if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    st.dataframe(corr)
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax_c)
    show_plot(fig_c)
else:
    st.info("Not enough numeric columns to compute correlation matrix.")

# =========================================================
#        ➤ RELATIONSHIP PLOTS (SCATTER, HEXBIN, CONTOUR, VIOLIN)
# =========================================================
st.markdown("<h2>➤ Relationship Plots</h2>", unsafe_allow_html=True)

if len(numeric_cols) >= 2:
    x_sel = st.selectbox("X-axis:", numeric_cols, index=0)
    y_sel = st.selectbox("Y-axis:", numeric_cols, index=1)

    # SCATTER
    st.subheader("Scatter Plot")
    fig_sc, ax_sc = plt.subplots(figsize=(5, 3))
    # hue fallback: if tsunami not numeric present, don't use hue
    if "tsunami" in df.columns and df["tsunami"].notna().any():
        try:
            sns.scatterplot(x=df[x_sel], y=df[y_sel], hue=df["tsunami"], palette="viridis", ax=ax_sc, s=20)
        except Exception:
            sns.scatterplot(x=df[x_sel], y=df[y_sel], ax=ax_sc, s=20)
    else:
        sns.scatterplot(x=df[x_sel], y=df[y_sel], ax=ax_sc, s=20)
    ax_sc.set_xlabel(x_sel)
    ax_sc.set_ylabel(y_sel)
    show_plot(fig_sc)

    # HEXBIN
    st.subheader("Hexbin Plot")
    fig_hb, ax_hb = plt.subplots(figsize=(5, 3))
    try:
        hb = ax_hb.hexbin(df[x_sel], df[y_sel], gridsize=25)
        fig_hb.colorbar(hb)
    except Exception:
        ax_hb.text(0.5, 0.5, "Cannot create hexbin with provided data", ha="center", va="center")
    show_plot(fig_hb)

    # CONTOUR
    st.subheader("Contour Plot")
    fig_ct, ax_ct = plt.subplots(figsize=(5, 3))
    try:
        sns.kdeplot(x=df[x_sel].dropna(), y=df[y_sel].dropna(), fill=True, levels=12, ax=ax_ct)
    except Exception:
        ax_ct.text(0.5, 0.5, "Cannot create contour with provided data", ha="center", va="center")
    show_plot(fig_ct)

    # VIOLIN
    st.subheader("Violin Plot")
    cat_cols = [c for c in ["mag_category", "depth_category", "region"] if c in df.columns]
    if len(cat_cols) == 0:
        st.info("No categorical columns available for violin plot.")
    else:
        vcat = st.selectbox("Category:", cat_cols)
        vy = st.selectbox("Numeric:", numeric_cols)
        fig_v, ax_v = plt.subplots(figsize=(5, 3))
        try:
            sns.violinplot(data=df, x=vcat, y=vy, ax=ax_v)
            plt.xticks(rotation=25)
        except Exception:
            ax_v.text(0.5, 0.5, "Cannot create violin with provided data", ha="center", va="center")
        show_plot(fig_v)
else:
    st.info("Not enough numeric columns for relationship plots.")
