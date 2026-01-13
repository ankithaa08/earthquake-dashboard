# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

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



# =========================================================
# REGION ANALYSIS
# =========================================================
st.subheader("Region-Based Analysis")

regions = sorted(df["region"].dropna().unique())
selected_region = st.selectbox("Select region:", regions)

df_r = df[df["region"] == selected_region]

# Simple count
total_eq = len(df_r)
st.markdown(f"**Total earthquakes in {selected_region}: {total_eq}**")

# Simple, meaningful insights
if total_eq == 0:
    st.info("No recorded earthquakes for this region in the dataset.")
else:
    avg_mag = df_r["magnitude"].mean()
    max_mag = df_r["magnitude"].max()

    st.info(
        f"In {selected_region}, earthquake occurrence is relatively "
        f"{'low' if total_eq < 20 else 'moderate'}. "
        f"The average magnitude is approximately {avg_mag:.2f}, "
        f"with a maximum recorded magnitude of {max_mag:.2f}. "
    )


# =========================================================
# MEANINGFUL RISK-BASED EARTHQUAKE MAP
# =========================================================
st.markdown("<h2>➤ Global Earthquake Risk Map</h2>", unsafe_allow_html=True)

# Keep only valid spatial & magnitude data
map_df = df.dropna(subset=["latitude", "longitude", "magnitude"]).copy()

# ---------- DATA-DRIVEN RELATIVE RISK ----------
q1 = map_df["magnitude"].quantile(0.33)
q2 = map_df["magnitude"].quantile(0.66)

def relative_risk(mag):
    if mag <= q1:
        return "Low"
    elif mag <= q2:
        return "Moderate"
    else:
        return "High"

map_df["risk"] = map_df["magnitude"].apply(relative_risk).astype(str)

# ---------- COLOR MAPPING (RGBA) ----------
color_map = {
    "Low": [0, 180, 0, 180],        # Green
    "Moderate": [255, 165, 0, 200], # Orange
    "High": [220, 20, 60, 220]      # Red
}

map_df["color"] = map_df["risk"].map(color_map)

# ---------- PYDECK LAYER ----------
import pydeck as pdk

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[longitude, latitude]",
    get_radius="magnitude * 9000",   # Cleaner scaling
    get_fill_color="color",
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=0,
    longitude=0,
    zoom=1,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "text": (
            "Magnitude: {magnitude}\n"
            "Depth: {depth} km\n"
            "Risk Level: {risk}\n"
            "Region: {region}"
        )
    },
)

st.pydeck_chart(deck)

st.info(
    "Conclusion: Higher-magnitude earthquakes (shown in red) exhibit clear spatial clustering "
    "along tectonic plate boundaries, particularly around the Pacific Ring of Fire. "
    "The risk levels shown are relative classifications derived from the magnitude distribution."
)




# =========================================================
# SUMMARY STATISTICS (MEANINGFUL ONLY)
# =========================================================
st.markdown("<h2>➤ Summary Statistics</h2>", unsafe_allow_html=True)

summary = []

for col in meaningful_cols:
    if col == "tsunami":
        continue   # skip tsunami (binary variable)

    s = df[col].dropna()
    summary.append({
        "Variable": col,
        "Mean": s.mean(),
        "Median": s.median(),
        "Std Dev": s.std(),
        "MAD": np.median(np.abs(s - s.median())),
        "IQR": s.quantile(0.75) - s.quantile(0.25)
    })

st.dataframe(pd.DataFrame(summary).set_index("Variable"))

st.info(
    "Conclusion: Continuous variables such as magnitude and depth show variability and skewness. "
    "Ordinal variables are better interpreted using median and IQR. "
)


# =========================================================
# BOXPLOTS
# =========================================================
st.markdown("<h2>➤ Box Plots</h2>", unsafe_allow_html=True)

st.markdown("## 🌊 Tsunami vs Earthquake Magnitude")

fig, ax = plt.subplots(figsize=(5, 4))

sns.boxplot(
    x="tsunami",
    y="magnitude",
    data=df,
    ax=ax
)

ax.set_xlabel("Tsunami (0 = No, 1 = Yes)")
ax.set_ylabel("Magnitude")
ax.set_title("Magnitude Distribution for Tsunami vs Non-Tsunami Earthquakes")

show_plot(fig)

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
# =========================================================
# =========================================================
# SPEARMAN CORRELATION WITH P-VALUE
# =========================================================
# SPEARMAN CORRELATION WITH CLEAR P-VALUE INTERPRETATION
# =========================================================
st.markdown("<h2>➤ Spearman Correlation Analysis</h2>", unsafe_allow_html=True)

from scipy.stats import spearmanr

# ---- Select ONLY valid columns for Spearman ----
spearman_cols = [c for c in meaningful_cols if c != "tsunami"]
data = df[spearman_cols].dropna()

# ---- Initialize matrices ----
corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
pval_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
sig_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=str)

# ---- Compute Spearman correlation ----
for col1 in data.columns:
    for col2 in data.columns:
        rho, pval = spearmanr(data[col1], data[col2], nan_policy="omit")

        corr_matrix.loc[col1, col2] = rho
        pval_matrix.loc[col1, col2] = pval

        if col1 == col2:
            sig_matrix.loc[col1, col2] = "-"
        elif pval < 0.05:
            sig_matrix.loc[col1, col2] = "Significant ✅"
        else:
            sig_matrix.loc[col1, col2] = "Not Significant ❌"

# ---- Heatmap ----
fig_c, ax_c = plt.subplots(figsize=(6, 4))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=ax_c
)
show_plot(fig_c)

# ---- Tables ----
st.subheader("Spearman Correlation Coefficient (ρ)")
st.dataframe(corr_matrix.round(3))

st.subheader("P-Value Matrix")
st.dataframe(pval_matrix.round(4))




st.subheader("Statistical Significance (α = 0.05)")
st.dataframe(sig_matrix)

# ---- Explanation ----
st.info("""
- Spearman correlation (ρ) measures monotonic relationships.
- It is suitable for skewed and ordinal variables.
- A p-value < 0.05 indicates a statistically significant relationship.
""")

# ---- Final conclusion (deduplicated & safe) ----
significant_pairs = set()

for i, col1 in enumerate(data.columns):
    for col2 in data.columns[i+1:]:
        if pval_matrix.loc[col1, col2] < 0.05:
            significant_pairs.add(f"{col1} ↔ {col2}")



# =========================================================
# RELATIONSHIP PLOTS (UNCHANGED)
# =========================================================
st.markdown("<h2>➤ Relationship Plots</h2>", unsafe_allow_html=True)

# ---------------- Scatter Plot ----------------
st.subheader("Scatter Plot: Magnitude vs Depth")

fig_sc, ax_sc = plt.subplots(figsize=(5,3))
sns.scatterplot(
    x=df["depth"],
    y=df["magnitude"],
    hue=df["tsunami"],
    palette={0: "blue", 1: "red"},
    alpha=0.5,
    s=20,
    ax=ax_sc
)

ax_sc.set_xlabel("Depth (km)")
ax_sc.set_ylabel("Magnitude")
ax_sc.set_title("Scatter Plot of Magnitude vs Depth")

show_plot(fig_sc)

st.info(
    "Scatter Plot Conclusion: The plot shows a wide spread of magnitudes across depths, "
    "indicating no strong linear relationship between earthquake depth and magnitude. "
    "Tsunami events (red) tend to occur at moderate to higher magnitudes."
)

# ---------------- Hexbin Plot ----------------
st.subheader("Hexbin Plot: Magnitude vs Depth (Density View)")

fig_hb, ax_hb = plt.subplots(figsize=(5,3))
hb = ax_hb.hexbin(
    df["depth"],
    df["magnitude"],
    gridsize=30,
    cmap="viridis"
)

ax_hb.set_xlabel("Depth (km)")
ax_hb.set_ylabel("Magnitude")
ax_hb.set_title("Hexbin Density Plot of Magnitude vs Depth")

fig_hb.colorbar(hb, ax=ax_hb, label="Event Density")
show_plot(fig_hb)

st.info(
    "Hexbin Plot Conclusion: The hexbin plot highlights dense clusters of shallow earthquakes, "
    "showing that most seismic activity occurs at lower depths, while deep and high-magnitude "
    "events are comparatively rare."
)

# ---------------- Violin Plot ----------------
st.subheader("Violin Plot: Magnitude vs Tsunami")

fig_v, ax_v = plt.subplots(figsize=(5,3))
sns.violinplot(
    data=df,
    x="tsunami",
    y="magnitude",
    inner="quartile",
    cut=0,
    ax=ax_v
)

ax_v.set_xlabel("Tsunami (0 = No, 1 = Yes)")
ax_v.set_ylabel("Magnitude")
ax_v.set_title("Distribution of Earthquake Magnitude by Tsunami Occurrence")

show_plot(fig_v)

st.title("Earthquake Data Hypothesis Testing")

# Problem Statement
st.header("Problem Statement")
st.write(
    "To test whether the average magnitude of earthquakes "
    "from 1995 to 2023 is significantly greater than 5.0."
)

# Load Dataset
st.header("Dataset Preview")
df = pd.read_csv("earthquake_1995-2023.csv")
st.dataframe(df.head())

# Select Magnitude Column
st.header("Hypothesis Testing")
magnitude_col = st.selectbox(
    "Select magnitude column:",
    df.select_dtypes(include=np.number).columns
)

data = df[magnitude_col].dropna()

# Hypothesis description
st.subheader("Hypotheses")
st.write("H₀: Mean magnitude = 5.0")
st.write("H₁: Mean magnitude > 5.0")

# One-sample t-test
t_stat, p_value = ttest_1samp(data, popmean=5)

# One-tailed p-value
p_value_one_tailed = p_value / 2

# Results
st.subheader("Test Results")
st.write(f"Sample Mean: **{data.mean():.2f}**")
st.write(f"T-statistic: **{t_stat:.3f}**")
st.write(f"P-value (one-tailed): **{p_value_one_tailed:.4f}**")

# Decision
alpha = 0.05
st.subheader("Conclusion")

if p_value_one_tailed < alpha and data.mean() > 5:
    st.success("Reject H₀: The average earthquake magnitude is significantly greater than 5.0")
else:
    st.info("Fail to Reject H₀: No significant evidence that the mean magnitude is greater than 5.0")






















