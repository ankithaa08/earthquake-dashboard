import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Earthquake & Tsunami Dashboard (1995–2023)", layout="wide")
sns.set(style="whitegrid")

# ------------------ DATA LOADING ------------------ #
@st.cache_data
def load_data():
    df = pd.read_csv("earthquake_1995-2023.csv")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month
    return df

df = load_data()

numeric_cols_all = [
    "magnitude", "cdi", "mmi", "tsunami", "sig",
    "nst", "dmin", "gap", "depth", "latitude", "longitude"
]
numeric_cols = [c for c in numeric_cols_all if c in df.columns]

# ------------------ HELPER CATEGORIES ------------------ #
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

# ------------------ REGION CLASSIFICATION ------------------ #
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

# ------------------ TITLE ------------------ #
st.markdown("<h1><b>Earthquake & Tsunami Analysis Dashboard (1995–2023)</b></h1>", unsafe_allow_html=True)
st.write("This dashboard explores global earthquake patterns using multiple visual analysis tools.")

# =========================================================
#  ➤ OVERVIEW DASHBOARDS
# =========================================================
st.markdown("<h2><b>➤ Overview Dashboards</b></h2>", unsafe_allow_html=True)

# ---------- LINE CHART ---------- #
st.subheader("Yearwise Earthquake & Tsunami Trend")

yearly = (
    df.groupby("year")
      .agg(
          earthquake_count=("magnitude", "count"),
          tsunami_count=("tsunami", lambda x: (x == 1).sum())
      )
      .reset_index()
)

choice = st.selectbox(
    "Select view:",
    ["Earthquake count per year", "Tsunami count per year", "Both"],
)

fig, ax = plt.subplots(figsize=(10, 4))

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
st.pyplot(fig)

# ---------- REGION ANALYSIS ---------- #
st.subheader("Region-Based Analysis")

region_list = sorted(df["region"].unique())
chosen_region = st.selectbox("Select region:", region_list)

df_r = df[df["region"] == chosen_region]

st.markdown(f"<h4>{chosen_region}: {len(df_r)} earthquakes</h4>", unsafe_allow_html=True)

fig_r, ax_r = plt.subplots(figsize=(6, 4))
sns.histplot(df_r["magnitude"], bins=20, kde=True, ax=ax_r)
ax_r.set_xlabel("Magnitude")
st.pyplot(fig_r)

# =========================================================
#  ➤ DATA INCONSISTENCY CHECK
# =========================================================
st.markdown("<h2><b>➤ Check the Data for Inconsistency</b></h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Missing Values")
    st.dataframe(df.isna().sum().to_frame("Missing"))

with col2:
    st.subheader("Duplicate Rows")
    st.metric("Duplicates", df.duplicated().sum())

# =========================================================
#  ➤ SUMMARY STATISTICS
# =========================================================
st.markdown("<h2><b>➤ Summary Statistics – Mean, Median, Quartiles</b></h2>", unsafe_allow_html=True)

summary = []
for col in numeric_cols:
    s = df[col].dropna()
    summary.append({
        "Column": col,
        "Mean": s.mean(),
        "Median": s.median(),
        "Min": s.min(),
        "Max": s.max(),
        "Q1 (25%)": s.quantile(0.25),
        "Q3 (75%)": s.quantile(0.75),
    })

summary_df = pd.DataFrame(summary).set_index("Column")
st.dataframe(summary_df)

# =========================================================
#  ➤ BOXPLOTS
# =========================================================
st.markdown("<h2><b>➤ Box Plots</b></h2>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    y1 = st.selectbox("Boxplot Y-axis 1:", numeric_cols, key="b1")
with colB:
    y2 = st.selectbox("Boxplot Y-axis 2:", numeric_cols, key="b2")

fig_b, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df[y1], ax=axes[0])
sns.boxplot(y=df[y2], ax=axes[1])
axes[0].set_title(y1)
axes[1].set_title(y2)
st.pyplot(fig_b)

# =========================================================
#  ➤ VARIABILITY METRICS
# =========================================================
st.markdown("<h2><b>➤ Variability Metrics – Std Dev, MAD, IQR</b></h2>", unsafe_allow_html=True)

def MAD(s):
    med = s.median()
    return np.median(np.abs(s - med))

metrics = []
for col in numeric_cols:
    s = df[col].dropna()
    metrics.append({
        "Column": col,
        "Std Dev": s.std(),
        "MAD": MAD(s),
        "IQR": s.quantile(0.75) - s.quantile(0.25)
    })

st.dataframe(pd.DataFrame(metrics).set_index("Column"))

# =========================================================
#  ➤ HISTOGRAMS & DENSITY
# =========================================================
st.markdown("<h2><b>➤ Histogram & Density Plot</b></h2>", unsafe_allow_html=True)

hvar = st.selectbox("Select variable:", numeric_cols)

colH1, colH2 = st.columns(2)

with colH1:
    fig_h, ax_h = plt.subplots()
    sns.histplot(df[hvar], bins=30, kde=False, ax=ax_h)
    st.pyplot(fig_h)

with colH2:
    fig_k, ax_k = plt.subplots()
    sns.kdeplot(df[hvar], fill=True, ax=ax_k)
    st.pyplot(fig_k)

# =========================================================
#  ➤ CORRELATION MATRIX
# =========================================================
st.markdown("<h2><b>➤ Correlation Matrix</b></h2>", unsafe_allow_html=True)

corr = df[numeric_cols].corr()
st.dataframe(corr)

fig_c, ax_c = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax_c)
st.pyplot(fig_c)

# =========================================================
#  ➤ RELATIONSHIP PLOTS
# =========================================================
st.markdown("<h2><b>➤ Relationship Plots – Scatter, Hexbin, Contour, Violin</b></h2>", unsafe_allow_html=True)

x_sel = st.selectbox("X-axis:", numeric_cols)
y_sel = st.selectbox("Y-axis:", numeric_cols)

# ---------- Scatter ---------- #
fig_sc, ax_sc = plt.subplots()
sns.scatterplot(x=df[x_sel], y=df[y_sel], hue=df["tsunami"], ax=ax_sc, palette="viridis")
st.subheader("Scatter Plot")
st.pyplot(fig_sc)

# ---------- Hexbin ---------- #
fig_hb, ax_hb = plt.subplots()
hb = ax_hb.hexbin(df[x_sel], df[y_sel], gridsize=40)
fig_hb.colorbar(hb)
st.subheader("Hexbin Plot")
st.pyplot(fig_hb)

# ---------- Contour ---------- #
fig_ct, ax_ct = plt.subplots()
sns.kdeplot(x=df[x_sel], y=df[y_sel], fill=True, levels=20, ax=ax_ct)
st.subheader("Contour Plot")
st.pyplot(fig_ct)

# ---------- Violin ---------- #
st.subheader("Violin Plot")

cat_cols = ["mag_category", "depth_category", "region"]
cat_cols = [c for c in cat_cols if c in df.columns]

vcat = st.selectbox("Category:", cat_cols)
vy = st.selectbox("Numeric:", numeric_cols)

fig_v, ax_v = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df, x=vcat, y=vy, ax=ax_v)
plt.xticks(rotation=20)
st.pyplot(fig_v)












