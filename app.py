import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Wide layout
st.set_page_config(
    page_title="Earthquake & Tsunami Dashboard (1995–2023)",
    layout="wide"
)

sns.set(style="whitegrid")

# -------- BIGGER HEADINGS -------- #
st.markdown("""
<style>
h1 {
    font-size: 38px !important;
    font-weight: 800 !important;
}
h2 {
    font-size: 30px !important;
    font-weight: 700 !important;
}
h3, .stSubheader {
    font-size: 22px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTION TO CENTER + REDUCE PLOT SIZE ---------- #
def show_plot(fig):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("earthquake_1995-2023.csv")
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month
    return df

df = load_data()

numeric_cols_all = [
    "magnitude", "cdi", "mmi", "tsunami", "sig",
    "nst", "dmin", "gap", "depth", "latitude", "longitude"
]
numeric_cols = [c for c in numeric_cols_all if c in df.columns]


# ------------------ TITLE ------------------ #
st.markdown("<h1>Earthquake & Tsunami Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =========================================================
#                    ➤ OVERVIEW DASHBOARDS
# =========================================================
st.markdown("<h2>➤ Overview Dashboards</h2>", unsafe_allow_html=True)

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

fig, ax = plt.subplots(figsize=(5, 3))   # SMALL FIGURE SIZE

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

regions = sorted(df["region"].unique())
selected_region = st.selectbox("Select region:", regions)

df_r = df[df["region"] == selected_region]

fig_r, ax_r = plt.subplots(figsize=(5, 3))  # SMALL
sns.histplot(df_r["magnitude"], bins=20, kde=True, ax=ax_r)
ax_r.set_xlabel("Magnitude")

show_plot(fig_r)


# =========================================================
# BOXPLOTS
# =========================================================

st.markdown("<h2>➤ Box Plots</h2>", unsafe_allow_html=True)

y1 = st.selectbox("Boxplot Y-axis 1:", numeric_cols)
y2 = st.selectbox("Boxplot Y-axis 2:", numeric_cols)

fig_b, axes = plt.subplots(1, 2, figsize=(7, 3))  # smaller & side-by-side
sns.boxplot(y=df[y1], ax=axes[0])
sns.boxplot(y=df[y2], ax=axes[1])
axes[0].set_title(y1)
axes[1].set_title(y2)

show_plot(fig_b)

# =========================================================
# HISTOGRAM & DENSITY
# =========================================================

st.markdown("<h2>➤ Histogram & Density Plot</h2>", unsafe_allow_html=True)

hvar = st.selectbox("Select variable:", numeric_cols)

fig_h, ax_h = plt.subplots(figsize=(5, 3))
sns.histplot(df[hvar], bins=30, kde=False, ax=ax_h)
ax_h.set_xlabel(hvar)
show_plot(fig_h)

fig_k, ax_k = plt.subplots(figsize=(5, 3))
sns.kdeplot(df[hvar], fill=True, ax=ax_k)
ax_k.set_xlabel(hvar)
show_plot(fig_k)

# =========================================================
# CORRELATION MATRIX
# =========================================================

st.markdown("<h2>➤ Correlation Matrix</h2>", unsafe_allow_html=True)

corr = df[numeric_cols].corr()
st.dataframe(corr)

fig_c, ax_c = plt.subplots(figsize=(6, 4))   # smaller heatmap
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax_c)
show_plot(fig_c)


# =========================================================
# SCATTER / HEXBIN / CONTOUR / VIOLIN
# =========================================================

st.markdown("<h2>➤ Relationship Plots</h2>", unsafe_allow_html=True)

x_sel = st.selectbox("X-axis:", numeric_cols)
y_sel = st.selectbox("Y-axis:", numeric_cols)

# SCATTER
fig_sc, ax_sc = plt.subplots(figsize=(5, 3))
sns.scatterplot(x=df[x_sel], y=df[y_sel], hue=df["tsunami"], ax=ax_sc, s=20)
show_plot(fig_sc)

# HEXBIN
fig_hb, ax_hb = plt.subplots(figsize=(5, 3))
hb = ax_hb.hexbin(df[x_sel], df[y_sel], gridsize=25)
fig_hb.colorbar(hb)
show_plot(fig_hb)

# CONTOUR
fig_ct, ax_ct = plt.subplots(figsize=(5, 3))
sns.kdeplot(x=df[x_sel], y=df[y_sel], fill=True, levels=12, ax=ax_ct)
show_plot(fig_ct)

# VIOLIN
cat_cols = ["mag_category", "depth_category", "region"]
vcat = st.selectbox("Category:", cat_cols)
vy = st.selectbox("Numeric:", numeric_cols)

fig_v, ax_v = plt.subplots(figsize=(5, 3))
sns.violinplot(data=df, x=vcat, y=vy, ax=ax_v)
plt.xticks(rotation=25)
show_plot(fig_v)
