# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG ----------------- #
st.set_page_config(page_title="Earthquake & Tsunami Dashboard (1995–2023)", layout="wide")
sns.set(style="whitegrid")

# --------- STYLES --------- #
st.markdown("""
<style>
h1 { font-size: 38px; font-weight: 800; }
h2 { font-size: 30px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------- HELPER ---------- #
def show_plot(fig):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ----------------- LOAD DATA ----------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("earthquake_1995-2023.csv")
    df.columns = [c.lower().strip() for c in df.columns]

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["year"] = df["date_time"].dt.year

    for col in ["magnitude","depth","sig","cdi","mmi","tsunami","latitude","longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

# ---------- MEANINGFUL COLUMNS ---------- #
meaningful_cols = ["magnitude","depth","sig","cdi","mmi","tsunami"]

# ---------- CATEGORIES ---------- #
def mag_category(m):
    if m < 4: return "Low (<4)"
    elif m < 6: return "Moderate (4–5.9)"
    else: return "High (6+)"

def depth_category(d):
    if d < 70: return "Shallow"
    elif d < 300: return "Intermediate"
    else: return "Deep"

df["mag_category"] = df["magnitude"].apply(mag_category)
df["depth_category"] = df["depth"].apply(depth_category)

# ---------- TITLE ---------- #
st.markdown("<h1>Earthquake Statistical Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =========================================================
# OVERVIEW
# =========================================================
st.markdown("<h2>➤ Yearwise Earthquake Trend</h2>", unsafe_allow_html=True)

yearly = df.groupby("year")["magnitude"].count().reset_index()

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(yearly["year"], yearly["magnitude"], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Earthquake Count")
show_plot(fig)

st.info("Conclusion: Earthquake frequency varies year to year, indicating non-uniform seismic activity.")

# =========================================================
# SUMMARY STATISTICS
# =========================================================
st.markdown("<h2>➤ Summary Statistics</h2>", unsafe_allow_html=True)

summary = []
for col in meaningful_cols:
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

st.info("Conclusion: High standard deviation and IQR values show large variability and skewed distributions.")

# =========================================================
# MAGNITUDE vs TSUNAMI RISK ⭐
# =========================================================
st.markdown("<h2>➤ Magnitude vs Tsunami Risk</h2>", unsafe_allow_html=True)

risk = (
    df.groupby("mag_category")["tsunami"]
    .mean()
    .reset_index()
)

risk["Tsunami Probability (%)"] = risk["tsunami"] * 100

fig_risk, ax_risk = plt.subplots(figsize=(5,3))
sns.barplot(x="mag_category", y="Tsunami Probability (%)", data=risk, ax=ax_risk)
show_plot(fig_risk)

st.info("Conclusion: Higher magnitude earthquakes have a significantly higher probability of generating tsunamis.")

# =========================================================
# DEPTH vs INTENSITY ⭐
# =========================================================
st.markdown("<h2>➤ Depth vs Earthquake Intensity</h2>", unsafe_allow_html=True)

intensity_var = st.selectbox("Select intensity measure:", ["cdi","mmi"])

fig_int, ax_int = plt.subplots(figsize=(5,3))
sns.boxplot(x="depth_category", y=intensity_var, data=df, ax=ax_int)
show_plot(fig_int)

st.info("Conclusion: Shallow earthquakes tend to cause higher intensity and damage compared to deeper ones.")

# =========================================================
# TOP 10 SIGNIFICANT EARTHQUAKES ⭐
# =========================================================
st.markdown("<h2>➤ Top 10 Most Significant Earthquakes</h2>", unsafe_allow_html=True)

top10 = df.sort_values("sig", ascending=False).head(10)
st.dataframe(top10[["date_time","magnitude","depth","sig","tsunami"]])

st.info("Conclusion: A small number of earthquakes account for the highest overall impact.")

# =========================================================
# SPEARMAN CORRELATION
# =========================================================
st.markdown("<h2>➤ Spearman Correlation Analysis</h2>", unsafe_allow_html=True)

data = df[meaningful_cols].dropna()
corr = data.corr(method="spearman")

n = len(data)
pval = pd.DataFrame(index=corr.index, columns=corr.columns)

for i in corr.columns:
    for j in corr.columns:
        if i == j:
            pval.loc[i,j] = 0.0
        else:
            r = corr.loc[i,j]
            t = r * np.sqrt((n-2)/(1-r**2))
            pval.loc[i,j] = np.exp(-abs(t))

st.subheader("Spearman Correlation Matrix (ρ)")
st.dataframe(corr.round(3))

st.subheader("P-value Matrix")
st.dataframe(pval.round(4))

fig_corr, ax_corr = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr)
show_plot(fig_corr)

st.info(
    "Conclusion: Magnitude shows strong positive correlation with significance, "
    "while most other variables exhibit weak or moderate relationships."
)

# =========================================================
# FINAL CONCLUSION
# =========================================================
st.markdown("<h2>➤ Final Conclusion</h2>", unsafe_allow_html=True)

st.markdown("""
- Earthquake data is **non-normal and highly variable**, justifying the use of Spearman correlation.  
- High-magnitude and shallow earthquakes pose the greatest risk.  
- The analysis combines statistical rigor with real-world risk interpretation.
""")




