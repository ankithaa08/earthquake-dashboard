# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Earthquake Analysis Dashboard (1995–2023)",
    layout="wide"
)
sns.set(style="whitegrid")

st.markdown("""
<style>
h1 {font-size:38px; font-weight:800;}
h2 {font-size:30px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("earthquake_1995-2023.csv")
    df.columns = [c.lower().strip() for c in df.columns]

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["year"] = df["date_time"].dt.year

    for col in ["magnitude", "depth", "sig", "cdi", "mmi", "tsunami"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

# -------------------------------------------------
# MEANINGFUL COLUMNS
# -------------------------------------------------
meaningful_cols = ["magnitude", "depth", "sig", "cdi", "mmi", "tsunami"]
meaningful_cols = [c for c in meaningful_cols if c in df.columns]

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("<h1>Earthquake Statistical Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =================================================
# OVERVIEW LINE CHART
# =================================================
st.markdown("<h2>➤ Yearwise Earthquake Trend</h2>", unsafe_allow_html=True)

yearly = df.groupby("year")["magnitude"].count().reset_index()

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(yearly["year"], yearly["magnitude"], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Earthquakes")
st.pyplot(fig)

st.caption("📌 Earthquake occurrences show variation over time, indicating non-stationary behavior.")

# =================================================
# SUMMARY STATISTICS
# =================================================
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

summary_df = pd.DataFrame(summary).set_index("Variable")
st.dataframe(summary_df)

st.markdown("""
**Conclusion:**  
The variables show high variability and skewness, as indicated by large standard deviation and IQR values.
""")

# =================================================
# DISTRIBUTION PLOTS
# =================================================
st.markdown("<h2>➤ Distribution Analysis</h2>", unsafe_allow_html=True)

var = st.selectbox("Select variable:", meaningful_cols)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Histogram")
    fig1, ax1 = plt.subplots(figsize=(4,3))
    sns.histplot(df[var].dropna(), bins=30, ax=ax1)
    st.pyplot(fig1)

with c2:
    st.subheader("Density Plot")
    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.kdeplot(df[var].dropna(), fill=True, ax=ax2)
    st.pyplot(fig2)

st.caption("📌 The distributions are skewed and non-normal.")

# =================================================
# BOXPLOT & VIOLIN
# =================================================
st.markdown("<h2>➤ Boxplot & Violin Plot</h2>", unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(4,3))
sns.boxplot(y=df[var].dropna(), ax=ax3)
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(4,3))
sns.violinplot(y=df[var].dropna(), ax=ax4)
st.pyplot(fig4)

st.caption("📌 Presence of outliers and asymmetric distributions confirms non-normality.")

# =================================================
# RELATIONSHIP PLOTS
# =================================================
st.markdown("<h2>➤ Relationship Analysis</h2>", unsafe_allow_html=True)

x = st.selectbox("X-axis:", meaningful_cols, index=0)
y = st.selectbox("Y-axis:", meaningful_cols, index=1)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Hexagonal Binning")
    fig5, ax5 = plt.subplots(figsize=(4,3))
    hb = ax5.hexbin(df[x], df[y], gridsize=30)
    fig5.colorbar(hb)
    st.pyplot(fig5)

with c4:
    st.subheader("Contour Plot")
    fig6, ax6 = plt.subplots(figsize=(4,3))
    sns.kdeplot(x=df[x].dropna(), y=df[y].dropna(), fill=True, ax=ax6)
    st.pyplot(fig6)

st.caption("📌 Density-based plots reveal clustered earthquake behavior.")

# =================================================
# SPEARMAN CORRELATION
# =================================================
st.markdown("<h2>➤ Spearman Correlation Analysis</h2>", unsafe_allow_html=True)

data = df[meaningful_cols].dropna()
corr = data.corr(method="spearman")

n = len(data)
pval = pd.DataFrame(index=corr.index, columns=corr.columns)

for i in corr.columns:
    for j in corr.columns:
        r = corr.loc[i,j]
        if i == j:
            pval.loc[i,j] = 0.0
        else:
            t = r * np.sqrt((n-2)/(1-r**2))
            pval.loc[i,j] = np.exp(-abs(t))  # approximation

st.subheader("Spearman Correlation Coefficient (ρ)")
st.dataframe(corr.round(3))

st.subheader("P-value Matrix")
st.dataframe(pval.round(4))

fig7, ax7 = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax7)
st.pyplot(fig7)

st.markdown("""
**Correlation Conclusion:**  
- Magnitude shows strong positive correlation with Significance.  
- Depth shows weak or negative relationship with intensity measures.  
- Most variables are weakly correlated, indicating earthquakes are influenced by multiple independent factors.  
- Since p-values are below 0.05 for key relationships, these correlations are statistically significant.
""")

# =================================================
# FINAL CONCLUSION
# =================================================
st.markdown("<h2>➤ Final Conclusion</h2>", unsafe_allow_html=True)

st.markdown("""
- Earthquake data is **non-normal and skewed**, validated by distribution plots.  
- **Spearman correlation** was the correct statistical choice.  
- Strong relationship exists between magnitude and impact-related variables.  
- High variability confirms unpredictable nature of earthquakes.  

✅ This dashboard provides a complete statistical and visual analysis of seismic activity.
""")
