# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk

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

# ---------- HELPER ---------- #
def show_plot(fig):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ----------------- LOAD DATA ----------------- #
@st.cache_data
def load_data(path="earthquake_1995-2023.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    df["date_time"] = pd.to_datetime(df.get("date_time"), errors="coerce")
    df["magnitude"] = pd.to_numeric(df.get("magnitude"), errors="coerce")
    df["depth"] = pd.to_numeric(df.get("depth"), errors="coerce")
    df["tsunami"] = pd.to_numeric(df.get("tsunami"), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

    for col in ["cdi", "mmi", "sig"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = df["date_time"].dt.year
    return df

df = load_data()

# ---------- MEANINGFUL COLUMNS ---------- #
meaningful_cols = [
    c for c in ["magnitude", "depth", "sig", "cdi", "mmi", "tsunami"]
    if c in df.columns and not df[c].isna().all()
]

# ---------- REGION CLASSIFICATION ---------- #
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
    return "Unknown"

df["region"] = df.apply(lambda r: classify_region(r["latitude"], r["longitude"]), axis=1)

# ---------- ALERT COLOR ---------- #
def alert_color(m):
    if m < 4:
        return [0, 200, 0]      # green
    elif m < 6:
        return [255, 165, 0]    # orange
    else:
        return [255, 0, 0]      # red

df["color"] = df["magnitude"].apply(alert_color)

# ---------- TITLE ---------- #
st.markdown("<h1>Earthquake & Tsunami Analysis Dashboard (1995–2023)</h1>", unsafe_allow_html=True)

# =========================================================
# OVERVIEW DASHBOARD
# =========================================================
st.markdown("<h2>➤ Overview Dashboards</h2>", unsafe_allow_html=True)

yearly = (
    df.groupby("year")
    .agg(
        earthquake_count=("magnitude", "count"),
        tsunami_count=("tsunami", lambda x: (x == 1).sum())
    )
    .dropna()
    .reset_index()
)

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

show_plot(fig)

# =========================================================
# REGION ANALYSIS + MAP
# =========================================================
st.markdown("<h2>➤ Region-Based Analysis</h2>", unsafe_allow_html=True)

regions = sorted(df["region"].unique())
selected_region = st.selectbox("Select region:", regions)

df_r = df[df["region"] == selected_region]

st.markdown(f"<h3>{selected_region}: {len(df_r)} earthquakes</h3>", unsafe_allow_html=True)

# ---- MAP ---- #
st.subheader("Earthquake Alert Map")

map_data = df_r.dropna(subset=["latitude", "longitude", "magnitude"])

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position=["longitude", "latitude"],
    get_fill_color="color",
    get_radius=20000,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=map_data["latitude"].mean(),
    longitude=map_data["longitude"].mean(),
    zoom=2,
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

st.caption("🟢 Low | 🟠 Moderate | 🔴 High magnitude earthquakes")

# =========================================================
# SUMMARY STATISTICS
# =========================================================
st.markdown("<h2>➤ Summary Statistics</h2>", unsafe_allow_html=True)

summary = []
for col in meaningful_cols:
    s = df[col].dropna()
    summary.append({
        "Column": col,
        "Mean": s.mean(),
        "Median": s.median(),
        "Q1": s.quantile(0.25),
        "Q3": s.quantile(0.75),
        "IQR": s.quantile(0.75) - s.quantile(0.25),
    })

st.dataframe(pd.DataFrame(summary).set_index("Column"))



