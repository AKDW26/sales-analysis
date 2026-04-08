import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import requests

from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

st.set_page_config(page_title="Diwali Sales Dashboard", layout="wide")

st.title("🪔 Diwali Sales Dashboard")

# ------------------ LOAD DATA ------------------
file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file, encoding='latin-1')

    # ------------------ CLEANING ------------------
    df = df.drop(["Status", "unnamed1"], axis=1, errors='ignore')
    df = df.drop_duplicates()
    df["User_ID"] = df["User_ID"].astype(str)

    imputer = SimpleImputer(strategy='median')
    df["Amount"] = imputer.fit_transform(df[["Amount"]])

    winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Amount'])
    df['Amount'] = winsor.fit_transform(df[['Amount']])

    # ------------------ FILTERS ------------------
    st.sidebar.header("Filters")

    states = st.sidebar.multiselect("State", df['State'].unique(), df['State'].unique())
    gender = st.sidebar.multiselect("Gender", df['Gender'].unique(), df['Gender'].unique())

    df = df[(df['State'].isin(states)) & (df['Gender'].isin(gender))]

    # ------------------ KPIs ------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{int(df['Amount'].sum()):,}")
    col2.metric("Average Sales", f"{int(df['Amount'].mean()):,}")
    col3.metric("Orders", df.shape[0])

    # ------------------ 🇮🇳 INDIA MAP ------------------
    st.subheader("🇮🇳 India State-wise Sales")

    state_sales = df.groupby('State')['Amount'].sum().reset_index()

    # Load India GeoJSON
    geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    geojson = requests.get(geojson_url).json()

    # IMPORTANT: State names must match GeoJSON
    fig_map = px.choropleth(
        state_sales,
        geojson=geojson,
        locations='State',
        featureidkey="properties.NAME_1",
        color='Amount',
        color_continuous_scale="Blues",
        title="Sales by State"
    )

    fig_map.update_geos(fitbounds="locations", visible=False)

    st.plotly_chart(fig_map, use_container_width=True)

    # ------------------ 📊 INTERACTIVE CHART ------------------
    st.subheader("📊 Analysis")

    option = st.selectbox(
        "Select Analysis",
        ["State Sales", "Gender Sales", "Occupation Sales", "Product Category"]
    )

    if option == "State Sales":
        data = df.groupby('State')['Amount'].sum().reset_index()
        fig = px.bar(data, x='State', y='Amount', color='Amount')

    elif option == "Gender Sales":
        data = df.groupby('Gender')['Amount'].sum().reset_index()
        fig = px.pie(data, names='Gender', values='Amount')

    elif option == "Occupation Sales":
        data = df.groupby('Occupation')['Amount'].sum().reset_index()
        fig = px.bar(data, x='Occupation', y='Amount', color='Amount')

    else:
        data = df.groupby('Product_Category')['Amount'].sum().reset_index()
        fig = px.bar(data, x='Product_Category', y='Amount', color='Amount')

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ 🤖 AI INSIGHTS ------------------
    st.subheader("AI Insights")

    top_state = state_sales.sort_values(by="Amount", ascending=False).iloc[0]['State']
    top_product = df.groupby('Product_Category')['Amount'].sum().idxmax()
    top_gender = df.groupby('Gender')['Amount'].sum().idxmax()

    st.success(f"Top State: {top_state}")
    st.info(f"Top Product Category: {top_product}")
    st.warning(f"Top Buyers: {top_gender}")

    # ------------------ DOWNLOAD ------------------
    st.download_button(
        "Download Cleaned Data",
        df.to_csv(index=False),
        "cleaned_data.csv"
    )

else:
    st.info("Upload dataset to begin")
