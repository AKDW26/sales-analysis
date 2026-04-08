import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

from sklearn.impute import SimpleImputer

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Diwali Sales Dashboard", layout="wide")

st.title("🪔 Diwali Sales Dashboard")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file, encoding='latin-1')

    # ------------------ DATA CLEANING ------------------
    df = df.drop(["Status", "unnamed1"], axis=1, errors='ignore')
    df = df.drop_duplicates()
    df["User_ID"] = df["User_ID"].astype(str)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df["Amount"] = imputer.fit_transform(df[["Amount"]])

    # ------------------ OUTLIER HANDLING (SAFE METHOD) ------------------
    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["Amount"] = np.clip(df["Amount"], lower, upper)

    # ------------------ SIDEBAR FILTERS ------------------
    st.sidebar.header("🔍 Filters")

    states = st.sidebar.multiselect("State", df['State'].unique(), df['State'].unique())
    gender = st.sidebar.multiselect("Gender", df['Gender'].unique(), df['Gender'].unique())

    filtered_df = df[
        (df['State'].isin(states)) &
        (df['Gender'].isin(gender))
    ]

    # ------------------ KPIs ------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{int(filtered_df['Amount'].sum()):,}")
    col2.metric("Average Sales", f"{int(filtered_df['Amount'].mean()):,}")
    col3.metric("Total Orders", filtered_df.shape[0])

    # ------------------ INDIA MAP ------------------
    st.subheader("🇮🇳 State-wise Sales Map")

    state_sales = filtered_df.groupby('State')['Amount'].sum().reset_index()

    # Load GeoJSON
    geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    geojson = requests.get(geojson_url).json()

    fig_map = px.choropleth(
        state_sales,
        geojson=geojson,
        locations='State',
        featureidkey="properties.NAME_1",
        color='Amount',
        color_continuous_scale="Blues"
    )

    fig_map.update_geos(fitbounds="locations", visible=False)

    st.plotly_chart(fig_map, use_container_width=True)

    # ------------------ INTERACTIVE CHARTS ------------------
    st.subheader("📈 Interactive Analysis")

    option = st.selectbox(
        "Select Analysis",
        ["State Sales", "Gender Sales", "Occupation Sales", "Product Category"]
    )

    if option == "State Sales":
        data = filtered_df.groupby('State')['Amount'].sum().reset_index()
        fig = px.bar(data, x='State', y='Amount', color='Amount')

    elif option == "Gender Sales":
        data = filtered_df.groupby('Gender')['Amount'].sum().reset_index()
        fig = px.pie(data, names='Gender', values='Amount')

    elif option == "Occupation Sales":
        data = filtered_df.groupby('Occupation')['Amount'].sum().reset_index()
        fig = px.bar(data, x='Occupation', y='Amount', color='Amount')

    else:
        data = filtered_df.groupby('Product_Category')['Amount'].sum().reset_index()
        fig = px.bar(data, x='Product_Category', y='Amount', color='Amount')

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ AI INSIGHTS ------------------
    st.subheader("🤖 AI Insights")

    if not state_sales.empty:
        top_state = state_sales.sort_values(by="Amount", ascending=False).iloc[0]['State']
        top_product = filtered_df.groupby('Product_Category')['Amount'].sum().idxmax()
        top_gender = filtered_df.groupby('Gender')['Amount'].sum().idxmax()

        st.success(f"Top performing state: {top_state}")
        st.info(f"Most popular product category: {top_product}")
        st.warning(f"Major buyers: {top_gender}")

        st.subheader("💡 Business Recommendations")
        st.write(f"""
        - Focus marketing campaigns in **{top_state}**
        - Increase stock for **{top_product}**
        - Target ads towards **{top_gender}**
        """)

    # ------------------ DATA VIEW ------------------
    with st.expander("📄 View Data"):
        st.dataframe(filtered_df)

    # ------------------ DOWNLOAD ------------------
    st.download_button(
        "📥 Download Cleaned Data",
        filtered_df.to_csv(index=False),
        "cleaned_data.csv",
        "text/csv"
    )

else:
    st.info("Please upload your dataset to start 🚀")
