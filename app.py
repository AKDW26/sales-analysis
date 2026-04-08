import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

st.set_page_config(page_title="Diwali Sales Dashboard", layout="wide")

# ------------------ TITLE ------------------
st.title("🪔 Diwali Sales Interactive Dashboard")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # ------------------ CLEANING ------------------
    df = df.drop(["Status", "unnamed1"], axis=1, errors='ignore')
    df = df.drop_duplicates()
    df["User_ID"] = df["User_ID"].astype(str)

    # Missing values
    imputer = SimpleImputer(strategy='median')
    df["Amount"] = imputer.fit_transform(df[["Amount"]])

    # Outlier handling
    winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Amount'])
    df['Amount'] = winsor.fit_transform(df[['Amount']])

    # ------------------ SIDEBAR FILTERS ------------------
    st.sidebar.header("🔍 Filter Data")

    states = st.sidebar.multiselect("Select State", df['State'].unique(), default=df['State'].unique())
    gender = st.sidebar.multiselect("Select Gender", df['Gender'].unique(), default=df['Gender'].unique())
    occupation = st.sidebar.multiselect("Select Occupation", df['Occupation'].unique(), default=df['Occupation'].unique())

    filtered_df = df[
        (df['State'].isin(states)) &
        (df['Gender'].isin(gender)) &
        (df['Occupation'].isin(occupation))
    ]

    # ------------------ KPIs ------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{int(filtered_df['Amount'].sum()):,}")
    col2.metric("Average Sales", f"{int(filtered_df['Amount'].mean()):,}")
    col3.metric("Total Orders", filtered_df.shape[0])

    # ------------------ CHART SELECTION ------------------
    st.subheader("📈 Dynamic Analysis")

    chart_option = st.selectbox(
        "Select Analysis Type",
        ["State-wise Sales", "Gender vs Sales", "Occupation Sales", "Product Category"]
    )

    fig, ax = plt.subplots()

    if chart_option == "State-wise Sales":
        data = filtered_df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10)
        data.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)

    elif chart_option == "Gender vs Sales":
        data = filtered_df.groupby('Gender')['Amount'].sum()
        data.plot(kind='bar', ax=ax)

    elif chart_option == "Occupation Sales":
        data = filtered_df.groupby('Occupation')['Amount'].sum().sort_values(ascending=False)
        data.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)

    elif chart_option == "Product Category":
        data = filtered_df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False)
        data.plot(kind='bar', ax=ax)
        plt.xticks(rotation=90)

    st.pyplot(fig)

    # ------------------ RAW DATA ------------------
    with st.expander("📄 View Raw Data"):
        st.write(filtered_df)

    # ------------------ DOWNLOAD BUTTON ------------------
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

    # ------------------ INSIGHTS ------------------
    st.subheader("🎯 Key Insights")

    top_state = filtered_df.groupby('State')['Amount'].sum().idxmax()
    top_product = filtered_df.groupby('Product_Category')['Amount'].sum().idxmax()

    st.success(f"Top performing state: {top_state}")
    st.success(f"Most sold product category: {top_product}")

else:
    st.info("Please upload your dataset to start analysis.")