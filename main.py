import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration
st.set_page_config(page_title="Black Friday Sale Analysis", layout="wide")

# Title
st.title("Black Friday Sale Data Analysis")

# File uploader
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload Black_Friday_Sale.csv", type=["csv"])

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    
    # Display data information
    with st.expander("Data Information"):
        st.write("**Data Info:**")
        st.write(df.info())
        st.write("**Data Description:**")
        st.dataframe(df.describe())

    # Step 2.1: Handle missing and duplicate values
    st.header("Step 2.1: Handle Missing and Duplicate Values")
    st.write(f"Number of records before handling: {df.shape[0]}")
    # Remove duplicates
    df = df.drop_duplicates()
    st.write(f"Number of records after removing duplicates: {df.shape[0]}")

    # Step 2.2: Remove outliers
    st.header("Step 2.2: Remove Outliers")
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    st.write(f"IQR Bounds for Purchase: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")

    initial_rows = df.shape[0]
    df = df[(df['Purchase'] >= lower_bound) & (df['Purchase'] <= upper_bound)]
    st.write(f"Records before outlier removal: {initial_rows}")
    st.write(f"Records after outlier removal: {df.shape[0]}")

    # Step 2.3: Handle dates and create new columns
    st.header("Step 2.3: Handle Dates and Create New Columns")
    df['Purchase_Date'] = pd.date_range(start='2023-11-01', periods=len(df), freq='min')
    df['Purchase_Weekday'] = df['Purchase_Date'].dt.day_name()
    df['Is_Weekend'] = df['Purchase_Weekday'].isin(['Saturday', 'Sunday']).astype(int)
    st.write("Columns created: 'Purchase_Date', 'Purchase_Weekday', 'Is_Weekend'")
    st.write("**Top 5 rows of new date columns:**")
    st.dataframe(df[['Purchase_Date', 'Purchase_Weekday', 'Is_Weekend']].head())

    # Step 2.4: Standardize data
    st.header("Step 2.4: Standardize Data")
    st.write(f"Number of records before standardization: {df.shape[0]}")

    # Encode categorical variables
    categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            st.warning(f"Column {col} not found, skipping encoding.")
    st.write("**Encoded categorical data (first 5 rows):**")
    st.dataframe(df[categorical_cols].head())

    # Standardize numerical features
    numerical_cols = ['Purchase', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(0))
    st.write("**Standardized numerical data (first 5 rows):**")
    st.dataframe(df[numerical_cols].head())
    st.write(f"Number of records after standardization: {df.shape[0]}")

    # Step 3.1: Basic statistics by groups
    st.header("Step 3.1: Basic Statistics of Purchase by Groups")
    
    # By Gender
    gender_stats = df.groupby('Gender')['Purchase'].agg(['mean', 'median', 'std']).round(4)
    st.write("**Statistics of Purchase by Gender (0 = Female, 1 = Male):**")
    st.dataframe(gender_stats)

    # By City_Category
    city_stats = df.groupby('City_Category')['Purchase'].agg(['mean', 'median', 'std']).round(4)
    st.write("**Statistics of Purchase by City Category (Encoded):**")
    st.dataframe(city_stats)

    # Visualizations
    st.header("Visualizations")

    # Bar chart: Average Purchase by Gender
    gender_means = df.groupby('Gender')['Purchase'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(gender_means['Gender'], gender_means['Purchase'], color=['pink', 'blue'])
    ax.set_title('Average Purchase by Gender (Standardized)')
    ax.set_xlabel('Gender (0 = Female, 1 = Male)')
    ax.set_ylabel('Average Purchase (Normalized Value)')
    ax.set_xticks(gender_means['Gender'])
    ax.set_xticklabels(['Female', 'Male'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(gender_means['Purchase']):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
    st.pyplot(fig)
    st.write("**Data used for bar chart:**")
    st.dataframe(gender_means)

    # Pie chart: Distribution of Customers by Gender
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(gender_counts['Count'], labels=['Female', 'Male'], colors=['pink', 'blue'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribution of Customer Numbers by Gender')
    ax.axis('equal')
    st.pyplot(fig)
    st.write("**Data used for pie chart:**")
    st.dataframe(gender_counts)

    # Line chart: Average Purchase by Age
    age_means = df.groupby('Age')['Purchase'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(age_means['Age'], age_means['Purchase'], marker='o', color='green', linestyle='-', linewidth=2, markersize=8)
    ax.set_title('Average Purchase Trend by Age (Normalized)')
    ax.set_xlabel('Age (Normalized)')
    ax.set_ylabel('Average Purchase (Normalized Value)')
    ax.grid(True)
    ax.set_xticks(age_means['Age'])
    for i, v in enumerate(age_means['Purchase']):
        ax.text(age_means['Age'][i], v + 0.01, f'{v:.2f}', ha='center')
    st.pyplot(fig)
    st.write("**Data used for line chart:**")
    st.dataframe(age_means)

    # Note: The histogram and boxplot sections in the original code use a randomly generated dataset.
    # We'll skip those as they don't align with the main dataset analysis.
    st.warning("Note: Histogram and boxplot visualizations from the original script are skipped as they use a randomly generated dataset unrelated to the main analysis.")
else:
    st.info("Please upload a CSV file to start the analysis.")
