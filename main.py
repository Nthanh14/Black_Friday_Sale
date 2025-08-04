import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Streamlit app title
st.title("Black Friday Sale Analysis Dashboard")
data_url = " https://raw.githubusercontent.com/Nthanh14/Black_Friday_Sale/refs/heads/main/Black_Friday_Sale.csv "

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)

    # Data Preprocessing
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    df = df.drop_duplicates()
    
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age'] = df['Age'].map(age_map)
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace('4+', 4).astype(int)
    df = pd.get_dummies(df, columns=['City_Category'], prefix='City')

    # Data Overview
    st.header("Data Overview")
    st.write("Data Information:")
    st.write(df.info())
    st.write("Data Description:")
    st.write(df.describe())
    st.write(f"Number of records after preprocessing: {len(df)}")
    st.write("Null values after preprocessing:")
    st.write(df.isnull().sum())

    # Summary Statistics
    st.header("Summary Statistics")
    st.write("Average Purchase by Gender:")
    st.write(df.groupby('Gender')['Purchase'].mean())
    st.write("Average Purchase by Age:")
    st.write(df.groupby('Age')['Purchase'].mean())
    st.write("Total Purchase by City Category:")
    st.write(df[['City_A', 'City_B', 'City_C']].multiply(df['Purchase'], axis=0).sum())

    # Visualizations
    st.header("Visualizations")
    
    # Bar Chart - Average Purchase by Gender
    st.subheader("Average Purchase by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    gender_purchase = df.groupby('Gender')['Purchase'].mean()
    sns.barplot(x=gender_purchase.index.map({0: 'Female', 1: 'Male'}), y=gender_purchase.values, ax=ax)
    ax.set_title('Average Purchase by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Average Purchase (K$)')
    st.pyplot(fig)

    # Pie Chart - Distribution of Customer Numbers by Gender
    st.subheader("Distribution of Customer Numbers by Gender")
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(gender_counts['Count'], labels=['Female', 'Male'], colors=['pink', 'blue'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribution of Customer Numbers by Gender')
    ax.axis('equal')
    st.pyplot(fig)

    # Bar Chart - Total Purchase by City Category
    st.subheader("Total Purchase by City Category")
    fig, ax = plt.subplots(figsize=(8, 6))
    city_purchase = df[['City_A', 'City_B', 'City_C']].multiply(df['Purchase'], axis=0).sum() / 1000
    sns.barplot(x=city_purchase.index, y=city_purchase.values, ax=ax)
    ax.set_title('Total Purchase by City Category')
    ax.set_xlabel('City Category')
    ax.set_ylabel('Total Purchase (K$)')
    st.pyplot(fig)

    # Pie Chart - Product Category 1 Distribution
    st.subheader("Product Category 1 Distribution")
    product_cat1_counts = df['Product_Category_1'].value_counts()
    product_cat1_counts = product_cat1_counts[product_cat1_counts > 5000]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(product_cat1_counts, labels=[f'Category {x}' for x in product_cat1_counts.index], autopct='%1.1f%%', startangle=140)
    ax.set_title('Product Category 1 Distribution')
    st.pyplot(fig)

    # Line Chart - Total Purchase by Stay in Current City
    st.subheader("Total Purchase by Years in Current City")
    fig, ax = plt.subplots(figsize=(8, 6))
    stay_purchase = df.groupby('Stay_In_Current_City_Years')['Purchase'].sum() / 1000
    ax.plot(stay_purchase.index, stay_purchase.values, marker='o')
    ax.set_title('Total Purchase by Years in Current City')
    ax.set_xlabel('Years in Current City')
    ax.set_ylabel('Total Purchase (K$)')
    st.pyplot(fig)

    # Clustering
    st.header("Customer Segmentation (K-Means Clustering)")
    features = ['Gender', 'Age', 'Occupation', 'Stay_In_Current_City_Years',
                'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3',
                'Purchase', 'City_A', 'City_B', 'City_C']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    st.write("First few rows with cluster assignments:")
    st.write(df[['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
                 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                 'Product_Category_2', 'Product_Category_3', 'Purchase', 'Cluster']].head())
    st.write(f"WCSS (Sum of squares within cluster): {kmeans.inertia_ * 0.001:.2f}")

    # Linear Regression
    st.header("Purchase Prediction (Linear Regression)")
    X = df.drop(['Purchase', 'User_ID', 'Product_ID'], axis=1)
    y = df['Purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared: {r2:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

else:
    st.write("Please upload a CSV file to start the analysis.")
