import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import altair as alt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="Amazon User Segmentation", layout="wide")

# ------------------------------------------------------------
# DATA GENERATION
# ------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n=300, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'User_ID': range(1, n + 1),
        'Age': np.random.randint(18, 70, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Annual_Income': np.random.randint(15000, 200000, n),
        'Purchase_Frequency': np.random.randint(1, 30, n),
        'Avg_Spending': np.random.randint(100, 25000, n),
        'Product_Category': np.random.choice(['Electronics', 'Fashion', 'Home', 'Books', 'Sports'], n),
        'Review_Score': np.random.randint(1, 6, n)
    })
    return df


df = generate_synthetic_data()

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    le_gender = LabelEncoder()
    le_category = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Product_Category'] = le_category.fit_transform(df['Product_Category'])

    features = [
        'Age', 'Gender', 'Annual_Income', 'Purchase_Frequency',
        'Avg_Spending', 'Product_Category', 'Review_Score'
    ]

    X = df[features].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, features, le_category


df, X_scaled, features, le_category = preprocess(df)

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filter Options")

selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    options=le_category.classes_,
    default=list(le_category.classes_)
)

category_indices = [i for i, c in enumerate(le_category.classes_) if c in selected_categories]
filtered_df = df[df['Product_Category'].isin(category_indices)]

num_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

# ------------------------------------------------------------
# KMEANS CLUSTERING
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
filtered_df = df[df['Product_Category'].isin(category_indices)]

# ------------------------------------------------------------
# CLUSTER SUMMARY
# ------------------------------------------------------------
st.title("🛒 Amazon User Segmentation Dashboard")
st.markdown("Analyze user purchase behavior using **K-Means Clustering & PCA Visualization**.")

st.subheader("📊 Cluster Summary")
summary = (
    filtered_df.groupby('Cluster')
    .agg({
        'User_ID': 'count',
        'Age': 'mean',
        'Annual_Income': 'mean',
        'Purchase_Frequency': 'mean',
        'Avg_Spending': 'mean',
        'Review_Score': 'mean'
    })
    .round(2)
    .rename(columns={'User_ID': 'Num_Users'})
)

st.dataframe(summary)
st.bar_chart(summary['Num_Users'])

# ------------------------------------------------------------
# PCA VISUALIZATION
# ------------------------------------------------------------
st.subheader("🎨 2D Cluster Visualization (PCA)")
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
vis_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
vis_df['Cluster'] = df['Cluster']

chart = (
    alt.Chart(vis_df)
    .mark_circle(size=90)
    .encode(
        x='PC1',
        y='PC2',
        color='Cluster:N',
        tooltip=['PC1', 'PC2', 'Cluster']
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

# ------------------------------------------------------------
# USER SEARCH
# ------------------------------------------------------------
st.subheader("🔍 Search User Profile")
user_id = st.number_input("Enter User ID to view details", min_value=1, max_value=df['User_ID'].max(), step=1)

user_profile = df[df['User_ID'] == user_id]
if not user_profile.empty:
    st.dataframe(user_profile)
    st.success(f"✅ User belongs to Cluster: {user_profile['Cluster'].values[0]}")
else:
    st.warning("No user found with this ID.")

# ------------------------------------------------------------
# CLUSTER-WISE PRODUCT INSIGHTS
# ------------------------------------------------------------
st.subheader("🧩 Cluster-wise Product Insights")
selected_cluster = st.selectbox(
    "Select Cluster",
    options=sorted(filtered_df['Cluster'].unique())
)

cluster_data = filtered_df[filtered_df['Cluster'] == selected_cluster]
product_summary = (
    cluster_data['Product_Category']
    .value_counts()
    .rename_axis('Product_Category')
    .reset_index(name='Count')
)

product_summary['Product_Category'] = product_summary['Product_Category'].apply(
    lambda x: le_category.inverse_transform([x])[0]
)

st.dataframe(product_summary)

# ------------------------------------------------------------
# DOWNLOAD FILTERED DATA
# ------------------------------------------------------------
st.subheader("⬇️ Download Cluster Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered CSV",
    data=csv,
    file_name="amazon_user_clusters.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Developed by **Chhavi ** 🧠")
