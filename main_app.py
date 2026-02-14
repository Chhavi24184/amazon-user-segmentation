import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import altair as alt
import utils
import config


# CONFIGURATION

st.set_page_config(page_title="Amazon User Segmentation", layout="wide")


df = utils.load_or_create_data(config.DATA_FILE)

# PREPROCESSING

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


# SIDEBAR FILTERS

st.sidebar.header("Filter Options")

selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    options=le_category.classes_,
    default=list(le_category.classes_)
)

category_indices = [i for i, c in enumerate(le_category.classes_) if c in selected_categories]
algo = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "GMM", "Agglomerative", "DBSCAN"])
if algo in ["KMeans", "GMM", "Agglomerative"]:
    num_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
else:
    eps = st.sidebar.slider("DBSCAN eps", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
    min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=3, max_value=20, value=5)

if algo == "KMeans":
    model = KMeans(n_clusters=num_clusters, random_state=config.RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
elif algo == "GMM":
    model = GaussianMixture(n_components=num_clusters, random_state=config.RANDOM_STATE)
    labels = model.fit(X_scaled).predict(X_scaled)
elif algo == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=num_clusters)
    labels = model.fit_predict(X_scaled)
else:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

df['Cluster'] = labels
filtered_df = df[df['Product_Category'].isin(category_indices)]

if len(set(labels)) >= 2:
    try:
        s = silhouette_score(X_scaled, labels)
        st.sidebar.metric("Silhouette Score", f"{s:.3f}")
    except Exception:
        pass


# CLUSTER SUMMARY

st.title("🛒 Amazon User Segmentation Dashboard")
st.markdown("Analyze user purchase behavior using clustering and PCA visualization.")

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


# PCA VISUALIZATION

st.subheader("🎨 2D Cluster Visualization (PCA)")
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
vis_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
vis_df['Cluster'] = df['Cluster']
vis_df['User_ID'] = df['User_ID']

chart = (
    alt.Chart(vis_df)
    .mark_circle(size=90)
    .encode(
        x='PC1',
        y='PC2',
        color='Cluster:N',
        tooltip=['PC1', 'PC2', 'Cluster', 'User_ID']
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

if algo == "KMeans":
    show_validation = st.sidebar.checkbox("Show KMeans validation", value=False)
    if show_validation:
        ks = list(range(2, 11))
        inertias = []
        silhouettes = []
        for k in ks:
            m = KMeans(n_clusters=k, random_state=config.RANDOM_STATE)
            lab = m.fit_predict(X_scaled)
            inertias.append(m.inertia_)
            if len(set(lab)) >= 2:
                try:
                    silhouettes.append(silhouette_score(X_scaled, lab))
                except Exception:
                    silhouettes.append(np.nan)
            else:
                silhouettes.append(np.nan)
        val_df = pd.DataFrame({"K": ks, "Inertia": inertias, "Silhouette": silhouettes})
        inertia_chart = alt.Chart(val_df).mark_line(point=True).encode(x="K:Q", y="Inertia:Q")
        silhouette_chart = alt.Chart(val_df).mark_line(point=True).encode(x="K:Q", y="Silhouette:Q")
        st.subheader("📉 Elbow (Inertia)")
        st.altair_chart(inertia_chart, use_container_width=True)
        st.subheader("📈 Silhouette vs K")
        st.altair_chart(silhouette_chart, use_container_width=True)

if algo == "GMM":
    show_gmm_validation = st.sidebar.checkbox("Show GMM validation", value=False)
    if show_gmm_validation:
        ks = list(range(2, 11))
        bics = []
        for k in ks:
            gm = GaussianMixture(n_components=k, random_state=config.RANDOM_STATE)
            gm.fit(X_scaled)
            bics.append(gm.bic(X_scaled))
        gmm_df = pd.DataFrame({"Components": ks, "BIC": bics})
        bic_chart = alt.Chart(gmm_df).mark_line(point=True).encode(x="Components:Q", y="BIC:Q")
        st.subheader("📉 GMM BIC vs Components")
        st.altair_chart(bic_chart, use_container_width=True)

if algo == "DBSCAN":
    n_clusters = len([l for l in set(labels) if l != -1])
    noise_ratio = float((labels == -1).mean())
    st.sidebar.metric("DBSCAN clusters", f"{n_clusters}")
    st.sidebar.metric("Noise ratio", f"{noise_ratio:.2f}")

# USER SEARCH

st.subheader("🔍 Search User Profile")
user_id = st.number_input("Enter User ID to view details", min_value=1, max_value=df['User_ID'].max(), step=1)

user_profile = df[df['User_ID'] == user_id]
if not user_profile.empty:
    st.dataframe(user_profile)
    st.success(f" User belongs to Cluster: {user_profile['Cluster'].values[0]}")
else:
    st.warning("No user found with this ID.")


# CLUSTER-WISE PRODUCT INSIGHTS

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

# DOWNLOAD FILTERED DATA

st.subheader("⬇️ Download Cluster Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered CSV",
    data=csv,
    file_name="amazon_user_clusters.csv",
    mime="text/csv"
)

save_segments = st.button("Save Segments CSV")
if save_segments:
    df.to_csv(config.OUTPUT_FILE, index=False)
    st.success(f" Saved to {config.OUTPUT_FILE}")

# CLUSTER FEATURE PROFILES
st.subheader("🧭 Cluster Feature Profiles")
scaled_df = pd.DataFrame(X_scaled, columns=features)
profile_features = [f for f in features if f != 'Product_Category']
scaled_df['Cluster'] = df['Cluster']
cluster_profile = (
    scaled_df.groupby('Cluster')[profile_features]
    .mean()
    .reset_index()
)
profile_long = cluster_profile.melt(id_vars='Cluster', var_name='Feature', value_name='Value')
profile_chart = (
    alt.Chart(profile_long)
    .mark_line(point=True)
    .encode(
        x='Feature:N',
        y='Value:Q',
        color='Cluster:N',
        tooltip=['Feature', 'Value', 'Cluster']
    )
    .interactive()
)
st.altair_chart(profile_chart, use_container_width=True)

st.subheader("📑 Cluster Descriptions")
raw_profile = (
    df.groupby('Cluster')[profile_features]
    .mean()
    .reset_index()
)
scaled_map = cluster_profile.set_index('Cluster')
counts_map = df['Cluster'].value_counts().to_dict()
label_t = st.sidebar.slider("Label sensitivity", min_value=0.2, max_value=1.0, value=0.5, step=0.05)
def _label_from_scaled(r, t):
    tags = []
    if r.get('Avg_Spending', 0) > t:
        tags.append('High spenders')
    elif r.get('Avg_Spending', 0) < -t:
        tags.append('Low spenders')
    if r.get('Annual_Income', 0) > t:
        tags.append('High income')
    elif r.get('Annual_Income', 0) < -t:
        tags.append('Low income')
    if r.get('Purchase_Frequency', 0) > t:
        tags.append('Frequent buyers')
    elif r.get('Purchase_Frequency', 0) < -t:
        tags.append('Infrequent buyers')
    if r.get('Review_Score', 0) > t:
        tags.append('Positive reviewers')
    elif r.get('Review_Score', 0) < -t:
        tags.append('Critical reviewers')
    if r.get('Age', 0) > t:
        tags.append('Older users')
    elif r.get('Age', 0) < -t:
        tags.append('Younger users')
    return ', '.join(tags) if tags else 'Mixed profile'
summary_df = raw_profile.copy()
summary_df['Num_Users'] = summary_df['Cluster'].map(lambda c: counts_map.get(c, 0))
summary_df['Label'] = summary_df['Cluster'].map(lambda c: _label_from_scaled(scaled_map.loc[c], label_t))
st.dataframe(summary_df.round(2))
summary_csv = summary_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Cluster Summary CSV", data=summary_csv, file_name="amazon_cluster_summary.csv", mime="text/csv")
save_labeled_segments = st.button("Save Labeled Segments CSV")
if save_labeled_segments:
    label_map = {c: _label_from_scaled(scaled_map.loc[c], label_t) for c in scaled_map.index}
    labeled_df = df.copy()
    labeled_df['Cluster_Label'] = labeled_df['Cluster'].map(lambda c: label_map.get(c, ''))
    labeled_path = "amazon_user_segments_labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)
    st.success(f" Saved to {labeled_path}")

st.markdown("---")
st.caption("Developed by **Chhavi & Vansh ** ")
