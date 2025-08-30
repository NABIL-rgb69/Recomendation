import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


df = pd.read_csv("Netflix_movies_and_tv_shows_clustering (1).csv")
df_movies = df[df['type'] == 'Movie'].copy()
df_movies['genres_str'] = df_movies['listed_in']


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_movies['genres_str'].fillna(""))

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_movies['Cluster'] = kmeans.fit_predict(X) + 1  # biar 1–6


terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

cluster_labels = {}
for i in range(6):
    top_genres = [terms[ind] for ind in order_centroids[i, :5]]  # ambil 5 genre teratas
    cluster_labels[i+1] = ", ".join(top_genres)

df_movies['Cluster_Label'] = df_movies['Cluster'].map(cluster_labels)


st.title("🎬 Rekomendasi Film Netflix (K-Means Clustering)")

st.markdown("### 📊 Informasi Cluster (Genre Dominan per Cluster)")
st.table(pd.DataFrame.from_dict(cluster_labels, orient='index', columns=['Top Genres']).rename_axis("Cluster"))


user = st.text_input("👤 Masukkan nama user")
film = st.selectbox("🎬 Pilih film:", sorted(df_movies['title'].dropna().unique()))
jumlah = st.number_input("🔢 Jumlah rekomendasi:", 1, 50, 10)

if st.button("✨ Tampilkan Rekomendasi"):
    if not user:
        st.warning("⚠️ Harap masukkan nama user terlebih dahulu.")
    else:
        cluster_pilihan = df_movies.loc[df_movies['title'] == film, 'Cluster'].values[0]
        cluster_label = df_movies.loc[df_movies['title'] == film, 'Cluster_Label'].values[0]
        genre_pilihan = df_movies.loc[df_movies['title'] == film, 'genres_str'].values[0]

        st.write(f"👤 **User**: {user}")
        st.write(f"🎬 **Film yang dipilih**: {film}")
        st.write(f"🎭 **Genre**: {genre_pilihan}")
        st.write(f"📊 **Cluster Genre**: {cluster_pilihan} ({cluster_label})")

        st.markdown("### ✅ Rekomendasi Film Berdasarkan Cluster yang Sama")

        rekom = df_movies[(df_movies['Cluster'] == cluster_pilihan) & (df_movies['title'] != film)]
        rekom = rekom.sample(n=min(jumlah, len(rekom)), random_state=None)

        if rekom.empty:
            st.warning("⚠️ Tidak ada rekomendasi lain dengan cluster yang sama.")
        else:
            st.dataframe(rekom[['title', 'genres_str', 'release_year', 'duration', 'Cluster', 'Cluster_Label']])
