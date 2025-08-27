import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="🎬 Rekomendasi Film", layout="wide")


st.title("🎬 Sistem Rekomendasi Film (Clustering)")

uploaded_file = st.file_uploader("📂 Upload dataset CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    
    df['genres_str'] = df['listed_in']
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)

    st.success("✅ Dataset berhasil dimuat!")
    st.dataframe(df.head())

    
    if st.button("🔄 Generate Cluster"):
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
        genre_matrix = vectorizer.fit_transform(df['genres_str'].fillna(""))

        numerik = df[['duration_int', 'release_year']].fillna(0)
        gabung = np.hstack((genre_matrix.toarray(), numerik.values))

        scaler = StandardScaler()
        X = scaler.fit_transform(gabung)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X) + 1

        st.session_state['df_clustered'] = df
        st.success("✅ Cluster berhasil dibuat (1–4)!")

    
    if 'df_clustered' in st.session_state:
        df_clustered = st.session_state['df_clustered']

        user = st.text_input("👤 Masukkan nama user")
        film = st.selectbox("🎬 Pilih film:", sorted(df_clustered['title'].dropna().unique()))

        tampilan = st.radio("🎨 Pilih tampilan rekomendasi:",
                            ["Tabel", "List", "Kartu"], horizontal=True)

        if st.button("✨ Tampilkan Rekomendasi"):
            if 'Cluster' not in df_clustered.columns:
                st.warning("⚠️ Harap klik tombol 'Generate Cluster' terlebih dahulu.")
            else:
                c = df_clustered[df_clustered['title'] == film]['Cluster'].values[0]
                genre_pilihan = df_clustered[df_clustered['title'] == film]['genres_str'].values[0]

                st.write(f"👤 **User**: {user}")
                st.write(f"🎬 **Film yang dipilih**: {film}")
                st.write(f"📊 **Cluster**: {c} (Cluster 1–4)")
                st.write(f"🎭 **Genre**: {genre_pilihan}")

                st.markdown("### ✅ Rekomendasi Film")

                
                rekom = df_clustered[(df_clustered['Cluster'] == c) & (df_clustered['title'] != film)]

                
                rekom = rekom.sample(frac=1, random_state=np.random.randint(0,1000)).head(5)

               
                if tampilan == "Tabel":
                    st.dataframe(rekom[['title', 'genres_str', 'release_year', 'duration_int']])
                elif tampilan == "List":
                    for i, row in rekom.iterrows():
                        st.write(f"- 🎬 **{row['title']}** ({row['release_year']}) | 🎭 {row['genres_str']} | ⏱ {row['duration_int']} min")
                elif tampilan == "Kartu":
                    cols = st.columns(2)
                    for idx, row in enumerate(rekom.itertuples()):
                        with cols[idx % 2]:
                            st.markdown(f"""
                            <div style="background:#f5f5f5; padding:15px; margin:10px; border-radius:10px; box-shadow:2px 2px 6px #ccc">
                                <h4>🎬 {row.title}</h4>
                                <p><b>Genre:</b> {row.genres_str}</p>
                                <p><b>Tahun:</b> {row.release_year}</p>
                                <p><b>Durasi:</b> {row.duration_int} menit</p>
                            </div>
                            """, unsafe_allow_html=True)
else:
    st.info("⬆️ Silakan upload dataset CSV terlebih dahulu.")
