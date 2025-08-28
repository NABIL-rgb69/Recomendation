import streamlit as st
import pandas as pd


st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="wide")


st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-title {
        font-size: 45px;
        font-weight: 800;
        color: #E50914; /* Warna merah khas Netflix */
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #bbbbbb;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🍿 Rekomendasi Film Netflix</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Temukan film sesuai favoritmu 🎭</div>', unsafe_allow_html=True)


df = pd.read_csv("Netflix_movies_and_tv_shows_clustering (1).csv")
df = df[df['type'] == 'Movie'].copy()
df['genres_str'] = df['listed_in']

st.success("✅ Dataset berhasil dimuat (khusus Film)!")


user = st.text_input("👤 Masukkan nama user")
film = st.selectbox("🎬 Pilih film:", sorted(df['title'].dropna().unique()))
jumlah = st.number_input("🔢 Jumlah rekomendasi:", 1, 50, 10)
tampilan = st.radio("🎨 Pilih tampilan:", ["Tabel"], horizontal=True)


if st.button("✨ Tampilkan Rekomendasi "):
    if not user:
        st.warning("⚠️ Harap masukkan nama user terlebih dahulu.")
    else:
        
        genre_pilihan = df.loc[df['title'] == film, 'genres_str'].values[0]

        st.write(f"👤 **User**: {user}")
        st.write(f"🎬 **Film yang dipilih**: {film}")
        st.write(f"🎭 **Genre**: {genre_pilihan}")
        st.markdown("### ✅ Rekomendasi Film")

        
        rekom = df[(df['genres_str'] == genre_pilihan) & (df['title'] != film)]
        rekom = rekom.sample(n=min(jumlah, len(rekom)), random_state=None)

        if rekom.empty:
            st.warning("⚠️ Tidak ada rekomendasi lain dengan genre yang sama.")
        elif tampilan == "Tabel":
            st.dataframe(rekom[['title', 'genres_str', 'release_year', 'duration']])
        elif tampilan == "List":
            for _, row in rekom.iterrows():
                st.write(f"- 🎬 **{row['title']}** ({row['release_year']}) | 🎭 {row['genres_str']} | ⏱ {row['duration']}")
        else:  
            cols = st.columns(2)
            for idx, row in enumerate(rekom.itertuples()):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div style="background:#f5f5f5; padding:15px; margin:10px; 
                                 border-radius:10px; box-shadow:2px 2px 6px #ccc">
                        <h4>🎬 {row.title}</h4>
                        <p><b>Genre:</b> {row.genres_str}</p>
                        <p><b>Tahun:</b> {row.release_year}</p>
                        <p><b>Durasi:</b> {row.duration}</p>
                    </div>
                    """, unsafe_allow_html=True)
