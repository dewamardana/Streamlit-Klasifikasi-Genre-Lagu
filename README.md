# 🎵 Klasifikasi Genre Lagu (Streamlit)

Aplikasi web untuk mengklasifikasikan genre lagu menggunakan model deep learning, dibungkus dalam antarmuka **Streamlit**.

## 📌 Deskripsi

Model deep learning (Keras/TensorFlow, `best_model.h5`) dilatih untuk mengenali genre musik dari data audio/fitur lagu, kemudian di-deploy sebagai aplikasi web interaktif menggunakan Streamlit.

## ⚙️ Teknologi

- **Bahasa:** Python
- **Model:** Keras/TensorFlow (`best_model.h5`) + `LabelEncoder` (`labelencoder.pkl`)
- **Framework web:** Streamlit
- **Notebook:** `FIX_Music_Genre_Classification.ipynb` (eksperimen & training model)

## 🚀 Cara Menjalankan

```bash
git clone https://github.com/dewamardana/Streamlit-Klasifikasi-Genre-Lagu.git
cd Streamlit-Klasifikasi-Genre-Lagu
pip install -r requirements.txt   # jika belum ada, buat dari import di app.py
streamlit run app.py
```

## 📁 Isi Repo

- `FIX_Music_Genre_Classification.ipynb` — notebook eksperimen & pelatihan model
- `app.py` — aplikasi Streamlit
- `best_model.h5` — model terlatih
- `labelencoder.pkl` — encoder label genre
- `Data Test/` — data uji
- `Link Presentasi Youtube.txt` — tautan video presentasi project
