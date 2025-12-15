import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Retail", layout="wide")

st.title("🛍️ Clustering & Prediction Dashboard")
st.caption("Metode: K-Means (Clustering) & Ensemble Learning (Regression)")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Streamlit Cloud akan mencari file ini di GitHub Anda
    try:
        # Coba baca dengan encoding standar
        df = pd.read_csv('data_retail.csv')
        return df
    except UnicodeDecodeError:
        # Jika gagal, coba encoding lain (biasa terjadi di data retail)
        df = pd.read_csv('data_retail.csv', encoding='unicode_escape')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # --- 1. PREPROCESSING & RFM ---
    st.sidebar.success("✅ Data Berhasil Dimuat!")
    
    # Check kolom wajib
    kolom_wajib = ['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceDate', 'InvoiceNo']
    if not all(col in df.columns for col in kolom_wajib):
        st.error(f"Data tidak memiliki kolom wajib: {kolom_wajib}")
        st.stop()

    # Cleaning sederhana
    df_clean = df.dropna(subset=['CustomerID'])
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # Buat RFM
    snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    # Hapus Outlier Ekstrem (Agar grafik tidak gepeng/rusak)
    Q1 = rfm['Monetary'].quantile(0.05)
    Q3 = rfm['Monetary'].quantile(0.95)
    rfm_filtered = rfm[(rfm['Monetary'] >= Q1) & (rfm['Monetary'] <= Q3)].copy()

    st.header("1. Data Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Customer", len(rfm))
    col2.metric("Customer (Setelah Filter Outlier)", len(rfm_filtered))

    # --- 2. CLUSTERING (K-MEANS) ---
    st.markdown("---")
    st.header("2. Clustering (K-Means)")
    
    k_val = st.slider("Pilih Jumlah Cluster", 2, 5, 3)
    
    # Scaling & Modeling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_filtered[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    rfm_filtered['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Visualisasi
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Segmentasi: Recency vs Monetary")
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm_filtered, x='Recency', y='Monetary', hue='Cluster', palette='viridis', ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Segmentasi: Recency vs Frequency")
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm_filtered, x='Recency', y='Frequency', hue='Cluster', palette='viridis', ax=ax)
        st.pyplot(fig)

    # --- 3. REGRESSION (ENSEMBLE) ---
    st.markdown("---")
    st.header("3. Regression (Prediksi Belanja)")
    st.info("Menggunakan Ensemble Methods untuk memprediksi Total Belanja (Monetary).")

    tipe_model = st.selectbox("Pilih Model:", ["Random Forest", "Gradient Boosting"])
    
    # Split Data
    X = rfm_filtered[['Recency', 'Frequency']] # Fitur
    y = rfm_filtered['Monetary']                # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    if tipe_model == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Hasil
    r2 = r2_score(y_test, y_pred)
    st.success(f"Akurasi Model ({tipe_model}): R2 Score = {r2:.4f}")
    
    # Prediksi Manual
    st.subheader("Coba Prediksi")
    c_input1, c_input2 = st.columns(2)
    with c_input1:
        in_rec = st.number_input("Terakhir Belanja (Hari lalu)", value=30)
    with c_input2:
        in_freq = st.number_input("Frekuensi Belanja (Total Transaksi)", value=5)
    
    if st.button("Hitung Prediksi"):
        hasil = model.predict([[in_rec, in_freq]])
        st.metric("Prediksi Total Belanja Nanti ($)", f"{hasil[0]:.2f}")

else:
    st.warning("⚠️ File 'data_retail.csv' belum ditemukan.")
    st.markdown("""
    **Solusi:**
    1. Pastikan Anda sudah menjalankan 'Sel 1' di Google Colab.
    2. Download file `data_retail.csv` hasil generate tersebut.
    3. Upload file tersebut ke Repository GitHub Anda bersama `app.py`.
    """)
