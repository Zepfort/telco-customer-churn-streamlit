import streamlit as st
import pandas as pd
import joblib

#  konfigurasi halaman
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="centered"
)

#  Load pipeline model
@st.cache_resource
def load_pipeline():
    return joblib.load('./models/churn_prediction_model.joblib')

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("File 'churn_prediction_model.joblib' tidak ada")
    st.stop()

st.title("📡 Telco Customer Churn Prediction")
st.markdown("""
Aplikasi ini menggunakan **Logistic Regression + ADASYN** untuk memprediksi risiko pelanggan berhenti berlangganan.
Model dioptimalkan untuk mendeteksi potensi churn sedini mungkin.
""")

st.write("")

#  form input (SIDEBAR) 
st.sidebar.header("Input Data Pelanggan")

def user_input_features():
    #  Fitur Kategorikal 
    dependents = st.sidebar.selectbox("Memiliki Tanggungan (Dependents)?", ["Yes", "No"])
    internet_service = st.sidebar.selectbox("Jenis Internet Service", ["DSL", "Fiber optic", "No"])    
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Tagihan Paperless?", ["Yes", "No"])
    
    #  Fitur Numerik 
    st.sidebar.markdown("")
    tenure = st.sidebar.slider("Lama Berlangganan (Bulan)", min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.number_input("Biaya Bulanan (Monthly Charges)", min_value=18.0, max_value=120.0, value=50.0, step=0.5) 
    data = {
        'Dependents': dependents,
        'tenure': tenure,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'InternetService': internet_service,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges
    }
    
    return pd.DataFrame(data, index=[0])

# input dari fungsi 
input_df = user_input_features()

# review dari input user
st.subheader("Review Data Pelanggan")
st.dataframe(input_df)

# tombbol prediksi
if st.button("Prediksi Risiko Churn"):
    
    with st.spinner('Sedang menganalisis profil pelanggan...'):
        try:
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1] # Probabilitas kelas 1 (Churn)
            
            #  Hasil
            st.write("")
            st.subheader("Hasil Analisis")

            # Logic Threshold: 0.5 
            THRESHOLD = 0.55 
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Probabilitas Churn", value=f"{probability*100:.2f}%")
            
            with col2:
                if probability > THRESHOLD:
                    st.error("**STATUS: BERISIKO CHURN** ⚠️")
                else:
                    st.success("**STATUS: PELANGGAN SETIA (LOYAL)** ✅")

            # Penjelasan Bisnis
            if probability > THRESHOLD:
                st.warning(
                    f"""
                    **Rekomendasi Tindakan:**
                    Pelanggan ini memiliki karakteristik yang mirip dengan pelanggan yang berhenti berlangganan.
                    - 📞 Hubungi pelanggan segera.
                    - 🏷️ Tawarkan diskon retensi atau upgrade layanan.
                    - 🤝 Tanyakan kendala yang dihadapi (terutama pengguna {input_df['InternetService'][0]} dengan kontrak {input_df['Contract'][0]}).
                    """
                )
            else:
                st.info(
                    """
                    **Rekomendasi Tindakan:**
                    Pelanggan terlihat puas.
                    - Pertahankan kualitas layanan.
                    - Tidak perlu promosi agresif saat ini.
                    """
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.write("Detail error untuk debugging:", e)

# Footer
st.markdown("")
st.caption("Tugas Praktik Aplikasi Web & Scripting Language - Machine Learning Classification")