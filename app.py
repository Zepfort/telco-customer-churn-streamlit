import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(
    page_title="Telco Churn Prediction",
    layout="wide" 
)

# Load Model & Data
@st.cache_resource
def load_pipeline():
    return joblib.load('./models/churn_prediction_model.joblib')

@st.cache_data
def load_data():
    df = pd.read_csv('./data/data_telco_customer_churn.csv')
    
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
    return df

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("File model tidak ditemukan.")
    st.stop()

try:
    df_eda = load_data()
except FileNotFoundError:
    st.warning("File CSV dataset tidak ditemukan. EDA tidak dapat ditampilkan")
    df_eda = None

# 3. Judul
st.title("Telco Customer Churn Dashboard")
st.markdown("Dashboard prediksi churn pelanggan dan analisis data .")

# Tab
tab1, tab2, tab3, tab4 = st.tabs(["Bussines Understanding","Data Overview", "EDA & Visualisasi", "Prediksi Churn"])


# Tab 1: Bussiness Understanding
with tab1:
    st.header("Business Understanding")
    
    # Bagian Ringkasan Teknis (Menggunakan st.info agar terlihat rapi)
    st.info("""
    **Model klasifikasi churn pelanggan untuk perusahaan telekomunikasi**
    
    **Menggunakan:**
    * Preprocessing (StandardScaler + One-Hot Encoding)
    * Resampling data dengan **ADASYN** untuk menangani ketidakseimbangan kelas
    * Model klasifikasi yang dipilih: **Logistic Regression + ADASYN**
    """)
    
    st.markdown("---")
    
    # Bagian Deskripsi Proyek
    st.subheader("Deskripsi Proyek")
    st.markdown("""
    Perusahaan telekomunikasi sering menghadapi masalah *customer churn* (pelanggan berhenti berlangganan). 
    Churn ini berdampak langsung pada pendapatan dan stabilitas bisnis. 
    
    Dengan model prediksi, perusahaan bisa:
    
    1. **Mengidentifikasi** pelanggan yang berisiko churn secara dini.
    2. **Melakukan strategi retensi** atau penawaran ulang layanan secara aktif.
    3. **Mengurangi biaya** akuisisi pelanggan baru.
    4. **Meningkatkan loyalitas** pelanggan dan profit perusahaan.
    """)
    
    
with tab2:
    st.header("Data Overview")
    
    df_raw = pd.read_csv('./data/data_telco_customer_churn.csv')
    total_raw = df_raw.shape[0]
    total_clean = df_eda.shape[0]
    total_dropped = total_raw - total_clean
    
    st.subheader("Cuplikan Dataset")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric(label="Total Data Asli", value=f"{total_raw}", help="Jumlah baris sebelum cleaning")
        
    with col_m2:
        st.metric(label="Data Bersih", value=f"{total_clean}", help="Jumlah baris setelah drop duplikat")
        
    with col_m3:
        st.metric(label="Data dibuang (Duplikat)",  value=f"{total_dropped}", delta_color="inverse")
        
    st.markdown("---")
    
    # Tabel 5 dataset teratas
    st.caption("Menampilkan 5 baris pertama dari dataset.")
    st.dataframe(df_eda.head())
    
    # Tabel Jenis Data
    st.subheader("Informasi Jenis Data")
    
    dtype_df = df_eda.dtypes.astype(str).reset_index()
    dtype_df.columns = ["Nana Kolom", "Tipe Data"]
    st.dataframe(dtype_df, hide_index=True, use_container_width=True)
        
    st.markdown("---")

    
    st.subheader("Statistik Deskriptif")
    
    desc_df = df_eda.describe().T
    st.dataframe(desc_df, use_container_width=True)

# Tab 3: EDA Interaktif 

with tab3:
    if df_eda is not None:
        st.header("Exploratory Data Analysis (Interaktif)")
        
        color_map = {'Yes': '#ff6b6b', 'No': '#4ecdc4'}  # Merah untuk Churn, Biru untuk Tidak
        churn_order = {"Churn": ["No", "Yes"]}
        
        #  Filter data
        st.subheader("1. Filter Data")
        with st.expander("Klik untuk memfilter segmen pelanggan"):
            st.caption("Gunakan filter ini untuk melihat perilaku churn pada kelompok pelanggan tertentu.")
            
            col_filter1, col_filter2 = st.columns(2)
            
            # Filter 1: Kontrak 
            with col_filter1:
                filter_contract = st.multiselect(
                    "Pilih Jenis Kontrak:",
                    options=df_eda['Contract'].unique(),
                    default=df_eda['Contract'].unique()
                )
            
            # Filter 2: Layanan Internet
            with col_filter2:
                filter_service = st.multiselect(
                    "Pilih Layanan Internet:",
                    options=df_eda['InternetService'].unique(),
                    default=df_eda['InternetService'].unique()
                )
            
            df_filtered = df_eda[
                (df_eda['Contract'].isin(filter_contract)) & 
                (df_eda['InternetService'].isin(filter_service))
            ]
            
            #  info jumlah data
            st.write(f"Menampilkan **{len(df_filtered)}** data dari total {len(df_eda)} pelanggan.")

        st.markdown("---")

        #  Visualisasi
        st.subheader("2. Visualisasi Distribusi Churn")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("**Proporsi Churn (Segmen Terpilih)**")
            if len(df_filtered) > 0:
                churn_counts = df_filtered['Churn'].value_counts().reset_index()
                churn_counts.columns = ['Churn', 'Count']
                
                fig_pie = px.pie(
                    churn_counts, 
                    values='Count', 
                    names='Churn', 
                    color='Churn',
                    color_discrete_map=color_map,
                    category_orders=churn_order,
                    hole=0.3
                )
                fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Data kosong. Silakan pilih filter lain.")
            
        with col_viz2:
            st.markdown("**Distribusi Biaya Bulanan (Monthly Charges)**")
            if len(df_filtered) > 0:
                fig_hist = px.histogram(
                    df_filtered, 
                    x="MonthlyCharges", 
                    color="Churn", 
                    marginal="box",
                    color_discrete_map=color_map,
                    category_orders=churn_order,
                    nbins=30,
                    barmode='overlay',
                    opacity=0.7
                )
                fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # Analisis fitur kategorikal
        st.subheader("3. Analisis Faktor Kategorikal Lainnya")
        
        # Daftar kolom kategorikal 
        cat_options = [
            'Contract', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'PaperlessBilling', 'Dependents'
        ]
        
        selected_cat = st.selectbox("Pilih faktor yang ingin dianalisis:", cat_options, index=0)
        
        if len(df_filtered) > 0:
            cat_data = df_filtered.groupby([selected_cat, 'Churn']).size().reset_index(name='Count')
            
            fig_bar = px.bar(
                cat_data, 
                x=selected_cat, 
                y="Count", 
                color="Churn", 
                barmode='group',
                color_discrete_map=color_map,
                category_orders=churn_order,
                text_auto=True,
                title=f"Hubungan {selected_cat} dengan Churn"
            )
            
            fig_bar.update_layout(
                xaxis_title=selected_cat, 
                yaxis_title="Jumlah Pelanggan",
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)

            st.caption(f"Grafik di atas memperlihatkan bagaimana sebaran Churn pada kategori **{selected_cat}**.")

        st.markdown("---")

        # Distribusi Tenure
        st.subheader("4. Apakah Pelanggan Baru Lebih Mudah Churn?")
        
        if len(df_filtered) > 0:
            fig_tenure = px.histogram(
                df_filtered, 
                x="tenure", 
                color="Churn", 
                marginal="box", 
                color_discrete_map=color_map,
                category_orders=churn_order,
                nbins=30,
                barmode='overlay',
                opacity=0.7,
                title="Distribusi Lama Berlangganan (Tenure)"
            )
            fig_tenure.update_layout(xaxis_title="Tenure (Bulan)", yaxis_title="Jumlah Pelanggan")
            st.plotly_chart(fig_tenure, use_container_width=True)

        st.markdown("---")

        # Korelasi numerik
        st.subheader("4. Korelasi Fitur Numerik")
        

        if len(df_filtered) > 0:
            numeric_cols = ['tenure', 'MonthlyCharges']
            numeric_df = df_filtered[numeric_cols].copy()
            
            if 'Churn' in df_filtered.columns:
                 numeric_df['Churn_Score'] = df_filtered['Churn'].map({'Yes': 1, 'No': 0})
            
            corr_matrix = numeric_df.corr().round(2)
            
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                origin='lower',
                title="Seberapa kuat hubungan Tenure & Biaya terhadap Churn?"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.info("Silakan upload file 'data_telco_customer_churn.csv' ke direktori proyek untuk melihat visualisasi.")

# Tab 4: Prediksi Churn 

with tab4:
    col_input, col_result = st.columns([1, 2], gap="large")
    
    with col_input:
        st.subheader("Input Data Pelanggan")
        
        # Input Form
        dependents = st.selectbox("Memiliki Tanggungan?", ["Yes", "No"])
        internet_service = st.selectbox("Jenis Internet Service", ["DSL", "Fiber optic", "No"])    
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Tagihan Paperless?", ["Yes", "No"])
        
        tenure = st.slider("Lama Berlangganan (Bulan)", 0, 72, 12)
        monthly_charges = st.number_input("Biaya Bulanan", 18.0, 120.0, 50.0, 0.5)

        # Tombol Prediksi
        predict_btn = st.button("Prediksi Risiko Churn")

    with col_result:
        if predict_btn:
            # Dataframe input model
            input_data = {
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
            input_df = pd.DataFrame(input_data, index=[0])
            
            st.subheader("Profil Pelanggan")
            st.dataframe(input_df)

            with st.spinner('Menganalisis...'):
                try:
                    probability = pipeline.predict_proba(input_df)[0][1]
                    THRESHOLD = 0.50
                    
                    st.markdown("---")
                    st.subheader("Hasil Analisis")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.metric("Probabilitas Churn", f"{probability*100:.2f}%")
                    
                    with col_res2:
                        if probability > THRESHOLD:
                            st.error("**BERISIKO CHURN** ⚠️")
                        else:
                            st.success("**PELANGGAN LOYAL** ✅")
                    
                    # Rekomendasi
                    if probability > THRESHOLD:
                        st.warning(f"""
                                   **Saran:** 
                                   Pelanggan ini memiliki karakteristik yang mirip dengan pelanggan yang berhenti berlangganan. 
                                   Segera tawarkan diskon retensi atau upgrade layanan. (terutama pengguna {internet_service}).
                                   """)
                    else:
                        st.info("**Saran:** Pertahankan kualitas layanan saat ini.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Mini Project Scripting Language - Machine Learning Classification")