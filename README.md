# Telco Customer Churn Prediction

**Model klasifikasi churn pelanggan untuk perusahaan telekomunikasi**  
Menggunakan:  
- Preprocessing (StandarScaller + one-hot encoding)  
- Resampling data dengan ADASYN untuk menangani ketidakseimbangan kelas  
- Model klasifikasi yang dipilih: Logistic Regression + ADASYN

---

## Deskripsi Proyek

Perusahaan telekomunikasi sering menghadapi masalah customer churn (pelanggan berhenti berlangganan). Churn ini berdampak langsung pada pendapatan dan stabilitas bisnis. Proyek ini bertujuan membangun sistem prediksi churn berdasarkan fitur pelanggan (durasi layanan, jenis kontrak, layanan internet, dsb), sehingga perusahaan bisa:

- Mengidentifikasi pelanggan yang berisiko churn secara dini  
- Melakukan strategi retensi atau penawaran ulang layanan secara proaktif  
- Mengurangi biaya akuisisi pelanggan baru  
- Meningkatkan loyalitas dan profitabilitas  

---

## Struktur Proyek

```text
telco-customer-churn-classification/
├── data/
│   └── data_telco_customer_churn.csv      
├── models/
│   └── churn_prediction_model.joblib          
├── app.py                
├── README.md                               
└── requirements.txt
```

---

## Fitur

1. Bussiness Understanding
1. EDA Interaktif: Filter data berdasarkan Jenis Kontrak dan Layanan Internet
2. Prediksi churn atau tidak.

---

## Parameter
Parameter yang digunakan untuk memprediksi churn atau tidak

1.  **Dependents** : tanggungan (Yes/No)
2.  **tenure** : durasi waktu (dalam bulan) pelanggan telah berlangganan (slider 0 - 72)
3.  **OnlineSecurity** : Layanan keamanan online tambahan (Yes, No, atau No internet service)
4.  **OnlineBackup** : Layanan pencadangan data online (Yes, No, atau No internet service)
5.  **InternetService** : Jenis layanan internet yang digunakan pelanggan (DSL, Fiber optic, atau No) 
6.  **DeviceProtection** : Layanan perlindungan perangkat (Yes, No, atau No internet service)
7.  **TechSupport** : Layanan dukungan teknis tambahan (Yes, No, atau No internet service)
8.  **Contract** : Jenis kontrak berlangganan pelanggan (Month-to-month, One year, atau Two year) 
9.  **PaperlessBilling** : Apakah pelanggan memilih tagihan tanpa kertas (elektronik) atau tidak (Yes/No)
10. **MonthlyCharges** : Jumlah tagihan bulanan yang dibebankan kepada pelanggan (input 20 - 120)

---

## Menjalankan Proyek (Local)

### 1. Clone Repository

```bash
git clone https://github.com/Zepfort/telco-customer-churn-streamlit.git
cd telco-customer-churn-streamlit
```

### 2.pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

### 3.Jalankan streamlit local
```bash
streamlit run app.py
```

---

## Pemenuhan tugas UAS Mini Project Scripting Language