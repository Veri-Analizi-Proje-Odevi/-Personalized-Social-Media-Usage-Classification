import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Sayfa Ayarları
st.set_page_config(page_title="Sosyal Medya Risk Analizi", page_icon="🧠")

# 1. Kayıtlı Modelleri Yükle
# app.py içindeki bu kısmı şöyle güncelleyin:
@st.cache_resource
def load_assets():
    model = joblib.load('en_iyi_model_lojistik_regresyon.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_assets()
except:
    st.error("Model dosyaları bulunamadı! Lütfen model.pkl, scaler.pkl ve model_columns.pkl dosyalarını yükleyin.")
    st.stop()

# 2. Arayüz ve Sorular
st.title("🧠 Sosyal Medya ve Psikolojik Sağlık Analizi")
st.write("Lütfen aşağıdaki anketi doldurunuz. Yapay zeka modelimiz risk durumunuzu anlık olarak analiz edecektir.")

# Soruları Kategorilere Ayıralım
with st.expander("Kişisel Bilgiler", expanded=True):
    age = st.number_input("1. Yaşınız:", min_value=10, max_value=100, value=20)
    gender = st.selectbox("2. Cinsiyet:", ["Male", "Female", "Non-binary"])
    rel_status = st.selectbox("3. İlişki Durumu:", ["Single", "Married", "In a relationship", "Divorced"])
    occupation = st.selectbox("4. Meslek/Durum:", ["University Student", "School Student", "Salaried Worker", "Retired"])
    org = st.selectbox("5. Kurum Tipi:", ["University", "Private", "Company", "Government", "N/A"])

with st.expander("Sosyal Medya Kullanımı"):
    use_sm = st.radio("6. Sosyal medya kullanıyor musunuz?", ["Yes", "No"])
    platforms = st.multiselect(
        "7. Yaygın olarak kullandığınız sosyal medya platformları nelerdir?",
        ["Facebook", "Twitter", "Instagram", "YouTube", "Discord", "Reddit", "Pinterest", "TikTok"]
    )
    
    platforms_str = ", ".join(platforms)
    time_spent = st.selectbox("8. Günlük ortalama kullanım süreniz:", 
                             ["Less than an Hour", "Between 1 and 2 hours", "Between 2 and 3 hours", 
                              "Between 3 and 4 hours", "Between 4 and 5 hours", "More than 5 hours"])

with st.expander("Psikolojik Değerlendirme (1: Hiç, 5: Çok Fazla)"):
    q9 = st.slider("9. Amaçsızca kullanma sıklığı?", 1, 5, 3)
    q10 = st.slider("10. Meşgulken dikkatinizin dağılması?", 1, 5, 3)
    q11 = st.slider("11. Kullanmadığınızda huzursuz hissetme?", 1, 5, 3)
    q12 = st.slider("12. Genel olarak dikkatinizin dağılma kolaylığı?", 1, 5, 3)
    q13 = st.slider("13. Endişelerden rahatsız olma düzeyi?", 1, 5, 3)
    q14 = st.slider("14. Konsantrasyon güçlüğü?", 1, 5, 3)
    q15 = st.slider("15. Kendinizi başkalarıyla kıyaslama?", 1, 5, 3)
    q16 = st.slider("16. Bu kıyaslamalar size ne hissettiriyor?", 1, 5, 3)
    q17 = st.slider("17. Onaylanma (like vb.) arama sıklığı?", 1, 5, 3)
    q18 = st.slider("18. Depresif veya üzgün hissetme sıklığı?", 1, 5, 3)
    q19 = st.slider("19. İlginizin dalgalanma sıklığı?", 1, 5, 3)
    q20 = st.slider("20. Uyku sorunları yaşama sıklığı?", 1, 5, 3)

# 3. Veriyi Modele Hazırlama (Preprocessing)
if st.button("ANALİZ ET"):
    # Zaman Mapping (Notebook'unuzdaki ile aynı)
    time_mapping = {
        'Less than an Hour': 0.5, 'Between 1 and 2 hours': 1.5,
        'Between 2 and 3 hours': 2.5, 'Between 3 and 4 hours': 3.5,
        'Between 4 and 5 hours': 4.5, 'More than 5 hours': 6.0
    }
    
    # Input Sözlüğü (Notebook'taki sütun yapılarını simüle eder)
    data = {
        '1. What is your age?': age,
        '2. Gender': gender,
        '3. Relationship Status': rel_status,
        '4. Occupation Status': occupation,
        '5. What type of organizations are you affiliated with?': org,
        '6. Do you use social media?': use_sm,
        'Usage_Hours': time_mapping[time_spent],
        '9. How often do you find yourself using Social media without a specific purpose?': q9,
        '10. How often do you get distracted by Social media when you are busy doing something?': q10,
        '11. Do you feel restless if you haven\'t used Social media in a while?': q11,
        '12. On a scale of 1 to 5, how easily distracted are you?': q12,
        '13. On a scale of 1 to 5, how much are you bothered by worries?': q13,
        '14. Do you find it difficult to concentrate on things?': q14,
        '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': q15,
        '16. Following the previous question, how do you feel about these comparisons, generally speaking?': q16,
        '17. How often do you look to seek validation from features of social media?': q17,
        '18. How often do you feel depressed or down?': q18,
        '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': q19,
        '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': q20
    }

    df_input = pd.DataFrame([data])
    
    # Dummy Variables (One-Hot Encoding)
    # Eğitimdeki sütunlarla birebir eşleşmesi için:
    df_encoded = pd.get_dummies(df_input)
    
    # Eksik sütunları 0 ile doldur (Eğitim setinde olup burada olmayan dummy sütunlar için)
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Sütun sırasını eğitim setiyle aynı yap
    df_final = df_encoded[model_columns]
    
    # Scaling
    X_scaled = scaler.transform(df_final)
    
    # Tahmin
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    # 4. Sonucu Göster
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ ANALİZ SONUCU: YÜKSEK RİSK (Olasılık: %{probability*100:.1f})")
        st.write("Sosyal medya kullanım alışkanlıklarınız psikolojik sağlığınızı olumsuz etkiliyor olabilir. Bir uzmana danışmayı veya dijital detoks yapmayı düşünebilirsiniz.")
    else:
        st.success(f"✅ ANALİZ SONUCU: DÜŞÜK RİSK (Olasılık: %{probability*100:.1f})")
        st.write("Tebrikler! Sosyal medya kullanım alışkanlıklarınız sağlıklı bir dengede görünüyor.")

    st.balloons()
