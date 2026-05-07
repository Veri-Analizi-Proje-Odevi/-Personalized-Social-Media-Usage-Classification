import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Sayfa Ayarları
st.set_page_config(page_title="Sosyal Medya Analizi", page_icon="🧠")

@st.cache_resource
def load_assets():
    model = joblib.load('en_iyi_model_lojistik_regresyon.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_assets()
except:
    st.error("Model dosyaları yüklenemedi!")
    st.stop()

st.title("🧠 Sosyal Medya ve Psikolojik Sağlık Analizi")
st.write("Lütfen aşağıdaki anketi doldurunuz. Modelimiz risk durumunuzu anlık olarak analiz edecektir.")

# --- TÜRKÇE -> İNGİLİZCE MAPPING SÖZLÜKLERİ ---
answer_map = {"Evet": "Yes", "Hayır": "No"}
gender_map = {"Erkek": "Male", "Kadın": "Female", "Diğer": "Non-binary"}
rel_map = {"Bekar": "Single", "Evli": "Married", "İlişkisi var": "In a relationship", "Boşanmış": "Divorced"}
occ_map = {"Üniversite Öğrencisi": "University Student", "Okul Öğrencisi": "School Student", "Çalışan": "Salaried Worker", "Emekli": "Retired"}
time_map_tr = {
    "1 Saatten Az": "Less than an Hour", 
    "1-2 Saat": "Between 1 and 2 hours", 
    "2-3 Saat": "Between 2 and 3 hours", 
    "3-4 Saat": "Between 3 and 4 hours", 
    "4-5 Saat": "Between 4 and 5 hours", 
    "5 Saatten Fazla": "More than 5 hours"
}

# 1. BÖLÜM: KİŞİSEL
with st.expander("Kişisel Bilgiler", expanded=True):
    age = st.number_input("Yaşınız:", 10, 100, 20)
    gender_tr = st.selectbox("Cinsiyet:", list(gender_map.keys()))
    rel_tr = st.selectbox("İlişki Durumu:", list(rel_map.keys()))
    occ_tr = st.selectbox("Meslek:", list(occ_map.keys()))
    org = st.text_input("Bağlı olduğunuz kurum (Okul/Şirket):", "-")

# 2. BÖLÜM: SOSYAL MEDYA
with st.expander("Sosyal Medya Kullanımı"):
    use_sm = st.radio("Sosyal medya kullanıyor musunuz?", list(answer_map.keys()))
    platforms = st.multiselect("Hangi platformları kullanıyorsunuz?", 
                               ["Facebook", "Instagram", "YouTube", "TikTok", "Twitter", "Discord"])
    time_tr = st.selectbox("Günlük ortalama kullanım süresi:", list(time_map_tr.keys()))

# 3. BÖLÜM: PSİKOLOJİK SORULAR
with st.expander("Değerlendirme (1: Hiç, 5: Çok Fazla)"):
    q9 = st.slider("Belirli bir amaç olmadan sosyal medyayı ne sıklıkla kullanıyorsunuz?", 1, 5, 3)
    q10 = st.slider("Bir şeyle meşgulken sosyal medya ne sıklıkla sizin dikkatinizi dağıtıyor?", 1, 5, 3)
    q11 = st.slider("Bir süre sosyal medya kullanmadığınızda huzursuz hissediyor musunuz?", 1, 5, 3)
    q12 = st.slider("Ne kadar kolay dikkatiniz dağılıyor?", 1, 5, 3)
    q13 = st.slider("Endişelerden ne kadar rahatsız oluyorsunuz?", 1, 5, 3)
    q14 = st.slider("Konsantre olmakta zorlanıyor musunuz?", 1, 5, 3)
    q15 = st.slider("Kendinizi diğer başarılı insanlarla ne sıklıkla karşılaştırıyorsunuz?", 1, 5, 3)
    q16 = st.slider("Önceki soruyu takiben, genel olarak bu karşılaştırmalardan ne kadar etkileniyorsunuz?", 1, 5, 3)
    q17 = st.slider("Onay (beğeni, yorum vb.) arama sıklığınız nedir?", 1, 5, 3)
    q18 = st.slider("Depresif veya üzgün hissetme sıklığınız nedir?", 1, 5, 3)
    q19 = st.slider("Günlük aktivitelere olan ilginiz ne sıklıkla dalgalanıyor?", 1, 5, 3)
    q20 = st.slider("Ne sıklıkla uyku sorunlarıyla karşılaşıyorsunuz?", 1, 5, 3)
    # Diğer soruları da buraya ekleyebilirsin...

if st.button("ANALİZ ET"):
    # Zaman Mapping (Sayısal Değerler için)
    numeric_time_map = {
        'Less than an Hour': 0.5, 'Between 1 and 2 hours': 1.5,
        'Between 2 and 3 hours': 2.5, 'Between 3 and 4 hours': 3.5,
        'Between 4 and 5 hours': 4.5, 'More than 5 hours': 6.0
    }
    
    # Kullanıcıdan gelen Türkçe veriyi İngilizceye çevirerek data oluşturuyoruz
    data = {
        '1. What is your age?': age,
        '2. Gender': gender_map[gender_tr], # Erkek seçilirse Male gider
        '3. Relationship Status': rel_map[rel_tr],
        '4. Occupation Status': occ_map[occ_tr],
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
        # Not: Modelin beklediği diğer sütunları da buraya 3 değerini vererek ekle
    }

    # DataFrame ve dummy işlemleri (Aynı kalıyor)
    df_input = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df_input)
    
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_final = df_encoded[model_columns]
    X_scaled = scaler.transform(df_final)
    
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ RİSK DURUMU: YÜKSEK (%{prob*100:.1f})")
    else:
        st.success(f"✅ RİSK DURUMU: DÜŞÜK (%{prob*100:.1f})")
    st.balloons()
