# 🛡️ İşçiKalkanı AI

**Kaza olduktan sonra değil, olmadan önce harekete geçen yapay kalkan.**

İşçiKalkanı AI'ın çıkış noktası, 9 Kasım 2026'da Aliağa HABAŞ Demir Çelik Fabrikası'nda yaşanan ve çok sayıda işçinin hayatını kaybettiği acı iş cinayetidir. Bu proje, benzer kayıpların yeniden yaşanmaması için geliştirilmiş **prediktif (önleyici)** bir iş güvenliği sistemidir. Reaktif raporlama yerine riski kaza gerçekleşmeden hesaplar, alarm verir ve geleceği simüle eder.

> *Hayatını kaybeden tüm işçilerin anısına.*

---

## ⚡ Öne Çıkan Farklar

- **Donanımdan bağımsız:** Pahalı sensör ve IoT altyapısı yok. Sistem yalnızca yazılımla, işçi beyanı, vardiya bilgisi ve ortam okumalarıyla çalışır.
- **İnsan fizyolojisi merkezde:** Açlık, susuzluk, yorgunluk ve ilaç saati risk hesabının temel girdisidir.
- **Kolektif risk:** Fizyolojik, sağlık, çevresel ve saha ihbarı verileri tek bir **Fabrika Risk Skoru**nda (%0-%100) birleşir.
- **Anonim ve baskısız:** İşçiler sahadaki tehlikeleri kimlik vermeden bildirir.
- **Öngörülü:** Dijital ikiz, "Yarın hava 40°C olursa ve 3 işçi fazla mesaiye kalırsa risk ne olur?" sorusunu yanıtlar.

---

## 🧩 6 Ana Modül

| # | Modül | Teknik Karşılığı |
|---|-------|------------------|
| 1 | **Fizyolojik Stres** | Çalışma saati, aralıksız çalışma, açlık ve susuzluk girdilerinden `RandomForestRegressor` ile risk tahmini; yorgunluk ve dikkat eğrisi. Model, gerçekçi korelasyonlarla üretilmiş sentetik veriyle eğitilir. |
| 2 | **Sağlık / İlaç Entegrasyonu** | Kronik rahatsızlık ve son ilaç alımından geçen süreye göre dinamik **sağlık katsayısı**; ilaç gecikince bireysel risk yukarı çekilir. |
| 3 | **Çevresel Ortam** | Sıcaklık, nem ve gaz (ppm) verisinden *hissedilen sıcaklık* ve çevresel risk. Sıcakta yorgunluk çarpanı 1.0x'ten 2.0x'e çıkar ve aralıksız çalışma tolerans süresi düşer. |
| 4 | **Anonim Near-Miss / NLP İhbar** | Serbest metin ihbarında risk anahtar kelimelerini (`gevşek`, `yağ`, `kablo`, `arıza` vb.) yakalayan kural tabanlı NLP; ilgili bölgenin riski +15 puan artar. İhbarlar `st.session_state` içinde tutulur. |
| 5 | **Akıllı Alarm Sistemi** | Risk %80'i aşınca yanıp sönen "Üretimi Durdurma" alarmı, ses efekti ve Telegram/SMS yönetici bildirimi (simülasyon) ile alarm günlüğü. Kritik gaz seviyesi yorgunluktan bağımsız acil durum başlatır. |
| 6 | **Dijital İkiz Simülatörü** | Yarının sıcaklığı, fazla mesai süresi, işçi sayısı ve bakım durumuna göre **saat saat risk eğrisi**; normal gün referansıyla karşılaştırma. |

---

## 🎨 UI/UX: Endüstriyel Karanlık Tema

- **Industrial Dark Theme:** Antrasit/kömür grisi (`#1E1E24`) arka plan, açık gri metinler, özel CSS ile yeniden tasarlanmış yan panel ve sekmeler.
- **Akıllı risk kartları:** `st.metric` yerine, risk seviyesine göre parlayan özel HTML/CSS kartları:
  - 🟢 **Neon Yeşil:** Risk < %50
  - 🟡 **Neon Sarı:** %50 ≤ Risk < %80
  - 🔴 **Neon Kırmızı:** Risk ≥ %80, nabız efektiyle yanıp söner
- **Koyu temalı grafikler:** Matplotlib grafikleri koyu zeminde, neon turuncu ve neon mavi çizgilerle çizilir.
- **Alarm modu:** Kritik eşikte ekranı kaplayan, yanıp sönen "CRITICAL" bandı.

---

## 🛠️ Teknolojiler ve Kurulum

**Teknolojiler:** Python 3.10+ · Streamlit · Scikit-Learn · Pandas · NumPy · Matplotlib

```bash
# Depoyu klonla
git clone https://github.com/<kullanici-adi>/iscikalkani-ai.git
cd iscikalkani-ai

# Bağımlılıkları kur
pip install streamlit pandas numpy scikit-learn matplotlib

# Uygulamayı başlat
streamlit run app.py
```

*(İsteğe bağlı)* Tüm bileşenlerin koyu görünmesi için `.streamlit/config.toml` dosyasına şunu ekle:

```toml
[theme]
base = "dark"
primaryColor = "#FF6700"
backgroundColor = "#1E1E24"
secondaryBackgroundColor = "#2A2A32"
textColor = "#EDEDED"
```

---

## ⚠️ Sorumluluk Notu

İşçiKalkanı AI bir **prototip ve mimari gösterimidir**. Model sentetik veriyle eğitilmiştir, katsayılar ve eşikler kalibre edilmemiş varsayımlardır. Telegram/SMS bildirimleri simülasyondur. Sistem, resmi iş sağlığı ve güvenliği süreçlerinin ve uzman değerlendirmesinin yerine geçmez. Sağlık verileri gerçek kullanımda KVKK kapsamında korunmalıdır.
