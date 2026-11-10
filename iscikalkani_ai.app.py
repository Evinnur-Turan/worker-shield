"""
İşçiKalkanı AI - FİNAL SÜRÜM
Modül 1: Fizyolojik Stres ve Yorgunluk Takibi + Yapay Veri Seti
Modül 2: Sağlık ve Kronik Hastalık Entegrasyonu
Modül 3: Çevresel Stres Faktörleri (sıcaklık, nem, zehirli gaz)
Modül 4: Anonim Kıl Payı Atlatma (Near-Miss) Raporlama
Modül 5: Akıllı Alarm ve Bildirim Sistemi (simülasyon)
Modül 6: Dijital İkiz - Gelecek Senaryo Simülatörü
Çalıştırma: streamlit run app.py
"""

import html
import io
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")  # Streamlit için GUI'siz çizim motoru
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# SABİTLER
# ---------------------------------------------------------------------------
KRITIK_ESIK = 80  # Risk (%) bu değere ulaşınca üretimi durdurma alarmı verilir
OZELLIKLER = [
    "toplam_calisma_saati",
    "araliksiz_calisma_saati",
    "aclik_suresi_saat",
    "susuzluk_suresi_saat",
]

# --- Modül 2 ---
HASTALIK_PROFILLERI = {
    "Hipertansiyon (Tansiyon)": {"bazal": 0.05, "hassasiyet": 1.0},
    "Diyabet (Şeker)": {"bazal": 0.07, "hassasiyet": 1.5},
    "Kalp Rahatsızlığı": {"bazal": 0.08, "hassasiyet": 1.3},
    "Astım / KOAH": {"bazal": 0.04, "hassasiyet": 0.8},
    "Epilepsi": {"bazal": 0.06, "hassasiyet": 1.4},
}
ILAC_YAKLASMA_SAATI = 3.0
ILAC_GECIKME_SAATI = 4.0
MAX_ILAC_CEZASI = 0.25
MAX_SAGLIK_KATSAYISI = 1.8
ILAC_DOZ_ARALIGI = 8  # Dijital ikizde: ilaç 07:00, 15:00, 23:00'te alınıyor varsayımı

# --- Modül 3 ---
SICAKLIK_NORMAL_SINIRI = 28.0
SICAKLIK_KRITIK_ESIK = 35.0
SICAKLIK_TEHLIKE_SINIRI = 45.0
GAZ_UYARI = 10.0       # ppm (ÖRNEK değerler; gerçekte gaz türüne göre belirlenmeli)
GAZ_TEHLIKELI = 25.0
GAZ_KRITIK = 50.0
CEVRE_AGIRLIGI = 0.4
GAZ_ACIL_RISK = 85.0
NORMAL_ARALIKSIZ_LIMIT = 4.0

# --- Modül 4 ---
BOLGELER = ["Ana Vinç Sahası", "Kaynak Atölyesi", "Kesim Hattı", "Boyahane", "Sevkiyat Rampası"]
NEARMISS_ARTIS = 15.0      # Tetikleyen her ihbar, o bölgenin riskini +15 puan artırır
NEARMISS_MAX_ARTIS = 45.0  # Bir bölge için en fazla +45 puan
ANAHTAR_KELIMELER = {
    "gevşek": "gevşeme / bağlantı sorunu",
    "yağ": "kayma / yanıcı madde riski",
    "kablo": "elektrik / kablo riski",
    "arıza": "ekipman arızası",
    "kaçak": "sızıntı / kaçak",
    "çatlak": "yapısal çatlak",
    "kıvılcım": "yangın / kıvılcım riski",
    "kopuk": "kopma / kopuk parça",
    "paslı": "korozyon / yıpranma",
}

# --- Modül 6 ---
BAKIM_CARPANLARI = {
    "Planlı bakım yapıldı": 1.0,
    "Bakım ertelendi": 1.15,
    "Ekipman bakımsız / arıza riski var": 1.35,
}

# --- Tema renkleri ---
ARKA_PLAN = "#1E1E24"
PANEL = "#25252C"
YAZI = "#F2F2F7"
NEON_TURUNCU = "#FF6700"
NEON_MAVI = "#04D9FF"
NEON_YESIL = "#39FF14"
NEON_SARI = "#FFF01F"
NEON_KIRMIZI = "#FF073A"
NEON_MOR = "#BC13FE"

TEMA_CSS = """<style>
.stApp {background-color: #1E1E24; color: #EDEDED;}
[data-testid="stHeader"] {background-color: #1E1E24;}
section[data-testid="stSidebar"] {background-color: #16161B; border-right: 1px solid #33333D;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
[data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color: #F2F2F7 !important;}
.stTextArea textarea, .stTextInput input {background-color: #2A2A32; color: #F2F2F7;}
div[data-baseweb="select"] > div {background-color: #2A2A32; color: #F2F2F7;}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {color: #F2F2F7;}
button[data-baseweb="tab"] p {color: #CFCFD8;}
button[data-baseweb="tab"][aria-selected="true"] p {color: #FF6700; font-weight: 700;}
[data-testid="stExpander"] {background-color: #25252C; border: 1px solid #3A3A44; border-radius: 10px;}
[data-testid="stExpander"] summary p {color: #F2F2F7;}
[data-testid="stForm"] {border: 1px solid #3A3A44; border-radius: 12px;}
.neon-kart {background: #2A2A32; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; border: 2px solid; min-height: 112px;}
.kart-baslik {font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; color: #B8B8C4;}
.kart-deger {font-size: 2rem; font-weight: 800; line-height: 1.3;}
.kart-alt {font-size: 0.78rem; color: #B8B8C4; min-height: 1rem;}
.neon-yesil {border-color: #39FF14; box-shadow: 0 0 14px #39FF14aa, inset 0 0 10px #39FF1422;}
.neon-yesil .kart-deger {color: #39FF14; text-shadow: 0 0 10px #39FF14;}
.neon-sari {border-color: #FFF01F; box-shadow: 0 0 14px #FFF01Faa, inset 0 0 10px #FFF01F22;}
.neon-sari .kart-deger {color: #FFF01F; text-shadow: 0 0 10px #FFF01F;}
.neon-kirmizi {border-color: #FF073A; box-shadow: 0 0 16px #FF073Aaa, inset 0 0 10px #FF073A22; animation: nabiz 1.2s ease-in-out infinite alternate;}
.neon-kirmizi .kart-deger {color: #FF073A; text-shadow: 0 0 12px #FF073A;}
@keyframes nabiz {from {box-shadow: 0 0 8px #FF073A88;} to {box-shadow: 0 0 28px #FF073A;}}
.alarm-dev {background: #B00020; color: #FFFFFF; text-align: center; font-size: 2.3rem; font-weight: 900; padding: 26px 10px; border-radius: 14px; border: 4px solid #FF073A; box-shadow: 0 0 40px #FF073A; animation: yanip 0.9s infinite; margin-bottom: 14px; letter-spacing: 0.04em;}
.alarm-dev span {display: block; font-size: 1.05rem; font-weight: 600; margin-top: 8px;}
@keyframes yanip {0%, 100% {opacity: 1;} 50% {opacity: 0.25;}}
.ihbar-kutu {background: #2A2A32; border-left: 5px solid #5A5A66; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;}
.ihbar-kutu.tetik {border-left-color: #FF6700;}
.ihbar-ust {font-size: 0.78rem; color: #B8B8C4;}
.ihbar-metin {color: #F2F2F7; font-size: 1rem; margin: 4px 0;}
.ihbar-etiket {font-size: 0.82rem; color: #04D9FF;}
.ihbar-kutu.tetik .ihbar-etiket {color: #FF6700; font-weight: 700;}
</style>"""


# ---------------------------------------------------------------------------
# 1) YAPAY VERİ SETİ ÜRETİMİ
# ---------------------------------------------------------------------------
def create_synthetic_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Gerçekçi korelasyonlara sahip yapay fizyolojik risk veri seti üretir.
    - 8 saatten sonraki her ek çalışma saati riski artırır.
    - Aralıksız çalışma yorgunluğun en güçlü belirleyicisidir.
    - Açlık ve susuzluk dikkat dağınıklığını artırır.
    - Susuzluk + aralıksız çalışma birleşince risk doğrusal olmayan artar.
    """
    rng = np.random.default_rng(seed)
    toplam = rng.uniform(4, 16, n_samples)
    araliksiz = np.minimum(rng.uniform(0.5, 10, n_samples), toplam)
    aclik = rng.uniform(0, 10, n_samples)
    susuzluk = rng.uniform(0, 8, n_samples)

    risk = (
        4
        + 2.5 * np.maximum(toplam - 8, 0)
        + 4.0 * araliksiz
        + 2.0 * aclik
        + 3.0 * susuzluk
        + 0.4 * araliksiz * susuzluk
        + rng.normal(0, 4, n_samples)
    )
    risk = np.clip(risk, 0, 100)

    return pd.DataFrame(
        {
            "toplam_calisma_saati": toplam,
            "araliksiz_calisma_saati": araliksiz,
            "aclik_suresi_saat": aclik,
            "susuzluk_suresi_saat": susuzluk,
            "fizyolojik_risk": risk,
        }
    )


# ---------------------------------------------------------------------------
# 2) MODÜL 1: FİZYOLOJİK STRES MODÜLÜ
# ---------------------------------------------------------------------------
class FizyolojikStresModulu:
    """Yorgunluk/dikkat eğrisini hesaplar ve ML ile fizyolojik risk skoru tahmin eder."""

    def __init__(self, veri: pd.DataFrame):
        self.veri = veri
        self.model = RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        )
        self.r2 = self._egit()

    def _egit(self) -> float:
        X = self.veri[OZELLIKLER]
        y = self.veri["fizyolojik_risk"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        return r2_score(y_test, self.model.predict(X_test))

    def risk_tahmin_et(self, toplam, araliksiz, aclik, susuzluk) -> float:
        """Tek işçi için fizyolojik risk (0-100)."""
        girdi = pd.DataFrame([[toplam, araliksiz, aclik, susuzluk]], columns=OZELLIKLER)
        return float(np.clip(self.model.predict(girdi)[0], 0, 100))

    def risk_tahmin_toplu(self, X: pd.DataFrame) -> np.ndarray:
        """Birden çok satır için tek seferde tahmin (dijital ikiz için hızlı)."""
        return np.clip(self.model.predict(X[OZELLIKLER]), 0, 100)

    @staticmethod
    def yorgunluk_egrisi(araliksiz, aclik, susuzluk, yorgunluk_carpani: float = 1.0):
        """Yorgunluk ve dikkat eğrisi. Sıcakta çarpan büyür, eğri dikleşir."""
        t = np.linspace(0, max(araliksiz, 1) + 4, 80)
        stres_carpani = (1 + 0.05 * aclik + 0.07 * susuzluk) * yorgunluk_carpani
        yorgunluk = 100 * (1 - np.exp(-((t * stres_carpani) / 7) ** 1.6))
        dikkat = 100 - 0.85 * yorgunluk
        return t, yorgunluk, dikkat


# ---------------------------------------------------------------------------
# 3) MODÜL 2: SAĞLIK MODÜLÜ
# ---------------------------------------------------------------------------
class SaglikModulu:
    """Kronik hastalık ve ilaç zamanlamasına göre sağlık katsayısı üretir."""

    @staticmethod
    def ilac_cezasi(ilac_sure_saat: float) -> float:
        if ilac_sure_saat < ILAC_YAKLASMA_SAATI:
            return 0.0
        if ilac_sure_saat <= ILAC_GECIKME_SAATI:
            oran = (ilac_sure_saat - ILAC_YAKLASMA_SAATI) / (
                ILAC_GECIKME_SAATI - ILAC_YAKLASMA_SAATI
            )
            return 0.05 * oran
        return min(0.05 + 0.06 * (ilac_sure_saat - ILAC_GECIKME_SAATI), MAX_ILAC_CEZASI)

    def katsayi_hesapla(self, hastaliklar: list, ilac_sure_saat: float) -> float:
        if not hastaliklar:
            return 1.0
        bazal = sum(HASTALIK_PROFILLERI[h]["bazal"] for h in hastaliklar)
        ceza = self.ilac_cezasi(ilac_sure_saat)
        ilac_etkisi = sum(HASTALIK_PROFILLERI[h]["hassasiyet"] * ceza for h in hastaliklar)
        return min(1.0 + bazal + ilac_etkisi, MAX_SAGLIK_KATSAYISI)

    def katsayi_egrisi(self, hastaliklar: list):
        saatler = np.linspace(0, 12, 97)
        return saatler, np.array([self.katsayi_hesapla(hastaliklar, s) for s in saatler])

    @staticmethod
    def ilac_durumu(hastaliklar: list, ilac_sure_saat: float) -> str:
        if not hastaliklar:
            return "yok"
        if ilac_sure_saat > ILAC_GECIKME_SAATI:
            return "gecikti"
        if ilac_sure_saat >= ILAC_YAKLASMA_SAATI:
            return "yaklasti"
        return "normal"


# ---------------------------------------------------------------------------
# 4) MODÜL 3: ÇEVRE MODÜLÜ
# ---------------------------------------------------------------------------
class CevreModulu:
    """Sıcaklık, nem ve gazdan çevresel risk ve yorgunluk çarpanı üretir."""

    @staticmethod
    def hissedilen_sicaklik(sicaklik: float, nem: float) -> float:
        """%50 nem üstü her puan, sıcak ortamda hissedilen sıcaklığı 0.1°C artırır."""
        sicak_orani = float(np.clip((sicaklik - 20) / 7, 0, 1))
        return sicaklik + 0.1 * max(0.0, nem - 50) * sicak_orani

    @staticmethod
    def yorgunluk_carpani(hissedilen: float) -> float:
        """28°C'ye kadar 1.0x, 35°C'de 1.5x, 45°C'de 2.0x (2 kat hızlı yorulma)."""
        ilk = 0.5 * np.clip((hissedilen - SICAKLIK_NORMAL_SINIRI) / 7, 0, 1)
        ikinci = 0.5 * np.clip((hissedilen - SICAKLIK_KRITIK_ESIK) / 10, 0, 1)
        return float(1.0 + ilk + ikinci)

    @staticmethod
    def sicaklik_stresi(hissedilen: float) -> float:
        return float(
            np.clip(
                (hissedilen - SICAKLIK_NORMAL_SINIRI)
                / (SICAKLIK_TEHLIKE_SINIRI - SICAKLIK_NORMAL_SINIRI),
                0, 1,
            )
        )

    @staticmethod
    def gaz_stresi(gaz_ppm: float) -> float:
        return float(np.clip((gaz_ppm - GAZ_UYARI) / (GAZ_KRITIK - GAZ_UYARI), 0, 1))

    @staticmethod
    def hava_kalitesi(gaz_ppm: float):
        if gaz_ppm >= GAZ_KRITIK:
            return "🔴 Kritik", "kritik"
        if gaz_ppm >= GAZ_TEHLIKELI:
            return "🟠 Tehlikeli", "tehlikeli"
        if gaz_ppm >= GAZ_UYARI:
            return "🟡 Dikkat", "dikkat"
        return "🟢 Temiz", "temiz"

    def degerlendir(self, sicaklik: float, nem: float, gaz_ppm: float) -> dict:
        hissedilen = self.hissedilen_sicaklik(sicaklik, nem)
        isi = self.sicaklik_stresi(hissedilen)
        gaz = self.gaz_stresi(gaz_ppm)
        etiket, gaz_durumu = self.hava_kalitesi(gaz_ppm)
        return {
            "hissedilen_sicaklik": hissedilen,
            "yorgunluk_carpani": self.yorgunluk_carpani(hissedilen),
            "cevresel_risk": float(100 * (1 - (1 - isi) * (1 - gaz))),
            "kritik_sicak": hissedilen > SICAKLIK_KRITIK_ESIK,
            "hava_kalitesi": etiket,
            "gaz_durumu": gaz_durumu,
        }


# ---------------------------------------------------------------------------
# 5) RİSK HESAPLARI
# ---------------------------------------------------------------------------
def bireysel_risk_hesapla(
    fiz_modul, saglik_modul, toplam, araliksiz, aclik, susuzluk,
    hastaliklar, ilac_sure_saat, yorgunluk_carpani: float = 1.0,
) -> dict:
    """Bireysel Risk = Fizyolojik Risk x Sağlık Katsayısı. Sıcakta aralıksız süre çarpanla büyür."""
    etkin_araliksiz = min(araliksiz * yorgunluk_carpani, 10.0)
    fizyolojik = fiz_modul.risk_tahmin_et(toplam, etkin_araliksiz, aclik, susuzluk)
    katsayi = saglik_modul.katsayi_hesapla(hastaliklar, ilac_sure_saat)
    return {
        "fizyolojik_risk": fizyolojik,
        "saglik_katsayisi": katsayi,
        "bireysel_risk": float(min(fizyolojik * katsayi, 100.0)),
        "etkin_araliksiz": etkin_araliksiz,
    }


def risk_birlestir(fiz_risk: float, saglik_katsayisi: float, cevre: dict):
    """
    Taban Risk = Fizyolojik + Sağlık etkisi + (CEVRE_AGIRLIGI x Çevresel), en fazla %100.
    Kritik gazda taban risk en az GAZ_ACIL_RISK olur. (toplam, bileşenler) döndürür.
    """
    bireysel = min(fiz_risk * saglik_katsayisi, 100.0)
    saglik_puan = bireysel - fiz_risk
    cevre_puan = CEVRE_AGIRLIGI * cevre["cevresel_risk"]
    ham = fiz_risk + saglik_puan + cevre_puan
    olcek = 100.0 / ham if ham > 100 else 1.0  # Grafik için orantılı küçültme
    toplam = min(ham, 100.0)

    acil = 0.0
    if cevre["gaz_durumu"] == "kritik" and toplam < GAZ_ACIL_RISK:
        acil = GAZ_ACIL_RISK - toplam
        toplam = GAZ_ACIL_RISK

    return float(toplam), {
        "Fizyolojik": fiz_risk * olcek,
        "Sağlık etkisi": saglik_puan * olcek,
        "Çevresel": cevre_puan * olcek,
        "Gaz acil durumu": acil,
    }


def fabrika_risk_hesapla(
    fiz_modul, saglik_modul, cevre_modul, toplam, araliksiz, aclik, susuzluk,
    hastaliklar, ilac_sure_saat, sicaklik, nem, gaz_ppm,
) -> dict:
    """Modül 1 + 2 + 3'ü birleştirir (near-miss etkisi bölge bazında ayrıca eklenir)."""
    cevre = cevre_modul.degerlendir(sicaklik, nem, gaz_ppm)
    birey = bireysel_risk_hesapla(
        fiz_modul, saglik_modul, toplam, araliksiz, aclik, susuzluk,
        hastaliklar, ilac_sure_saat, cevre["yorgunluk_carpani"],
    )
    taban, bilesenler = risk_birlestir(birey["fizyolojik_risk"], birey["saglik_katsayisi"], cevre)
    return {"taban_risk": taban, "bilesenler": bilesenler, "birey": birey, "cevre": cevre}


# ---------------------------------------------------------------------------
# 6) MODÜL 4: NEAR-MISS (NLP) MANTIĞI
# ---------------------------------------------------------------------------
def turkce_kucult(metin: str) -> str:
    """Türkçe harfleri doğru küçültür (İ -> i, I -> ı)."""
    return metin.replace("İ", "i").replace("I", "ı").lower()


def ihbar_analiz_et(metin: str) -> dict:
    """Basit NLP: metinde risk anahtar kelimeleri arar. Bulursa 'yapay zeka tetiklenir'."""
    temiz = turkce_kucult(metin)
    bulunan = [k for k in ANAHTAR_KELIMELER if k in temiz]
    return {"tetiklendi": bool(bulunan), "kelimeler": bulunan}


def ihbar_ekle():
    """Form gönderilince çalışır (on_click callback). İhbarı session_state listesine ekler."""
    metin = st.session_state.get("ihbar_metni", "").strip()
    bolge = st.session_state.get("ihbar_bolge", BOLGELER[0])
    if len(metin) < 5:
        st.session_state["ihbar_mesaj"] = ("warning", "İhbar metni çok kısa, lütfen biraz daha ayrıntı yazın.")
        return
    analiz = ihbar_analiz_et(metin)
    # Anonimlik: kimlik tutulmaz, yalnızca zaman, bölge ve metin saklanır
    st.session_state["ihbarlar"].insert(0, {
        "zaman": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "bolge": bolge,
        "metin": metin,
        "tetiklendi": analiz["tetiklendi"],
        "kelimeler": analiz["kelimeler"],
    })
    if analiz["tetiklendi"]:
        st.session_state["ihbar_mesaj"] = (
            "success",
            f"İhbar alındı. 🤖 Yapay zeka riskli kelimeler buldu ({', '.join(analiz['kelimeler'])}): "
            f"{bolge} bölgesinin riski +{NEARMISS_ARTIS:.0f} puan arttırıldı.",
        )
    else:
        st.session_state["ihbar_mesaj"] = ("info", "İhbar alındı. Risk anahtar kelimesi bulunamadı, kayıt altına alındı.")


def ihbarlari_temizle():
    st.session_state["ihbarlar"] = []


def bolge_risklerini_hesapla(taban_risk: float, ihbarlar: list):
    """Her bölgenin riski = taban risk + (tetikleyen ihbar sayısı x 15 puan), en fazla %100."""
    sayilar = {b: 0 for b in BOLGELER}
    for ihbar in ihbarlar:
        if ihbar["tetiklendi"] and ihbar["bolge"] in sayilar:
            sayilar[ihbar["bolge"]] += 1
    riskler = {}
    for bolge, n in sayilar.items():
        artis = min(NEARMISS_ARTIS * n, NEARMISS_MAX_ARTIS)
        riskler[bolge] = float(min(taban_risk + artis, 100.0))
    return riskler, sayilar


# ---------------------------------------------------------------------------
# 7) MODÜL 6: DİJİTAL İKİZ
# ---------------------------------------------------------------------------
class DijitalIkizModulu:
    """Yarının saat saat risk eğrisini simüle eder."""

    def __init__(self, fiz_modul, saglik_modul, cevre_modul):
        self.fiz = fiz_modul
        self.saglik = saglik_modul
        self.cevre = cevre_modul

    def simule_et(self, sicaklik_max, fazla_mesai, mesai_isci, bakim, nem, gaz_ppm, hastaliklar) -> dict:
        """
        Vardiya 08:00-17:00 (12:00-13:00 öğle arası) + fazla mesai (mola yok).
        Sıcaklık günlük sinüs eğrisiyle 15:00'te tepe yapar.
        """
        bitis = 17 + int(fazla_mesai)
        saatler = np.arange(8, bitis + 1)
        ogle_oncesi = saatler <= 12

        toplam = np.where(ogle_oncesi, saatler - 8, saatler - 9).astype(float)
        araliksiz = np.where(ogle_oncesi, saatler - 8, saatler - 13).astype(float)
        aclik = np.where(ogle_oncesi, saatler - 7.5, saatler - 13).astype(float)
        sicaklik = (sicaklik_max - 10) + 5 * (1 + np.cos(2 * np.pi * (saatler - 15) / 24))
        # Sıcakta daha hızlı susuzlaşma
        susuzluk = np.minimum(8.0, araliksiz * (0.4 + 0.05 * np.maximum(0, sicaklik - 25)))

        cevreler = [self.cevre.degerlendir(float(s), nem, gaz_ppm) for s in sicaklik]
        carpanlar = np.array([c["yorgunluk_carpani"] for c in cevreler])
        etkin = np.minimum(araliksiz * carpanlar, 10.0)

        X = pd.DataFrame({
            OZELLIKLER[0]: toplam, OZELLIKLER[1]: etkin,
            OZELLIKLER[2]: aclik, OZELLIKLER[3]: susuzluk,
        })
        fiz = self.fiz.risk_tahmin_toplu(X)

        riskler = []
        for i, saat in enumerate(saatler):
            ilac = float((int(saat) - 7) % ILAC_DOZ_ARALIGI) if hastaliklar else 0.0
            katsayi = self.saglik.katsayi_hesapla(hastaliklar, ilac)
            risk, _ = risk_birlestir(float(fiz[i]), katsayi, cevreler[i])
            carpan = BAKIM_CARPANLARI[bakim]
            if saat > 17:  # Fazla mesaide kalabalık yorgun grup etkisi
                carpan *= min(1 + 0.02 * (mesai_isci - 1), 1.4)
            riskler.append(min(risk * carpan, 100.0))

        return {"saat": saatler, "risk": np.array(riskler), "sicaklik": sicaklik}


# ---------------------------------------------------------------------------
# 8) YARDIMCILAR: durum, ses, alarm, kartlar
# ---------------------------------------------------------------------------
def durumu_baslat():
    """Tüm uygulama durumunu st.session_state içinde başlatır."""
    varsayilanlar = {
        "ihbarlar": [],
        "alarm_loglari": [],
        "alarm_aktif": False,
        "simulasyon": None,
        "ihbar_mesaj": None,
    }
    for anahtar, deger in varsayilanlar.items():
        if anahtar not in st.session_state:
            st.session_state[anahtar] = deger


@st.cache_data
def alarm_sesi_olustur() -> bytes:
    """Kodla üretilen siren benzeri bir alarm sesi (WAV)."""
    sr = 16000
    parcalar = []
    for _ in range(3):
        for frekans in (880, 660):
            t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
            parcalar.append(0.4 * np.sin(2 * np.pi * frekans * t))
    ses16 = (np.concatenate(parcalar) * 32767).astype(np.int16)
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ses16.tobytes())
    return tampon.getvalue()


def risk_seviyesi(risk: float) -> str:
    if risk >= KRITIK_ESIK:
        return "kirmizi"
    if risk >= 50:
        return "sari"
    return "yesil"


def neon_kart(baslik: str, deger: str, seviye: str, alt: str = "") -> str:
    """Parlayan HTML/CSS metrik kartı (tek satır HTML: Markdown kod bloğu sanmasın)."""
    return (
        f'<div class="neon-kart neon-{seviye}"><div class="kart-baslik">{baslik}</div>'
        f'<div class="kart-deger">{deger}</div><div class="kart-alt">{alt}</div></div>'
    )


def kart_satiri(kartlar: list):
    kolonlar = st.columns(len(kartlar))
    for kol, (baslik, deger, seviye, alt) in zip(kolonlar, kartlar):
        kol.markdown(neon_kart(baslik, deger, seviye, alt), unsafe_allow_html=True)


def alarm_durumunu_guncelle(risk: float, bolge: str) -> bool:
    """Kritik eşiğe girişte ve çıkışta alarm günlüğüne kayıt düşer (her yenilemede tekrar etmez)."""
    kritik = risk >= KRITIK_ESIK
    onceki = st.session_state["alarm_aktif"]
    zaman = datetime.now().strftime("%H:%M:%S")
    if kritik and not onceki:
        st.session_state["alarm_loglari"].extend([
            f"[{zaman}] 🚨 ALARM: Fabrika riski %{risk:.1f}. ÜRETİMİ DURDURMA alarmı tetiklendi (bölge: {bolge})",
            f"[{zaman}] 🔊 Sesli anons yayınlandı: 'Zorunlu havalandırma ve mola'",
            f"[{zaman}] 📲 Telegram Bildirimi Yöneticilere Gönderildi",
            f"[{zaman}] 📱 SMS Bildirimi Yöneticilere Gönderildi",
        ])
    elif not kritik and onceki:
        st.session_state["alarm_loglari"].append(
            f"[{zaman}] ✅ Risk %{risk:.1f}'e düştü, alarm normale döndü."
        )
    st.session_state["alarm_aktif"] = kritik
    st.session_state["alarm_loglari"] = st.session_state["alarm_loglari"][-60:]
    return kritik


def alarm_goster(risk: float, bolge: str, ses_acik: bool):
    """Modül 5: yanıp sönen dev alarm, st.error, ses efekti ve bildirim kayıtları."""
    st.markdown(
        f'<div class="alarm-dev">🚨 CRITICAL: ÜRETİMİ DURDURMA ALARMI 🚨'
        f'<span>Fabrika risk skoru %{risk:.1f} | En riskli bölge: {html.escape(bolge)}</span></div>',
        unsafe_allow_html=True,
    )
    st.error("🛑 Zorunlu havalandırma ve mola başlatılsın. Kritik görevler durdurulsun, bölge kontrol edilsin.")
    st.code("\n".join(st.session_state["alarm_loglari"][-4:]), language=None)
    if ses_acik:
        try:
            st.audio(alarm_sesi_olustur(), format="audio/wav", autoplay=True)
        except TypeError:  # Eski Streamlit sürümlerinde autoplay yok
            st.audio(alarm_sesi_olustur(), format="audio/wav")


# ---------------------------------------------------------------------------
# 9) MATPLOTLIB KARANLIK TEMA GRAFİKLERİ
# ---------------------------------------------------------------------------
def karanlik_fig(boyut=(7, 4)):
    """Koyu arka planlı, beyaz eksenli Matplotlib figürü oluşturur."""
    fig, ax = plt.subplots(figsize=boyut)
    fig.patch.set_facecolor(ARKA_PLAN)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=YAZI)
    for kenar in ax.spines.values():
        kenar.set_color("#5A5A66")
    ax.grid(True, color="#3A3A44", alpha=0.6, linewidth=0.7)
    return fig, ax


def etiketle(ax, baslik, xlabel=None, ylabel=None):
    ax.set_title(baslik, color=YAZI, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=YAZI)
    if ylabel:
        ax.set_ylabel(ylabel, color=YAZI)


def lejant(ax, **kwargs):
    ax.legend(facecolor=PANEL, edgecolor="#5A5A66", labelcolor=YAZI, **kwargs)


def yorgunluk_grafigi(t, yorgunluk, dikkat, mevcut, normal_dikkat=None):
    fig, ax = karanlik_fig()
    ax.plot(t, yorgunluk, color=NEON_TURUNCU, lw=2.8, label="Yorgunluk")
    ax.plot(t, dikkat, color=NEON_MAVI, lw=2.8, label="Dikkat seviyesi")
    if normal_dikkat is not None:
        ax.plot(t, normal_dikkat, color=NEON_MAVI, lw=1.5, ls=":", label="Dikkat (normal ortamda)")
    ax.axvline(mevcut, color="#AAAAAA", ls="--", lw=1)
    ax.set_ylim(0, 100)
    etiketle(ax, "Yorgunluk ve Dikkat Dağınıklığı Eğrisi", "Aralıksız çalışma (saat)", "Seviye (%)")
    lejant(ax, loc="center right")
    fig.tight_layout()
    return fig


def ilac_grafigi(saatler, katsayilar, mevcut_saat, mevcut_katsayi):
    fig, ax = karanlik_fig()
    ax.plot(saatler, katsayilar, color=NEON_MAVI, lw=2.8, label="Sağlık katsayısı")
    ax.scatter([mevcut_saat], [mevcut_katsayi], color=NEON_TURUNCU, s=90, zorder=5, label="Şu anki durum")
    ax.axvspan(ILAC_YAKLASMA_SAATI, ILAC_GECIKME_SAATI, color=NEON_SARI, alpha=0.15, label="İlaç saati yaklaşıyor")
    ax.axvspan(ILAC_GECIKME_SAATI, 12, color=NEON_KIRMIZI, alpha=0.12, label="İlaç gecikti")
    etiketle(ax, "İlaç Süresine Göre Sağlık Katsayısı", "Son ilaçtan beri (saat)", "Katsayı (1.0 = ek risk yok)")
    lejant(ax, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def bilesen_grafigi(bilesenler: dict, toplam: float):
    renkler = {
        "Fizyolojik": NEON_MAVI, "Sağlık etkisi": NEON_MOR, "Çevresel": NEON_TURUNCU,
        "Gaz acil durumu": NEON_KIRMIZI, "Near-miss": NEON_SARI,
    }
    fig, ax = karanlik_fig((7, 2.8))
    sol = 0.0
    for ad, deger in bilesenler.items():
        if deger > 0.05:
            ax.barh([0], [deger], left=sol, color=renkler[ad], height=0.5, label=f"{ad} ({deger:.1f})")
            sol += deger
    ax.axvline(KRITIK_ESIK, color=NEON_KIRMIZI, ls="--", lw=1.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    etiketle(ax, f"Risk Bileşenleri (Toplam: %{toplam:.1f})", "Risk (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3,
              facecolor=PANEL, edgecolor="#5A5A66", labelcolor=YAZI, fontsize=8)
    fig.tight_layout()
    return fig


def bolge_grafigi(riskler: dict):
    fig, ax = karanlik_fig((7, 2.8))
    adlar = list(riskler.keys())
    degerler = [riskler[a] for a in adlar]
    renkler = [NEON_KIRMIZI if d >= KRITIK_ESIK else NEON_SARI if d >= 50 else NEON_YESIL for d in degerler]
    ax.barh(adlar, degerler, color=renkler, height=0.55)
    for i, d in enumerate(degerler):
        ax.text(min(d + 1, 90), i, f"%{d:.1f}", va="center", color=YAZI, fontsize=9)
    ax.axvline(KRITIK_ESIK, color=NEON_KIRMIZI, ls="--", lw=1.5)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    etiketle(ax, "Bölge Bazlı Risk Haritası (Near-miss dahil)", "Risk (%)")
    fig.tight_layout()
    return fig


def ikiz_grafigi(senaryo: dict, normal: dict, mesai_var: bool):
    fig, ax = karanlik_fig((9, 4))
    saat, risk = senaryo["saat"], senaryo["risk"]
    ax.plot(saat, risk, color=NEON_TURUNCU, lw=3, marker="o", label="Yarının senaryosu")
    ax.plot(normal["saat"], normal["risk"], color=NEON_MAVI, lw=2, ls=":", marker="o", ms=3,
            label="Normal gün (referans)")
    ax.axhline(KRITIK_ESIK, color=NEON_KIRMIZI, ls="--", lw=1.5, label=f"Kritik eşik %{KRITIK_ESIK}")
    ax.fill_between(saat, risk, KRITIK_ESIK, where=risk >= KRITIK_ESIK,
                    color=NEON_KIRMIZI, alpha=0.35, interpolate=True)
    if mesai_var:
        ax.axvline(17, color="#AAAAAA", ls="--", lw=1)
        ax.text(17.05, 3, "mesai sonu", color=YAZI, fontsize=8)
    ax.set_xticks(saat)
    ax.set_xticklabels([f"{int(h):02d}:00" for h in saat], rotation=45, color=YAZI)
    ax.set_ylim(0, 100)
    etiketle(ax, "Yarının Tahmini Kaza Risk Eğrisi", "Saat", "Risk (%)")
    lejant(ax, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10) ÖNBELLEK
# ---------------------------------------------------------------------------
@st.cache_resource
def modulleri_hazirla():
    fiz = FizyolojikStresModulu(create_synthetic_dataset())
    saglik = SaglikModulu()
    cevre = CevreModulu()
    return fiz, saglik, cevre, DijitalIkizModulu(fiz, saglik, cevre)


# ---------------------------------------------------------------------------
# 11) ARAYÜZ
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="İşçiKalkanı AI", page_icon="🛡️", layout="wide")
    st.markdown(TEMA_CSS, unsafe_allow_html=True)
    durumu_baslat()
    fiz_modul, saglik_modul, cevre_modul, ikiz_modul = modulleri_hazirla()

    st.title("🛡️ İşçiKalkanı AI")
    st.caption("Prediktif İş Güvenliği Paneli | Fizyolojik • Sağlık • Çevre • Near-Miss • Alarm • Dijital İkiz")

    # ---------------- Yan panel ----------------
    st.sidebar.header("👷 İşçi Fizyolojik Verileri")
    toplam = st.sidebar.slider("Toplam çalışma saati (bugün)", 0.0, 16.0, 8.0, 0.5)
    araliksiz = st.sidebar.slider("Aralıksız çalışma süresi (saat)", 0.0, 10.0, 3.0, 0.5)
    aclik = st.sidebar.slider("Son yemekten beri geçen süre (saat)", 0.0, 10.0, 3.0, 0.5)
    susuzluk = st.sidebar.slider("Son su içmeden beri geçen süre (saat)", 0.0, 8.0, 2.0, 0.5)
    if araliksiz > toplam:
        st.sidebar.warning("Aralıksız çalışma toplam süreden uzun olamaz; eşitlendi.")
        araliksiz = toplam

    st.sidebar.divider()
    st.sidebar.header("🩺 Sağlık ve İlaç Bilgisi")
    hastaliklar = st.sidebar.multiselect("Kronik rahatsızlık (varsa)", list(HASTALIK_PROFILLERI.keys()))
    ilac_sure = st.sidebar.slider(
        "Son ilaç alımından beri geçen süre (saat)", 0.0, 12.0, 1.0, 0.5, disabled=not hastaliklar
    )

    st.sidebar.divider()
    st.sidebar.header("🏭 Çevresel Ortam")
    sicaklik = st.sidebar.slider("Ortam sıcaklığı (°C)", 10.0, 50.0, 25.0, 0.5)
    nem = st.sidebar.slider("Ortam nemi (%)", 10, 100, 50, 1)
    gaz_ppm = st.sidebar.slider("Zehirli gaz seviyesi (ppm)", 0.0, 100.0, 2.0, 0.5)

    st.sidebar.divider()
    ses_acik = st.sidebar.checkbox("🔊 Sesli alarm efekti", value=True)

    # ---------------- Hesaplamalar ----------------
    sonuc = fabrika_risk_hesapla(
        fiz_modul, saglik_modul, cevre_modul, toplam, araliksiz, aclik, susuzluk,
        hastaliklar, ilac_sure, sicaklik, nem, gaz_ppm,
    )
    birey, cevre, taban_risk = sonuc["birey"], sonuc["cevre"], sonuc["taban_risk"]

    # Modül 4: near-miss etkisi bölge bazında eklenir
    bolge_riskleri, bolge_tetik = bolge_risklerini_hesapla(taban_risk, st.session_state["ihbarlar"])
    en_riskli_bolge = max(bolge_riskleri, key=bolge_riskleri.get)
    fabrika_risk = bolge_riskleri[en_riskli_bolge]  # Güvenlik öncelikli: en riskli bölge belirleyici
    bilesenler = dict(sonuc["bilesenler"])
    bilesenler["Near-miss"] = fabrika_risk - taban_risk

    # Diğer modüller / sonraki adımlar için durumu sakla
    st.session_state["fizyolojik_risk"] = birey["fizyolojik_risk"]
    st.session_state["bireysel_risk"] = birey["bireysel_risk"]
    st.session_state["cevresel_risk"] = cevre["cevresel_risk"]
    st.session_state["fabrika_toplam_risk"] = fabrika_risk

    # Modül 5: alarm durumu
    kritik = alarm_durumunu_guncelle(fabrika_risk, en_riskli_bolge)
    if kritik:
        alarm_goster(fabrika_risk, en_riskli_bolge, ses_acik)

    # ---------------- Neon metrik kartları ----------------
    carpan = cevre["yorgunluk_carpani"]
    etkin = birey["etkin_araliksiz"]
    nem_etkisi = cevre["hissedilen_sicaklik"] - sicaklik
    sag_k = birey["saglik_katsayisi"]
    hissedilen = cevre["hissedilen_sicaklik"]

    kart_satiri([
        ("🏭 Fabrika Toplam Risk", f"%{fabrika_risk:.1f}", risk_seviyesi(fabrika_risk), f"En riskli bölge: {en_riskli_bolge}"),
        ("Bireysel Risk Skoru", f"%{birey['bireysel_risk']:.1f}", risk_seviyesi(birey["bireysel_risk"]), "Fizyolojik x sağlık katsayısı"),
        ("Çevresel Risk", f"%{cevre['cevresel_risk']:.1f}", risk_seviyesi(cevre["cevresel_risk"]), "Isı + gaz stresi"),
        ("Fabrika Hava Kalitesi", cevre["hava_kalitesi"],
         {"temiz": "yesil", "dikkat": "sari"}.get(cevre["gaz_durumu"], "kirmizi"), f"{gaz_ppm:.1f} ppm"),
    ])
    kart_satiri([
        ("Fizyolojik Risk (temel)", f"%{birey['fizyolojik_risk']:.1f}", risk_seviyesi(birey["fizyolojik_risk"]), "ML tahmini (sıcak etkisi dahil)"),
        ("Sağlık Katsayısı", f"x{sag_k:.2f}", "yesil" if sag_k < 1.15 else "sari" if sag_k < 1.4 else "kirmizi",
         "Kronik rahatsızlık / ilaç"),
        ("Hissedilen Sıcaklık", f"{hissedilen:.1f} °C",
         "yesil" if hissedilen < SICAKLIK_NORMAL_SINIRI else "sari" if hissedilen <= SICAKLIK_KRITIK_ESIK else "kirmizi",
         f"+{nem_etkisi:.1f} °C nem etkisi" if nem_etkisi > 0.05 else "nem etkisi yok"),
        ("Aralıksız Çalışma", f"{araliksiz:.1f} saat",
         "yesil" if etkin <= NORMAL_ARALIKSIZ_LIMIT else "sari" if etkin <= 7 else "kirmizi",
         f"etkin süre {etkin:.1f} saat (x{carpan:.2f})" if carpan > 1.01 else "normal ortam"),
    ])

    # ---------------- Uyarı alanı ----------------
    durum = saglik_modul.ilac_durumu(hastaliklar, ilac_sure)
    if durum == "gecikti":
        st.warning(f"💊 İlaç gecikti: son alımdan {ilac_sure:.1f} saat geçti.")
    elif durum == "yaklasti":
        st.info("💊 İlaç saati yaklaşıyor. İşçinin ilacını alması sağlanmalı.")

    if cevre["kritik_sicak"]:
        st.warning(
            f"🌡️ Aşırı sıcak: hissedilen {hissedilen:.1f}°C. Yorgunluk x{carpan:.2f} hızlı birikiyor. "
            f"Önerilen azami aralıksız çalışma yaklaşık {NORMAL_ARALIKSIZ_LIMIT / carpan:.1f} saat."
        )
    elif carpan > 1.01:
        st.info(f"🌡️ Sıcak ortam: yorgunluk x{carpan:.2f} hızlı birikiyor.")

    if cevre["gaz_durumu"] == "kritik":
        st.error(f"☣️ KRİTİK GAZ SEVİYESİ ({gaz_ppm:.1f} ppm): Bölgeyi tahliye edin, havalandırmayı çalıştırın.")
    elif cevre["gaz_durumu"] == "tehlikeli":
        st.warning(f"☣️ Tehlikeli gaz seviyesi ({gaz_ppm:.1f} ppm): Havalandırmayı artırın, maske zorunlu olsun.")
    elif cevre["gaz_durumu"] == "dikkat":
        st.info(f"☣️ Gaz seviyesi yükseliyor ({gaz_ppm:.1f} ppm): Ölçümleri sıklaştırın.")

    if not kritik:
        if fabrika_risk >= 50:
            st.warning("⚠️ Yüksek risk: Kısa mola, su, yiyecek ve havalandırma önerilir.")
        else:
            st.success("✅ Fabrika toplam riski kabul edilebilir seviyede.")

    # ---------------- Sekmeler ----------------
    sekme_analiz, sekme_ihbar, sekme_ikiz, sekme_log = st.tabs([
        "📊 Analiz Paneli", "📝 Anonim Near-Miss İhbarı",
        "🔮 Gelecek Senaryo Simülatörü", "📜 Alarm Günlüğü",
    ])

    # --- Sekme 1: Analiz ---
    with sekme_analiz:
        t, yorgunluk, dikkat = fiz_modul.yorgunluk_egrisi(araliksiz, aclik, susuzluk, carpan)
        normal_dikkat = None
        if carpan > 1.01:
            _, _, normal_dikkat = fiz_modul.yorgunluk_egrisi(araliksiz, aclik, susuzluk, 1.0)

        g1, g2 = st.columns(2)
        with g1:
            fig = yorgunluk_grafigi(t, yorgunluk, dikkat, araliksiz, normal_dikkat)
            st.pyplot(fig)
            plt.close(fig)
        with g2:
            if hastaliklar:
                saatler, katsayilar = saglik_modul.katsayi_egrisi(hastaliklar)
                fig = ilac_grafigi(saatler, katsayilar, ilac_sure, sag_k)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("İlaç grafiği için sol menüden bir kronik rahatsızlık seçin.")

        g3, g4 = st.columns(2)
        with g3:
            fig = bilesen_grafigi(bilesenler, fabrika_risk)
            st.pyplot(fig)
            plt.close(fig)
        with g4:
            fig = bolge_grafigi(bolge_riskleri)
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("🤖 Model ve Yapay Veri Seti Bilgisi"):
            st.caption(f"Model: RandomForestRegressor | Test R² skoru: {fiz_modul.r2:.3f}")
            st.dataframe(fiz_modul.veri.head(10).round(2))

    # --- Sekme 2: Near-miss ---
    with sekme_ihbar:
        st.subheader("📝 Anonim Kıl Payı Atlatma (Near-Miss) Bildirimi")
        st.caption(
            "Kimlik bilgisi istenmez ve saklanmaz. 'gevşek', 'yağ', 'kablo', 'arıza' gibi risk "
            f"kelimeleri bulunursa ilgili bölgenin riski +{NEARMISS_ARTIS:.0f} puan artar."
        )
        with st.form("ihbar_formu", clear_on_submit=True):
            st.selectbox("Bölge", BOLGELER, key="ihbar_bolge")
            st.text_area(
                "Sahada gördüğünüz riski yazın", key="ihbar_metni", height=110,
                placeholder='Örn: "Ana vinçte halat gevşek", "Zemin yağlı"',
            )
            st.form_submit_button("🛡️ Anonim Olarak Gönder", on_click=ihbar_ekle)

        mesaj = st.session_state.get("ihbar_mesaj")
        if mesaj:
            getattr(st, mesaj[0])(mesaj[1])
            st.session_state["ihbar_mesaj"] = None  # Mesaj bir kez gösterilir

        ihbarlar = st.session_state["ihbarlar"]
        tetik_toplam = sum(1 for i in ihbarlar if i["tetiklendi"])
        st.caption(f"Toplam ihbar: {len(ihbarlar)} | Yapay zekanın tetiklediği: {tetik_toplam}")
        st.button("🗑️ İhbarları temizle", on_click=ihbarlari_temizle)

        if not ihbarlar:
            st.info("Henüz ihbar yok.")
        for ihbar in ihbarlar[:20]:
            guvenli_metin = html.escape(ihbar["metin"]).replace("\n", "<br>")
            if ihbar["tetiklendi"]:
                etiket = f'🤖 YZ tetiklendi: {", ".join(ihbar["kelimeler"])} → +{NEARMISS_ARTIS:.0f} puan risk'
                sinif = "ihbar-kutu tetik"
            else:
                etiket = "✅ Risk anahtar kelimesi bulunmadı"
                sinif = "ihbar-kutu"
            st.markdown(
                f'<div class="{sinif}"><div class="ihbar-ust">🕒 {ihbar["zaman"]} · 📍 {html.escape(ihbar["bolge"])}</div>'
                f'<div class="ihbar-metin">{guvenli_metin}</div><div class="ihbar-etiket">{etiket}</div></div>',
                unsafe_allow_html=True,
            )

    # --- Sekme 3: Dijital ikiz ---
    with sekme_ikiz:
        st.subheader("🔮 Gelecek Senaryo Simülatörü (Dijital İkiz)")
        st.caption("Soldaki işçi sağlık profili, nem ve gaz değerleri simülasyonda baz alınır.")
        k1, k2 = st.columns(2)
        with k1:
            yarin_sicaklik = st.slider("Yarınki tahmini en yüksek sıcaklık (°C)", 15, 50, 36, key="ikiz_sicaklik")
            fazla_mesai = st.slider("Fazla mesai süresi (saat)", 0, 6, 2, key="ikiz_mesai")
        with k2:
            mesai_isci = st.slider("Fazla mesaiye kalacak işçi sayısı", 1, 30, 3, key="ikiz_isci")
            bakim = st.selectbox("Ekipman bakım durumu", list(BAKIM_CARPANLARI.keys()), key="ikiz_bakim")

        if st.button("⚙️ Simüle Et", type="primary", key="ikiz_buton"):
            with st.spinner("Dijital ikiz yarını simüle ediyor..."):
                senaryo = ikiz_modul.simule_et(
                    yarin_sicaklik, fazla_mesai, mesai_isci, bakim, nem, gaz_ppm, hastaliklar
                )
                normal = ikiz_modul.simule_et(
                    26, 0, 1, "Planlı bakım yapıldı", nem, gaz_ppm, hastaliklar
                )
            st.session_state["simulasyon"] = {
                "senaryo": senaryo, "normal": normal, "mesai": fazla_mesai,
                "sicaklik": yarin_sicaklik, "bakim": bakim,
            }

        sim = st.session_state["simulasyon"]
        if sim is not None:
            senaryo, normal = sim["senaryo"], sim["normal"]
            tepe = float(senaryo["risk"].max())
            tepe_saat = int(senaryo["saat"][int(senaryo["risk"].argmax())])
            asan = int((senaryo["risk"] >= KRITIK_ESIK).sum())
            fark = tepe - float(normal["risk"].max())

            kart_satiri([
                ("Tahmini Tepe Risk", f"%{tepe:.1f}", risk_seviyesi(tepe), f"saat {tepe_saat:02d}:00"),
                ("Kritik Eşik Aşılan Süre", f"{asan} saat", "kirmizi" if asan else "yesil", f"eşik: %{KRITIK_ESIK}"),
                ("Normal Güne Göre Fark", f"{fark:+.1f} puan", "kirmizi" if fark > 25 else "sari" if fark > 5 else "yesil",
                 "referans: 26°C, mesaisiz, bakımlı"),
            ])

            fig = ikiz_grafigi(senaryo, normal, sim["mesai"] > 0)
            st.pyplot(fig)
            plt.close(fig)

            oneriler = []
            if sim["sicaklik"] >= 35:
                oneriler.append("sıcak saatlerde (13:00-17:00) ağır işleri kaydırın, sık mola ve su planlayın")
            if sim["mesai"] > 0 and asan > 0:
                oneriler.append("fazla mesaiyi kısaltın veya iptal edin")
            if sim["bakim"] != "Planlı bakım yapıldı":
                oneriler.append("ekipman bakımını bugün öne çekin")
            if asan > 0:
                st.error(f"Yarın risk {asan} saat kritik eşiğin üzerinde kalıyor. Öneri: " + "; ".join(oneriler) + ".")
            elif tepe >= 50:
                st.warning("Yarın risk yüksek seyredecek. Yakın takip ve ek önlem önerilir.")
            else:
                st.success("Yarınki senaryo kabul edilebilir risk seviyesinde.")
        else:
            st.info("Parametreleri seçip 'Simüle Et' düğmesine basın.")

    # --- Sekme 4: Alarm günlüğü ---
    with sekme_log:
        st.subheader("📜 Alarm ve Bildirim Günlüğü (simülasyon)")
        loglar = st.session_state["alarm_loglari"]
        if loglar:
            st.code("\n".join(reversed(loglar)), language=None)
        else:
            st.info("Henüz alarm kaydı yok. Risk %80'i geçince burada görünecek.")


if __name__ == "__main__":
    main()