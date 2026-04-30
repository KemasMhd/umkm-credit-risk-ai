"""
UMKM Credit Risk AI — Streamlit Demo App
Datathon: Ekonomi Digital & Inklusi Keuangan

Jalankan:
  Local  : streamlit run app.py
  Azure  : streamlit run app.py --server.port 8080 --server.address 0.0.0.0
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UMKM Credit Risk AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .approve-badge {
        background: #d4edda; color: #155724;
        padding: 0.5rem 1.5rem; border-radius: 20px;
        font-size: 1.2rem; font-weight: bold;
        display: inline-block;
    }
    .reject-badge {
        background: #f8d7da; color: #721c24;
        padding: 0.5rem 1.5rem; border-radius: 20px;
        font-size: 1.2rem; font-weight: bold;
        display: inline-block;
    }
    .chat-user {
        background: #e3f2fd; padding: 0.8rem 1rem;
        border-radius: 12px 12px 0 12px;
        margin: 0.5rem 0; margin-left: 2rem;
    }
    .chat-bot {
        background: #f5f5f5; padding: 0.8rem 1rem;
        border-radius: 12px 12px 12px 0;
        margin: 0.5rem 0; margin-right: 2rem;
    }
    .info-box {
        background: #e8f4fd; padding: 1rem;
        border-radius: 8px; border-left: 4px solid #2d6a9f;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd; padding: 1rem;
        border-radius: 8px; border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── 20 Fitur Lengkap (sesuai model signature) ────────────────────────────────
FEATURE_COLS = [
    "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_grade",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "loan_to_income_ratio", "monthly_interest_burden",
    "debt_service_ratio", "emp_stability_score", "age_group",
    "income_log", "income_quartile", "interest_risk_band",
    "domain_risk_score"
]

# ─── Sample UMKM Profiles ─────────────────────────────────────────────────────
SAMPLE_PROFILES = {
    "🟢 UMKM Sehat — Warung Makan (Solo)": {
        "person_age": 34,
        "person_income": 72_000_000,
        "person_home_ownership": "OWN",
        "person_emp_length": 6.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10_000_000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.14,
        "cb_person_default_on_file": False,
        "cb_person_cred_hist_length": 5,
        "loan_to_income_ratio": 0.14,
        "monthly_interest_burden": 87_500.0,
        "debt_service_ratio": 0.22,
        "emp_stability_score": "high",
        "age_group": "26-35",
        "income_log": 18.09,
        "income_quartile": "Q3",
        "interest_risk_band": "low",
        "domain_risk_score": 1,
        "desc": "Warung makan 6 tahun, pendapatan stabil, cicilan ringan"
    },
    "🟡 UMKM Borderline — Toko Kelontong (Medan)": {
        "person_age": 28,
        "person_income": 45_000_000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "VENTURE",
        "loan_grade": "C",
        "loan_amnt": 18_000_000,
        "loan_int_rate": 14.2,
        "loan_percent_income": 0.40,
        "cb_person_default_on_file": False,
        "cb_person_cred_hist_length": 2,
        "loan_to_income_ratio": 0.40,
        "monthly_interest_burden": 213_000.0,
        "debt_service_ratio": 0.41,
        "emp_stability_score": "medium",
        "age_group": "26-35",
        "income_log": 17.62,
        "income_quartile": "Q2",
        "interest_risk_band": "medium",
        "domain_risk_score": 2,
        "desc": "Toko kelontong 3 tahun, pendapatan cukup tapi beban cicilan tinggi"
    },
    "🔴 UMKM Berisiko — Usaha Baru (Surabaya)": {
        "person_age": 24,
        "person_income": 28_000_000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1.0,
        "loan_intent": "VENTURE",
        "loan_grade": "E",
        "loan_amnt": 22_000_000,
        "loan_int_rate": 19.5,
        "loan_percent_income": 0.79,
        "cb_person_default_on_file": True,
        "cb_person_cred_hist_length": 1,
        "loan_to_income_ratio": 0.79,
        "monthly_interest_burden": 357_500.0,
        "debt_service_ratio": 0.68,
        "emp_stability_score": "low",
        "age_group": "<=25",
        "income_log": 17.15,
        "income_quartile": "Q1",
        "interest_risk_band": "high",
        "domain_risk_score": 3,
        "desc": "Usaha baru 1 tahun, pinjaman besar relatif terhadap pendapatan"
    }
}

# Helper: derive engineered features dari raw input
def derive_features(age, income, emp_length, loan_amnt, loan_int_rate,
                    home_ownership, loan_intent, loan_grade,
                    cb_default, cred_hist, domain_risk):
    monthly_interest = loan_amnt * (loan_int_rate / 100 / 12)
    monthly_income   = income / 12
    dsr  = monthly_interest / monthly_income if monthly_income > 0 else 0
    lti  = loan_amnt / income if income > 0 else 0
    lpi  = loan_amnt / income if income > 0 else 0
    ilog = np.log(income) if income > 0 else 0

    # income_quartile
    if income < 36_000_000:
        iq = "Q1"
    elif income < 60_000_000:
        iq = "Q2"
    elif income < 96_000_000:
        iq = "Q3"
    else:
        iq = "Q4"

    # age_group
    if age <= 25:
        ag = "<=25"
    elif age <= 35:
        ag = "26-35"
    elif age <= 45:
        ag = "36-45"
    else:
        ag = ">45"

    # emp_stability_score
    if emp_length >= 5:
        ess = "high"
    elif emp_length >= 2:
        ess = "medium"
    else:
        ess = "low"

    # interest_risk_band
    if loan_int_rate < 11:
        irb = "low"
    elif loan_int_rate < 16:
        irb = "medium"
    else:
        irb = "high"

    return {
        "person_age": int(age),
        "person_income": int(income),
        "person_home_ownership": home_ownership,
        "person_emp_length": float(emp_length),
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": int(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": round(lpi, 4),
        "cb_person_default_on_file": cb_default,
        "cb_person_cred_hist_length": int(cred_hist),
        "loan_to_income_ratio": round(lti, 4),
        "monthly_interest_burden": round(monthly_interest, 2),
        "debt_service_ratio": round(dsr, 4),
        "emp_stability_score": ess,
        "age_group": ag,
        "income_log": round(ilog, 4),
        "income_quartile": iq,
        "interest_risk_band": irb,
        "domain_risk_score": int(domain_risk)
    }

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(conn_str: str, container: str):
    if conn_str:
        try:
            from azure.storage.blob import BlobServiceClient
            import joblib, tempfile

            client = BlobServiceClient.from_connection_string(conn_str)
            blob   = client.get_blob_client(container=container,
                                             blob="models/best_model.pkl")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                tmp.write(blob.download_blob().readall())
                tmp_path = tmp.name

            model = joblib.load(tmp_path)
            return model, "✅ Model loaded dari Azure Blob (StackEnsemble AutoML)"
        except Exception as e:
            return None, f"⚠️ Blob load gagal: {e}"

    # Fallback: coba load lokal
    try:
        import joblib
        model = joblib.load('./best_model.pkl')
        return model, "✅ Model loaded dari local file"
    except Exception:
        pass

    # Final fallback: rule-based proxy
    class DummyModel:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.values
            probs = []
            for row in X:
                lti = float(row[11]) if len(row) > 11 else 0.3
                dsr = float(row[13]) if len(row) > 13 else 0.3
                ir  = float(row[7])  if len(row) > 7  else 12.0
                inc = float(row[1])  if len(row) > 1  else 50e6
                s = np.clip(
                    lti*0.40 + dsr*0.30 + (ir/25)*0.20 + (1-min(inc/100e6,1))*0.10
                    + np.random.normal(0, 0.02), 0.02, 0.98
                )
                probs.append([1-s, s])
            return np.array(probs)
        def predict(self, X):
            return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

    return DummyModel(), "⚠️ Demo mode: rule-based proxy (set Azure credentials untuk model asli)"

# ─── Load Dataset ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset(conn_str: str, container: str):
    if not conn_str:
        return None, "No credentials"
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(conn_str)
        blob   = client.get_blob_client(
            container=container,
            blob="processed/processed_credit_risk.csv"
        )
        df = pd.read_csv(io.BytesIO(blob.download_blob().readall()))
        return df, f"✅ Dataset loaded: {len(df):,} baris"
    except Exception as e:
        return None, f"⚠️ {e}"

# ─── GenAI ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_genai_client(token: str):
    if not token:
        return None
    try:
        from openai import OpenAI
        return OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token
        )
    except Exception:
        return None

def call_genai(client, system, user, history=None, max_tokens=600):
    if client is None:
        return "⚠️ Set GITHUB_TOKEN untuk mengaktifkan AI Advisor."
    messages = [{"role": "system", "content": system}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user})
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            max_tokens=max_tokens, temperature=0.7
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

# ─── SHAP Plot ────────────────────────────────────────────────────────────────
def plot_shap_bar(shap_vals, feature_names, title="SHAP Feature Contribution"):
    df_s = pd.DataFrame({'Feature': feature_names, 'SHAP': shap_vals})
    df_s = df_s.reindex(df_s['SHAP'].abs().sort_values().index)
    colors = ['#d73027' if v > 0 else '#4575b4' for v in df_s['SHAP']]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_s['Feature'], df_s['SHAP'], color=colors, alpha=0.85)
    ax.axvline(x=0, color='black', lw=0.8)
    ax.set_xlabel('SHAP Value')
    ax.set_title(title, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    r = mpatches.Patch(color='#d73027', label='↑ Meningkatkan risiko')
    b = mpatches.Patch(color='#4575b4', label='↓ Menurunkan risiko')
    ax.legend(handles=[r, b], fontsize=8)
    plt.tight_layout()
    return fig

def estimate_shap(input_df, model):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(input_df)
        return vals[1][0] if isinstance(vals, list) else vals[0]
    except Exception:
        base = model.predict_proba(input_df)[0][1]
        result = []
        for i in range(input_df.shape[1]):
            p = input_df.copy()
            try:
                p.iloc[0, i] = p.iloc[0, i] * 0.7
            except Exception:
                pass
            try:
                result.append(base - model.predict_proba(p)[0][1])
            except Exception:
                result.append(0.0)
        return np.array(result)

def risk_color(prob):
    if prob < 0.35:
        return "#28a745", "RENDAH"
    elif prob < 0.55:
        return "#ffc107", "SEDANG"
    else:
        return "#dc3545", "TINGGI"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 UMKM Credit Risk AI")
    st.markdown("---")

    st.markdown("### ⚙️ Konfigurasi")
    with st.expander("🔑 API Credentials", expanded=False):
        github_token_input = st.text_input(
            "GitHub Token (GenAI)", type="password",
            value=os.environ.get("GITHUB_TOKEN", ""),
            help="GitHub PAT untuk GPT-4o-mini"
        )
        azure_conn_input = st.text_input(
            "Azure Storage Connection String", type="password",
            value=os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        )
        azure_container_input = st.text_input(
            "Container Name",
            value=os.environ.get("AZURE_CONTAINER_NAME", "dataset")
        )

    if github_token_input:
        os.environ["GITHUB_TOKEN"] = github_token_input
    if azure_conn_input:
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = azure_conn_input
    if azure_container_input:
        os.environ["AZURE_CONTAINER_NAME"] = azure_container_input

    github_token = os.environ.get("GITHUB_TOKEN", "")
    azure_conn   = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    azure_cont   = os.environ.get("AZURE_CONTAINER_NAME", "dataset")

    st.markdown("### 📡 Status")
    st.success("✅ GitHub Models") if github_token else st.warning("⚠️ GitHub Token belum diset")
    st.success("✅ Azure Blob")    if azure_conn   else st.info("ℹ️ Azure Storage — demo mode")

    st.markdown("---")
    st.markdown("### 🏆 Model Performance")
    st.markdown("""
    <div style='background:#e8f4fd;padding:0.8rem;border-radius:8px;font-size:0.85rem'>
    <b>StackEnsemble AutoML</b><br>
    AUC &nbsp;: <b>0.9507</b><br>
    Acc &nbsp;: <b>93.61%</b><br>
    F1 &nbsp;&nbsp;: <b>93.33%</b><br>
    MCC &nbsp;: <b>0.8062</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Deployment Path")
    st.markdown("""
    <div style='background:#e8f4fd;padding:0.8rem;border-radius:8px;font-size:0.85rem'>
    <b>Dev &nbsp;:</b> GitHub Models<br>
    <b>Prod :</b> Azure OpenAI<br>
    <i>OpenAI-compatible API<br>Zero code change!</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Home", "🔍 Credit Scorer", "🤖 AI Advisor", "📊 Analytics"],
        label_visibility="collapsed"
    )

# Load resources
model, model_status = load_model(azure_conn, azure_cont)
genai_client = get_genai_client(github_token)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🏦 UMKM Credit Risk AI</h1>
        <p style='font-size:1.1rem;margin:0'>
        Sistem Scoring Kredit Cerdas untuk Inklusi Keuangan UMKM Indonesia<br>
        <small>Powered by Azure ML AutoML (StackEnsemble) + GitHub Models (GPT-4o-mini)</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(model_status)

    st.markdown("### 🏆 Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC",       "0.9507", "↑ Excellent")
    c2.metric("Accuracy",  "93.61%")
    c3.metric("F1 Score",  "93.33%")
    c4.metric("Precision", "93.66%")
    c5.metric("MCC",       "0.8062")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Framework 4C")
        st.markdown("""
        Model dibangun dengan **20 fitur** dari framework **4C**:

        | Dimensi | Fitur Utama |
        |---------|-------------|
        | **Capacity** | loan_to_income_ratio, debt_service_ratio |
        | **Capital** | loan_percent_income, loan_amnt |
        | **Conditions** | loan_int_rate, loan_grade, interest_risk_band |
        | **Character** | cb_person_default_on_file, emp_stability_score |

        > **Temuan:** `loan_to_income_ratio` = prediktor #1.
        > Kemampuan bayar relatif > income absolut.
        """)

    with col2:
        st.markdown("### ⚖️ Fairness & Inklusi")
        st.markdown("""
        Analisis fairness menunjukkan:

        - 📊 **Income Gap (Q1 vs Q4):** ~30pp
          Refleksi realita ketimpangan, bukan bias diskriminatif

        - 🎂 **Age Gap:** <5%
          Model **age-blind** — adil untuk semua usia

        - 💡 Model dapat menjadi *credit access equalizer*
          dengan program pendampingan untuk UMKM Q1
        """)

    st.markdown("---")
    st.markdown("### 🚀 Fitur Aplikasi")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""#### 🔍 Credit Scorer
- Input profil UMKM lengkap
- Prediksi real-time dari model AutoML
- Penjelasan SHAP per individu
- Rekomendasi AI jika ditolak""")
    with f2:
        st.markdown("""#### 🤖 AI Advisor
- Chatbot GPT-4o-mini
- Context-aware (data model + fairness)
- Multi-turn conversation
- Bahasa Indonesia""")
    with f3:
        st.markdown("""#### 📊 Analytics
- Distribusi fitur dataset
- Fairness per income/age group
- SHAP global summary
- Export-ready charts""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CREDIT SCORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Credit Scorer":
    st.markdown("## 🔍 Credit Risk Scorer")
    st.markdown("Masukkan profil UMKM atau pilih contoh untuk melihat prediksi risiko kredit real-time.")

    input_mode = st.radio(
        "Mode Input:", ["📋 Sample Preloaded", "✏️ Input Manual"],
        horizontal=True
    )

    if input_mode == "📋 Sample Preloaded":
        selected = st.selectbox("Pilih profil UMKM:", list(SAMPLE_PROFILES.keys()))
        profile  = SAMPLE_PROFILES[selected]
        st.info(f"📝 {profile['desc']}")
        input_data = {k: v for k, v in profile.items()
                      if k in FEATURE_COLS}

        with st.expander("📄 Detail Profil", expanded=True):
            cols = st.columns(4)
            items = [(k, v) for k, v in input_data.items()]
            for i, (k, v) in enumerate(items):
                with cols[i % 4]:
                    if isinstance(v, bool):
                        st.metric(k.replace('_',' ').title(), "Ya" if v else "Tidak")
                    elif isinstance(v, float) and v < 100:
                        st.metric(k.replace('_',' ').title(), f"{v:.3f}")
                    elif isinstance(v, (int, float)):
                        st.metric(k.replace('_',' ').title(), f"{v:,.0f}")
                    else:
                        st.metric(k.replace('_',' ').title(), str(v))

    else:
        st.markdown("#### Profil UMKM")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Data Peminjam**")
            age    = st.number_input("Usia (tahun)", 18, 70, 30)
            income = st.number_input("Pendapatan Tahunan (Rp)",
                                     10_000_000, 500_000_000, 60_000_000,
                                     step=5_000_000, format="%d")
            home   = st.selectbox("Status Tempat Tinggal",
                                  ["RENT", "OWN", "MORTGAGE", "OTHER"])
            emp    = st.slider("Lama Usaha (tahun)", 0, 20, 3)
            cb_def = st.checkbox("Pernah default sebelumnya?", value=False)
            cred   = st.slider("Lama Riwayat Kredit (tahun)", 0, 30, 3)

        with col2:
            st.markdown("**💰 Data Pinjaman**")
            intent    = st.selectbox("Tujuan Pinjaman",
                                     ["PERSONAL","VENTURE","EDUCATION",
                                      "MEDICAL","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
            grade     = st.selectbox("Grade Pinjaman",
                                     ["A","B","C","D","E","F","G"])
            loan_amnt = st.number_input("Jumlah Pinjaman (Rp)",
                                        1_000_000, 200_000_000, 20_000_000,
                                        step=1_000_000, format="%d")
            int_rate  = st.slider("Suku Bunga (%)", 5.0, 25.0, 12.0, 0.5)
            domain_r  = st.slider("Domain Risk Score (1=rendah, 5=tinggi)", 1, 5, 2)

        with col3:
            st.markdown("**📐 Derived Metrics (otomatis)**")
            monthly_int = loan_amnt * (int_rate / 100 / 12)
            monthly_inc = income / 12
            dsr = monthly_int / monthly_inc if monthly_inc > 0 else 0
            lti = loan_amnt / income if income > 0 else 0
            st.metric("Loan-to-Income Ratio", f"{lti:.3f}",
                      "✅ Aman" if lti < 0.4 else "⚠️ Tinggi")
            st.metric("Debt Service Ratio", f"{dsr:.3f}",
                      "✅ Aman" if dsr < 0.35 else "⚠️ Tinggi")
            st.metric("Monthly Interest (Rp)", f"{monthly_int:,.0f}")
            st.metric("Loan % Income", f"{lti:.1%}")

        input_data = derive_features(
            age, income, emp, loan_amnt, int_rate,
            home, intent, grade, cb_def, cred, domain_r
        )

    # ── Predict ──
    st.markdown("---")
    if st.button("🚀 Analisis Risiko Kredit", type="primary", use_container_width=True):

        input_df = pd.DataFrame([input_data])[FEATURE_COLS]

        with st.spinner("⏳ Menganalisis profil kredit..."):
            try:
                prob   = model.predict_proba(input_df)[0][1]
                pred   = int(prob >= 0.5)
                color, risk_label = risk_color(prob)
                shap_v = estimate_shap(input_df, model)
                success = True
            except Exception as e:
                st.error(f"❌ Error prediksi: {e}")
                success = False

        if success:
            st.markdown("### 📊 Hasil Analisis")
            r1, r2, r3 = st.columns([2, 1, 2])

            with r1:
                verdict     = "❌ DITOLAK" if pred == 1 else "✅ DISETUJUI"
                badge_class = "reject-badge" if pred == 1 else "approve-badge"
                st.markdown(f"""
                <div style='text-align:center;padding:1.5rem;
                            background:#f8f9fa;border-radius:12px;'>
                    <div class='{badge_class}'>{verdict}</div><br>
                    <h2 style='color:{color};margin:0'>{prob:.1%}</h2>
                    <p style='color:gray;margin:0'>Probabilitas Default</p>
                    <p><b>Risiko: <span style='color:{color}'>{risk_label}</span></b></p>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                fig_g, ax = plt.subplots(figsize=(2.5, 2.5))
                theta = np.linspace(0, np.pi, 100)
                ax.plot(np.cos(theta), np.sin(theta), 'lightgray', lw=8)
                ct = np.linspace(0, np.pi * prob, 100)
                ax.plot(np.cos(ct), np.sin(ct), color=color, lw=8)
                angle = np.pi * (1 - prob)
                ax.annotate('', xy=(np.cos(angle)*0.7, np.sin(angle)*0.7),
                            xytext=(0,0),
                            arrowprops=dict(arrowstyle='->', color='black', lw=2))
                ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.3, 1.2)
                ax.axis('off')
                ax.set_title(f'{prob:.0%}', fontsize=14, fontweight='bold', pad=0)
                plt.tight_layout()
                st.pyplot(fig_g, use_container_width=True)

            with r3:
                lti_v = input_data.get('loan_to_income_ratio', 0)
                dsr_v = input_data.get('debt_service_ratio', 0)
                ir_v  = input_data.get('loan_int_rate', 0)
                el_v  = input_data.get('person_emp_length', 0)
                st.markdown("**📐 Ringkasan Metrik:**")
                st.markdown(f"""
                | Metrik | Nilai | Status |
                |--------|-------|--------|
                | Loan-to-Income | {lti_v:.3f} | {'✅' if lti_v < 0.4 else '⚠️'} |
                | Debt Service Ratio | {dsr_v:.3f} | {'✅' if dsr_v < 0.35 else '⚠️'} |
                | Suku Bunga | {ir_v:.1f}% | {'✅' if ir_v < 14 else '⚠️'} |
                | Lama Usaha | {el_v:.0f} thn | {'✅' if el_v >= 3 else '⚠️'} |
                """)

            # SHAP
            st.markdown("### 🔍 Penjelasan SHAP — Faktor Penentu Keputusan")
            num_cols = [c for c in FEATURE_COLS
                        if input_data.get(c) is not None
                        and isinstance(input_data.get(c), (int, float))]
            shap_subset = shap_v[:len(num_cols)]

            fig_shap = plot_shap_bar(
                shap_subset, num_cols,
                f"Kontribusi Fitur — {'Penolakan' if pred==1 else 'Persetujuan'}"
            )
            st.pyplot(fig_shap, use_container_width=True)

            # AI Rekomendasi
            if genai_client and pred == 1:
                st.markdown("### 💡 Rekomendasi AI untuk Peminjam")
                top3 = sorted(
                    zip(num_cols, shap_subset),
                    key=lambda x: abs(x[1]), reverse=True
                )[:3]
                reasons = "\n".join([f"- {f}: {v:+.4f}" for f, v in top3])

                with st.spinner("🤖 AI menyiapkan rekomendasi..."):
                    cf = call_genai(
                        genai_client,
                        system="""Kamu adalah konsultan keuangan UMKM yang empatik.
Ketika UMKM ditolak, berikan:
1. Penjelasan singkat alasan penolakan (2 kalimat)
2. 3 langkah konkret dalam 3-6 bulan
3. Estimasi perbaikan yang diperlukan
4. Pesan motivasi singkat
Bahasa Indonesia, tone hangat & memberdayakan.""",
                        user=f"""UMKM ditolak pinjaman:
Probabilitas default: {prob:.1%}
Loan-to-Income: {lti_v:.3f} | Debt Service Ratio: {dsr_v:.3f}
Suku Bunga: {ir_v:.1f}% | Lama Usaha: {el_v:.0f} tahun

3 Faktor penolakan (SHAP):
{reasons}

Berikan rekomendasi counterfactual yang actionable.""",
                        max_tokens=500
                    )

                st.markdown(f"""
                <div class='warning-box'>
                {cf.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

            elif pred == 0:
                st.success("✅ Profil keuangan memenuhi kriteria kelayakan kredit.")
                st.info("💡 Gunakan tab **🤖 AI Advisor** untuk strategi memaksimalkan modal.")

            st.session_state['last_prediction'] = {
                'prob': prob, 'verdict': verdict,
                'input': input_data, 'pred': pred
            }

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Advisor":
    st.markdown("## 🤖 Smart Policy Advisor")
    st.markdown("Tanya apapun tentang kredit UMKM, inklusi keuangan, atau model ini.")

    if not genai_client:
        st.error("❌ Set GitHub Token di sidebar untuk mengaktifkan AI Advisor.")
        st.stop()

    pred_ctx = ""
    if 'last_prediction' in st.session_state:
        lp = st.session_state['last_prediction']
        pred_ctx = f"\nHasil scoring terakhir: {lp['verdict']} (P={lp['prob']:.1%})"

    SYSTEM = f"""Kamu adalah UMKM Credit AI Advisor — asisten strategis kredit UMKM Indonesia.

DATA MODEL:
- StackEnsemble AutoML (Azure ML) | AUC: 0.9507 | Acc: 93.6% | F1: 93.3%
- 32.000+ data pinjaman UMKM | Default rate: 22%

SHAP INSIGHTS:
- #1: loan_to_income_ratio | #2: loan_percent_income | #3: person_income
- Framework 4C: Capacity paling dominan

FAIRNESS:
- Income Gap Q1 vs Q4: ~30pp (refleksi realita, bukan bias)
- Age Gap: <5% — model age-blind (positif untuk inklusi)

KONTEKS INDONESIA:
- 64 juta UMKM | 60%+ belum bankable
- Program: KUR, BPUM, PNM Mekaar, KUR Mikro
- Target inklusi OJK: 90%
{pred_ctx}

Jawab dalam Bahasa Indonesia profesional. Berikan rekomendasi konkret dan actionable."""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.markdown("**💬 Pertanyaan populer:**")
    qs = [
        "Bagaimana UMKM meningkatkan skor kredit?",
        "Apa itu debt service ratio?",
        "Program kredit untuk UMKM Q1?",
        "Mengapa loan-to-income jadi faktor #1?",
        "Bagaimana AI ini bantu inklusi keuangan?",
        "Risiko deploy model tanpa fairness check?"
    ]
    qc = st.columns(3)
    for i, q in enumerate(qs):
        with qc[i % 3]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("🤖 ..."):
                    reply = call_genai(genai_client, SYSTEM, q,
                                       st.session_state.chat_history[:-1], 500)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.markdown("---")

    # Chat display
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center;color:gray;padding:2rem;'>
        🤖 Halo! Saya UMKM Credit AI Advisor.<br>
        Tanya saya tentang kredit, inklusi keuangan, atau analisis model ini.
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-user'>
            <b>👤 Kamu</b><br>{msg['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-bot'>
            <b>🤖 AI Advisor</b><br>{msg['content'].replace(chr(10),'<br>')}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5, 1])
        with ci:
            user_in = st.text_input(
                "Pertanyaan:",
                placeholder="Contoh: Bagaimana cara UMKM saya bisa lolos kredit?",
                label_visibility="collapsed"
            )
        with cb:
            send = st.form_submit_button("Kirim 📤", use_container_width=True)

    if send and user_in.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_in})
        with st.spinner("🤖 ..."):
            reply = call_genai(genai_client, SYSTEM, user_in,
                               st.session_state.chat_history[:-1], 500)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Hapus Riwayat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")

    df, data_status = load_dataset(azure_conn, azure_cont)
    st.info(data_status)

    if df is None:
        st.warning("Menggunakan data sintetik untuk demo.")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'loan_to_income_ratio': np.random.beta(2, 5, n),
            'debt_service_ratio':   np.random.beta(2, 5, n),
            'loan_int_rate':        np.random.uniform(6, 24, n),
            'person_income':        np.random.lognormal(17.5, 0.6, n),
            'person_emp_length':    np.random.randint(0, 15, n).astype(float),
            'loan_amnt':            np.random.randint(5_000_000, 100_000_000, n),
            'loan_status':          np.random.binomial(1, 0.22, n)
        })

    # Overview
    st.markdown("### 📈 Dataset Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Total Data", f"{len(df):,}")
    o2.metric("Default Rate", f"{df['loan_status'].mean():.1%}" if 'loan_status' in df.columns else "N/A")
    o3.metric("Fitur", str(df.shape[1]))
    if 'person_income' in df.columns:
        o4.metric("Median Income", f"Rp {df['person_income'].median()/1e6:.1f}jt")

    # Distribusi fitur
    st.markdown("### 📊 Distribusi Fitur Utama")
    plot_f = [f for f in ['loan_to_income_ratio','debt_service_ratio',
                           'loan_int_rate','person_income'] if f in df.columns]
    if plot_f:
        fig_d, axes = plt.subplots(1, len(plot_f), figsize=(14, 3.5))
        if len(plot_f) == 1:
            axes = [axes]
        for ax, feat in zip(axes, plot_f):
            if 'loan_status' in df.columns:
                for lbl, clr, nm in [(0,'#4575b4','Lancar'),(1,'#d73027','Default')]:
                    ax.hist(df[df['loan_status']==lbl][feat].dropna(),
                            bins=30, alpha=0.6, color=clr, label=nm, density=True)
                ax.legend(fontsize=8)
            else:
                ax.hist(df[feat].dropna(), bins=30, color='#2d6a9f', alpha=0.7)
            ax.set_title(feat.replace('_','\n'), fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.suptitle('Distribusi: Peminjam Lancar vs Default', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)

    # Fairness
    st.markdown("### ⚖️ Fairness Analysis — Income Quartile")
    if 'person_income' in df.columns and 'loan_status' in df.columns:
        df['iq'] = pd.qcut(df['person_income'], q=4,
                            labels=['Q1\nRendah','Q2\nBwh-Mngah',
                                    'Q3\nAts-Mngah','Q4\nTinggi'])
        fd = df.groupby('iq', observed=True)['loan_status'].mean().reset_index()
        fd.columns = ['Quartile','Default Rate']

        fig_f, axes = plt.subplots(1, 2, figsize=(12, 4))
        clrs = ['#d73027','#fc8d59','#fee090','#4575b4']
        bars = axes[0].bar(fd['Quartile'], fd['Default Rate'],
                           color=clrs, alpha=0.85, width=0.6)
        axes[0].set_ylabel('Default Rate')
        axes[0].set_title('Default Rate per Kelompok Pendapatan', fontsize=11)
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        for bar, val in zip(bars, fd['Default Rate']):
            axes[0].text(bar.get_x()+bar.get_width()/2.,
                         bar.get_height()+0.005,
                         f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        gap = fd.iloc[0]['Default Rate'] - fd.iloc[-1]['Default Rate']
        axes[1].barh(['Income Gap\n(Q1 vs Q4)'], [gap],
                     color='#d73027' if gap > 0.15 else '#fee090', height=0.4)
        axes[1].axvline(x=0.15, color='red', ls='--', lw=1.5, label='Threshold tinggi')
        axes[1].axvline(x=0.05, color='gray', ls='--', lw=1.5, label='Threshold rendah')
        axes[1].set_title(f'Disparity Gap: {gap:.1%}', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_f, use_container_width=True)

        st.markdown(f"""
        <div class='info-box'>
        <b>📌 Insight Inklusi Keuangan:</b><br>
        Gap default rate Q1 vs Q4 sebesar <b>{gap:.1%}</b> mencerminkan
        <b>ketimpangan ekonomi struktural</b>, bukan bias algoritmik.
        UMKM berpendapatan rendah membutuhkan program pendampingan
        (KUR, PNM Mekaar) di samping scoring model.
        </div>
        """, unsafe_allow_html=True)

    # SHAP Global
    st.markdown("### 🔍 Global Feature Importance")
    st.markdown("""
    | Rank | Fitur | Kategori 4C | Interpretasi |
    |------|-------|-------------|--------------|
    | 🥇 1 | `loan_to_income_ratio` | Capacity | Prediktor terkuat |
    | 🥈 2 | `loan_percent_income` | Capital | Eksposur pinjaman |
    | 🥉 3 | `person_income` | Capacity | Income absolut |
    | 4 | `loan_int_rate` | Conditions | Pricing risk |
    | 5 | `debt_service_ratio` | Capacity | Beban cicilan |

    > **Kesimpulan:** Kemampuan bayar **relatif** (rasio vs pendapatan)
    > jauh lebih prediktif dari nilai absolut.
    > Konsisten dengan literatur credit scoring global.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:gray;font-size:0.8rem;'>
🏦 UMKM Credit Risk AI | Datathon: Ekonomi Digital & Inklusi Keuangan<br>
<b>Azure ML AutoML (StackEnsemble)</b> + <b>GitHub Models GPT-4o-mini</b>
| Compatible dengan <b>Azure OpenAI</b> untuk production
</div>
""", unsafe_allow_html=True)
