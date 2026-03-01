import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Predicción ICFES Saber 11",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #f0f4f8; }

/* ── HERO ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(120deg, #071a3e 0%, #0f3580 45%, #1a56db 100%);
    padding: 2.4rem 3.5rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero-title span { color: #60a5fa; }
.hero-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0.25rem;
}
.hero-data {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    font-style: italic;
    margin-bottom: 1rem;
}
.hero-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px;
    padding: 0.22rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.8px;
}

/* ── CONTENIDO ─────────────────────────────────────────────────── */
.main-content { padding: 1.6rem 3.5rem 0 3.5rem; background: #f0f4f8; }

.warn {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.81rem;
    color: #78450a;
    margin-bottom: 1.4rem;
    line-height: 1.55;
}

/* ── TARJETAS DE SECCION ──────────────────────────────────────── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #e2e8f0;
}
.sec-icon {
    width: 34px; height: 34px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.sec-icon-blue  { background: #dbeafe; }
.sec-icon-green { background: #dcfce7; }
.sec-icon-amber { background: #fef3c7; }
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #1e40af;
}

/* ── RESULTADOS ──────────────────────────────────────────────── */
.results-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}
.rcard {
    background: linear-gradient(135deg, #0f2c5e 0%, #1a56db 100%);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.rcard::after {
    content: '';
    position: absolute;
    top: -20px; right: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.rcard:hover { transform: translateY(-2px); }
.rcard-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.3rem;
}
.rcard-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.rcard-meta { font-size: 0.73rem; opacity: 0.65; }

/* ── TABLA ──────────────────────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── BOTÓN ──────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0f2c5e, #1a56db) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── FOOTER ─────────────────────────────────────────────────── */
.footer {
    background: linear-gradient(120deg, #071a3e 0%, #0f3580 50%, #1a56db 100%);
    padding: 2.2rem 3.5rem;
    margin-top: 2.5rem;
    text-align: center;
}
.footer-names {
    font-weight: 700;
    font-size: 1rem;
    color: #fff;
    margin-bottom: 0.35rem;
    letter-spacing: 0.3px;
}
.footer-inst {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0.2rem;
}
.footer-data {
    font-size: 0.74rem;
    color: rgba(255,255,255,0.45);
    font-style: italic;
    margin-bottom: 0.8rem;
}
.footer-pills { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }

/* ── Eliminar padding por defecto de Streamlit ── */
header[data-testid="stHeader"] { display: none !important; }
#root > div:first-child { margin: 0 !important; }
.appview-container { padding: 0 !important; }
.appview-container > section { padding: 0 !important; }

/* Block container: márgenes laterales aquí, no en HTML */
.block-container {
    padding: 0 0 2rem 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* Wrapper para contenido con márgenes consistentes */
.stApp > div > div > div > div > div[data-testid="stVerticalBlock"] {
    padding: 0 3.5rem !important;
}

/* Excepciones: hero y footer ocupan ancho completo */
.stApp > div > div > div > div > div[data-testid="stVerticalBlock"] > div:first-child,
.stApp > div > div > div > div > div[data-testid="stVerticalBlock"] > div:last-child {
    padding: 0 !important;
    margin: 0 !important;
}

/* Columnas sin padding extra */
[data-testid="column"] { padding: 0 0.4rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* Reducir gap vertical entre elementos */
div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE ARTEFACTOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    path = "pipeline_artefacts.joblib"
    if not os.path.exists(path):
        st.error(f"No se encontró '{path}'. Coloca todos los .joblib en la misma carpeta que app.py.")
        st.stop()

    arts       = joblib.load(path)
    scaler     = arts["scaler"]
    encoders   = arts["encoders"]
    indep_vars = arts["independent_vars"]
    score_cols = arts["score_cols"]
    asset_cols = arts["asset_cols"]

    loaded_models = {}
    for target in score_cols:
        fname = f"icfes_mejor_{target.lower()}.joblib"
        if not os.path.exists(fname):
            st.error(f"Modelo no encontrado: {fname}")
            st.stop()
        md = joblib.load(fname)
        loaded_models[target] = {
            "modelo" : md["modelo_fit"],
            "vars"   : md["vars_sig"],
            "nombre" : md.get("model_name", target),
            "r2"     : md.get("metricas", {}).get("r2_test"),
        }
    return scaler, encoders, indep_vars, score_cols, asset_cols, loaded_models


scaler, encoders, indep_vars, score_cols, asset_cols, loaded_models = load_artefacts()


def get_clases(enc):
    if enc is None:
        return []
    if hasattr(enc, "categories_"):
        return list(enc.categories_[0])
    if hasattr(enc, "classes_"):
        return list(enc.classes_)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    bin_cols = []
    for col in asset_cols:
        if col in df.columns:
            df[f"{col}_BIN"] = df[col].map({"SI": 1, "NO": 0}).fillna(0).astype(int)
            bin_cols.append(f"{col}_BIN")
    df["INDICE_BIENES"] = df[bin_cols].sum(axis=1).astype(float)
    df.drop(columns=[c for c in asset_cols if c in df.columns], inplace=True, errors="ignore")
    df.drop(columns=bin_cols, inplace=True, errors="ignore")

    for col, enc in encoders.items():
        if col not in df.columns:
            continue
        try:
            if type(enc).__name__ == "OrdinalEncoder":
                df[col] = enc.transform(df[[col]]).flatten()
            elif type(enc).__name__ == "LabelEncoder":
                df[col] = enc.transform(df[col].astype(str))
            else:
                mapping = {cls: i for i, cls in enumerate(get_clases(enc))}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        except Exception:
            df[col] = -1

    def fe(nombre, fn):
        try:
            df[nombre] = fn(df).replace([np.inf, -np.inf], np.nan)
        except Exception:
            df[nombre] = np.nan

    fe("ESTRATO_X_EDU_MADRE",  lambda d: d["FAMI_ESTRATOVIVIENDA"] * d["FAMI_EDUCACIONMADRE"])
    fe("ESTRATO_X_EDU_PADRE",  lambda d: d["FAMI_ESTRATOVIVIENDA"] * d["FAMI_EDUCACIONPADRE"])
    fe("DENSIDAD_HOGAR",       lambda d: d["FAMI_PERSONASHOGAR"] / (d["FAMI_CUARTOSHOGAR"] + 1))
    fe("INTERNET_X_EDU_MADRE", lambda d: d["FAMI_EDUCACIONMADRE"] * 0)
    fe("LOG_PERSONAS",         lambda d: np.log1p(d["FAMI_PERSONASHOGAR"].clip(lower=0)))
    fe("LOG_CUARTOS",          lambda d: np.log1p(d["FAMI_CUARTOSHOGAR"].clip(lower=0)))

    for col in ["PROM_GLOBAL_MCPIO", "PROM_GLOBAL_COLEGIO"]:
        if col in indep_vars and col not in df.columns:
            df[col] = 0.5

    if "ANIO" in indep_vars and "ANIO" not in df.columns:
        df["ANIO"] = 2018.0

    for col in indep_vars:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[indep_vars].copy()
    df = df.fillna(df.median())
    df = df.fillna(0)

    return pd.DataFrame(scaler.transform(df), columns=indep_vars)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Sistema de predicción académica</div>
    <div class="hero-title">Predicción de Puntajes<br><span>ICFES Saber 11</span></div>
    <div class="hero-sub">Modelo de Machine Learning para estimación de resultados académicos</div>
    <div class="hero-data">📊 Entrenado con resultados del Departamento del Quindío · 2014 – 2022</div>
    <div class="hero-pills">
        <span class="pill">UPB</span>
        <span class="pill">Medellín</span>
        <span class="pill">Aprendizaje de Máquinas</span>
        <span class="pill">2026</span>
    </div>
</div>
<div class="main-content">
""", unsafe_allow_html=True)

st.markdown("""
<div class="warn">
⚠️ <strong>Nota:</strong> Las predicciones son estimaciones estadísticas basadas en resultados
históricos del Quindío (2014–2022). No constituyen garantía de resultados reales
ni son directamente extrapolables a otros departamentos.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORMULARIO — 3 columnas compactas
# ─────────────────────────────────────────────────────────────────────────────
col_est, col_cole, col_fam = st.columns(3, gap="medium")

# ── Estudiante ────────────────────────────────────────────────────────────────
with col_est:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-icon sec-icon-blue">👤</div>
        <span class="sec-label">Estudiante</span>
    </div>
    """, unsafe_allow_html=True)

    genero_opts = get_clases(encoders.get("ESTU_GENERO")) or ["F", "M"]
    genero = st.radio(
        "Género",
        options=genero_opts,
        format_func=lambda x: "Femenino" if x == "F" else "Masculino",
        horizontal=True,
    )
    edad = st.slider("Edad (años)", min_value=12, max_value=30, value=17)
    trimestre = st.radio(
        "Trimestre de presentación",
        options=[1, 2, 3, 4],
        horizontal=True,
        help="1=Ene-Mar · 2=Abr-Jun · 3=Jul-Sep · 4=Oct-Dic",
    )

# ── Colegio ───────────────────────────────────────────────────────────────────
with col_cole:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-icon sec-icon-green">🏫</div>
        <span class="sec-label">Colegio</span>
    </div>
    """, unsafe_allow_html=True)

    area = st.selectbox("Área",
        get_clases(encoders.get("COLE_AREA_UBICACION")) or ["URBANO", "RURAL"])
    calendario = st.selectbox("Calendario",
        get_clases(encoders.get("COLE_CALENDARIO")) or ["A", "B"])
    jornada = st.selectbox("Jornada",
        get_clases(encoders.get("COLE_JORNADA")) or ["MANANA", "TARDE", "COMPLETA"])
    caracter = st.selectbox("Carácter",
        get_clases(encoders.get("COLE_CARACTER")) or ["ACADEMICO", "TECNICO", "OTRO"])
    bilingue_opts = get_clases(encoders.get("COLE_BILINGUE")) or ["S", "N"]
    bilingue = st.radio("Bilingüe", options=bilingue_opts, horizontal=True)

# ── Familia ───────────────────────────────────────────────────────────────────
with col_fam:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-icon sec-icon-amber">🏠</div>
        <span class="sec-label">Entorno Familiar</span>
    </div>
    """, unsafe_allow_html=True)

    estrato_opts = get_clases(encoders.get("FAMI_ESTRATOVIVIENDA")) or \
        ["Sin Estrato","Estrato 1","Estrato 2","Estrato 3","Estrato 4","Estrato 5","Estrato 6"]
    estrato   = st.selectbox("Estrato", options=estrato_opts)
    edu_opts  = get_clases(encoders.get("FAMI_EDUCACIONMADRE")) or ["Ninguno", "Postgrado"]
    edu_madre = st.selectbox("Educación madre", options=edu_opts)
    edu_padre = st.selectbox("Educación padre",
        options=get_clases(encoders.get("FAMI_EDUCACIONPADRE")) or edu_opts)
    personas = st.slider("Personas en el hogar", 1, 15, 4)
    cuartos  = st.slider("Cuartos en el hogar",  1, 15, 3)

    st.markdown("**Bienes del hogar**")
    ca, cb = st.columns(2)
    with ca:
        tiene_auto = st.checkbox("🚗 Automóvil")
        tiene_comp = st.checkbox("💻 Computador", value=True)
    with cb:
        tiene_inet = st.checkbox("🌐 Internet", value=True)
        tiene_lava = st.checkbox("🫧 Lavadora",  value=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOTÓN
# ─────────────────────────────────────────────────────────────────────────────
st.write("")
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predecir = st.button("⚡ Calcular predicciones", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
if predecir:
    user_raw = {
        "ESTU_GENERO"          : genero,
        "EDAD"                 : float(edad),
        "TRIMESTRE"            : float(trimestre),
        "COLE_AREA_UBICACION"  : area,
        "COLE_CALENDARIO"      : calendario,
        "COLE_JORNADA"         : jornada,
        "COLE_CARACTER"        : caracter,
        "COLE_BILINGUE"        : bilingue,
        "FAMI_ESTRATOVIVIENDA" : estrato,
        "FAMI_EDUCACIONMADRE"  : edu_madre,
        "FAMI_EDUCACIONPADRE"  : edu_padre,
        "FAMI_PERSONASHOGAR"   : float(personas),
        "FAMI_CUARTOSHOGAR"    : float(cuartos),
        "FAMI_TIENEAUTOMOVIL"  : "SI" if tiene_auto else "NO",
        "FAMI_TIENECOMPUTADOR" : "SI" if tiene_comp else "NO",
        "FAMI_TIENEINTERNET"   : "SI" if tiene_inet else "NO",
        "FAMI_TIENELAVADORA"   : "SI" if tiene_lava else "NO",
    }

    with st.spinner("Calculando predicciones…"):
        try:
            X_scaled = preprocess_input(user_raw)
        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("### 📋 Resultados de la predicción")

    nombres = {
        "PUNT_GLOBAL"              : ("Puntaje Global",        "📊"),
        "PUNT_MATEMATICAS"         : ("Matemáticas",           "🔢"),
        "PUNT_INGLES"              : ("Inglés",                "🇬🇧"),
        "PUNT_LECTURA_CRITICA"     : ("Lectura Crítica",       "📖"),
        "PUNT_C_NATURALES"         : ("Ciencias Naturales",    "🔬"),
        "PUNT_SOCIALES_CIUDADANAS" : ("Sociales y Ciudadanas", "🏛️"),
    }

    cols_res = st.columns(3, gap="medium")
    tabla    = []

    for i, (target, info) in enumerate(loaded_models.items()):
        modelo   = info["modelo"]
        vars_sig = info["vars"]
        cols_ok  = [v for v in vars_sig if v in X_scaled.columns]
        X_target = X_scaled[cols_ok]

        try:
            pred = float(modelo.predict(X_target)[0])
            pred = max(0.0, pred)
        except Exception as e:
            pred = np.nan
            st.warning(f"Error prediciendo {target}: {e}")

        label, icon = nombres.get(target, (target, "📌"))
        r2_txt = f"R² {info['r2']:.3f}" if info["r2"] is not None else ""
        valor_str = f"{pred:.1f}" if not np.isnan(pred) else "—"

        with cols_res[i % 3]:
            st.markdown(f"""
            <div class="rcard">
                <div class="rcard-label">{icon} {label}</div>
                <div class="rcard-score">{valor_str}</div>
                <div class="rcard-meta">{info['nombre']} · {r2_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        tabla.append({
            "Prueba"           : label,
            "Puntaje predicho" : valor_str,
            "Modelo"           : info["nombre"],
            "R² test"          : f"{info['r2']:.4f}" if info["r2"] else "—",
            "Variables usadas" : len(vars_sig),
        })

    st.markdown("")
    st.dataframe(pd.DataFrame(tabla).set_index("Prueba"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-names">
        Sebastián Muñoz &nbsp;·&nbsp; Sebastián Velasco &nbsp;·&nbsp; Iván Velasco
    </div>
    <div class="footer-inst">Universidad Pontificia Bolivariana · Medellín, Colombia</div>
    <div class="footer-data">
        Modelo entrenado con resultados ICFES del Departamento del Quindío · 2014 – 2022
    </div>
    <div class="footer-pills">
        <span class="pill">UPB</span>
        <span class="pill">Aprendizaje de Máquinas</span>
        <span class="pill">2026</span>
    </div>
</div>
""", unsafe_allow_html=True)
