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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding: 0 !important; max-width: 100% !important; }

h1  { font-family:'IBM Plex Mono',monospace; font-size:1.85rem;
      color:#0f2c5e; letter-spacing:-1px; margin-bottom:0; }
h3  { font-family:'IBM Plex Mono',monospace; color:#0f2c5e; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a1f44 0%, #1a56db 60%, #0a1f44 100%);
    padding: 2.8rem 4rem 2.2rem 4rem;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: white;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.97rem;
    color: rgba(255,255,255,0.78);
    margin-bottom: 0.5rem;
    line-height: 1.5;
}
.hero-data-note {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.60);
    margin-bottom: 0.8rem;
    font-style: italic;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 20px;
    padding: 0.25rem 1rem;
    font-size: 0.73rem;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1.2px;
}

/* ── Contenido ── */
.content-area {
    padding: 2rem 4rem 1rem 4rem;
}

.warn-box {
    background: #fff8e1;
    border-left: 4px solid #f59e0b;
    border-radius: 4px;
    padding: 0.8rem 1.1rem;
    font-size: 0.84rem;
    color: #78450a;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

.section-card {
    background: #f7f9fc;
    border-left: 4px solid #1a56db;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    min-height: 420px;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    font-weight: 600;
    color: #1a56db;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 0.9rem;
}

/* ── Resultados ── */
.result-card {
    background: linear-gradient(135deg, #0f2c5e 0%, #1a56db 100%);
    border-radius: 8px;
    padding: 1.5rem 1.8rem;
    color: white;
    margin-bottom: 0.9rem;
}
.result-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}
.result-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.3rem;
    font-weight: 600;
    line-height: 1;
}
.result-meta {
    font-size: 0.78rem;
    opacity: 0.72;
    margin-top: 0.35rem;
}

/* ── Botón ── */
.stButton>button {
    background: #0f2c5e;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.7rem 2.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.87rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    width: 100%;
    transition: background 0.2s;
}
.stButton>button:hover { background: #1a56db; }

/* ── Footer ── */
.footer-band {
    background: linear-gradient(135deg, #0a1f44 0%, #1a56db 100%);
    padding: 2.5rem 4rem;
    margin-top: 2.5rem;
    text-align: center;
}
.footer-names {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
.footer-inst {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.75);
    margin-bottom: 0.3rem;
}
.footer-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 20px;
    padding: 0.22rem 0.9rem;
    font-size: 0.71rem;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1.2px;
    margin-top: 0.7rem;
}
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

    # INDICE_BIENES
    bin_cols = []
    for col in asset_cols:
        if col in df.columns:
            df[f"{col}_BIN"] = df[col].map({"SI": 1, "NO": 0}).fillna(0).astype(int)
            bin_cols.append(f"{col}_BIN")
    df["INDICE_BIENES"] = df[bin_cols].sum(axis=1).astype(float)
    df.drop(columns=[c for c in asset_cols if c in df.columns], inplace=True, errors="ignore")
    df.drop(columns=bin_cols, inplace=True, errors="ignore")

    # Encoding
    for col, enc in encoders.items():
        if col not in df.columns:
            continue
        enc_tipo = type(enc).__name__
        try:
            if enc_tipo == "OrdinalEncoder":
                df[col] = enc.transform(df[[col]]).flatten()
            elif enc_tipo == "LabelEncoder":
                df[col] = enc.transform(df[col].astype(str))
            else:
                mapping = {cls: i for i, cls in enumerate(get_clases(enc))}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        except Exception:
            df[col] = -1

    # Feature Engineering
    def fe(nombre, fn):
        try:
            val = fn(df).replace([np.inf, -np.inf], np.nan)
            df[nombre] = val
        except Exception:
            df[nombre] = np.nan

    fe("ESTRATO_X_EDU_MADRE",
       lambda d: d["FAMI_ESTRATOVIVIENDA"] * d["FAMI_EDUCACIONMADRE"])
    fe("ESTRATO_X_EDU_PADRE",
       lambda d: d["FAMI_ESTRATOVIVIENDA"] * d["FAMI_EDUCACIONPADRE"])
    fe("DENSIDAD_HOGAR",
       lambda d: d["FAMI_PERSONASHOGAR"] / (d["FAMI_CUARTOSHOGAR"] + 1))
    fe("INTERNET_X_EDU_MADRE",
       lambda d: d["FAMI_TIENEINTERNET"] * d["FAMI_EDUCACIONMADRE"]
                 if "FAMI_TIENEINTERNET" in d.columns
                 else d["FAMI_EDUCACIONMADRE"] * 0)
    fe("LOG_PERSONAS",
       lambda d: np.log1p(d["FAMI_PERSONASHOGAR"].clip(lower=0)))
    fe("LOG_CUARTOS",
       lambda d: np.log1p(d["FAMI_CUARTOSHOGAR"].clip(lower=0)))

    # Variables contextuales no calculables en produccion
    for col in ["PROM_GLOBAL_MCPIO", "PROM_GLOBAL_COLEGIO"]:
        if col in indep_vars and col not in df.columns:
            df[col] = 0.5

    # ANIO: no es variable predictora (futuros estudiantes no la conocen).
    # Se imputa con la mediana del dataset de entrenamiento si aparece en indep_vars.
    if "ANIO" in indep_vars and "ANIO" not in df.columns:
        df["ANIO"] = 2018.0   # mediana aprox del rango 2014-2022

    # Garantizar todas las columnas esperadas
    for col in indep_vars:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordenar, imputar NaN y escalar
    df = df[indep_vars].copy()
    df = df.fillna(df.median())
    df = df.fillna(0)

    df_scaled = pd.DataFrame(scaler.transform(df), columns=indep_vars)
    return df_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 3. HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎓 Predicción de Puntajes ICFES Saber 11</div>
    <div class="hero-sub">
        Modelo de Machine Learning para estimación de resultados académicos
    </div>
    <div class="hero-data-note">
        📊 Basado en los resultados del Departamento del Quindío · Años 2014 – 2022
    </div>
    <div class="hero-badge">UPB &nbsp;·&nbsp; MEDELLÍN &nbsp;·&nbsp; APRENDIZAJE DE MÁQUINAS &nbsp;·&nbsp; 2026</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content-area">', unsafe_allow_html=True)

st.markdown(
    '<div class="warn-box">'
    '⚠️ <strong>Nota:</strong> Las predicciones son estimaciones estadísticas '
    'entrenadas con resultados históricos del Quindío (2014–2022). '
    'No constituyen garantía de resultados reales ni son extrapolables '
    'directamente a otros departamentos.'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORMULARIO
# ─────────────────────────────────────────────────────────────────────────────
col_est, col_cole, col_fam = st.columns(3, gap="large")

with col_est:
    st.markdown('<div class="section-card"><div class="section-title">👤 Estudiante</div>',
                unsafe_allow_html=True)
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
        help="1 = Ene-Mar  |  2 = Abr-Jun  |  3 = Jul-Sep  |  4 = Oct-Dic"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_cole:
    st.markdown('<div class="section-card"><div class="section-title">🏫 Colegio</div>',
                unsafe_allow_html=True)
    area = st.selectbox(
        "Área de ubicación",
        get_clases(encoders.get("COLE_AREA_UBICACION")) or ["URBANO", "RURAL"],
    )
    calendario = st.selectbox(
        "Calendario",
        get_clases(encoders.get("COLE_CALENDARIO")) or ["A", "B"],
    )
    jornada = st.selectbox(
        "Jornada",
        get_clases(encoders.get("COLE_JORNADA")) or ["MANANA", "TARDE", "COMPLETA"],
    )
    caracter = st.selectbox(
        "Carácter del colegio",
        get_clases(encoders.get("COLE_CARACTER")) or ["ACADEMICO", "TECNICO", "OTRO"],
    )
    bilingue_opts = get_clases(encoders.get("COLE_BILINGUE")) or ["S", "N"]
    bilingue = st.radio("Bilingüe", options=bilingue_opts, horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_fam:
    st.markdown('<div class="section-card"><div class="section-title">🏠 Entorno Familiar</div>',
                unsafe_allow_html=True)
    estrato_opts = get_clases(encoders.get("FAMI_ESTRATOVIVIENDA")) or \
                  ["Sin Estrato","Estrato 1","Estrato 2","Estrato 3",
                   "Estrato 4","Estrato 5","Estrato 6"]
    estrato   = st.selectbox("Estrato de vivienda", options=estrato_opts)
    edu_opts  = get_clases(encoders.get("FAMI_EDUCACIONMADRE")) or ["Ninguno", "Postgrado"]
    edu_madre = st.selectbox("Educación de la madre", options=edu_opts)
    edu_padre = st.selectbox(
        "Educación del padre",
        options=get_clases(encoders.get("FAMI_EDUCACIONPADRE")) or edu_opts,
    )
    personas = st.slider("Personas en el hogar", min_value=1, max_value=15, value=4)
    cuartos  = st.slider("Cuartos en el hogar",  min_value=1, max_value=15, value=3)
    st.markdown("**Bienes del hogar**")
    ca, cb = st.columns(2)
    with ca:
        tiene_auto = st.checkbox("Automóvil")
        tiene_comp = st.checkbox("Computador", value=True)
    with cb:
        tiene_inet = st.checkbox("Internet", value=True)
        tiene_lava = st.checkbox("Lavadora",  value=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.write("")
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predecir = st.button("⚡ Calcular predicciones", type="primary")

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
        # ANIO: no se pide al usuario, se imputa internamente con la mediana
    }

    with st.spinner("Calculando predicciones…"):
        try:
            X_scaled = preprocess_input(user_raw)
        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("### Resultados de la predicción")

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
        r2_txt = f"R² = {info['r2']:.3f}" if info["r2"] is not None else ""

        with cols_res[i % 3]:
            valor_str = f"{pred:.1f}" if not np.isnan(pred) else "—"
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">{icon} {label}</div>
                <div class="result-value">{valor_str}</div>
                <div class="result-meta">{info['nombre']} &nbsp;|&nbsp; {r2_txt}</div>
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
    st.dataframe(
        pd.DataFrame(tabla).set_index("Prueba"),
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FOOTER AZUL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-band">
    <div class="footer-names">
        Sebastián Muñoz &nbsp;·&nbsp; Sebastián Velasco &nbsp;·&nbsp; Iván Velasco
    </div>
    <div class="footer-inst">
        Universidad Pontificia Bolivariana &nbsp;·&nbsp; Medellín, Colombia
    </div>
    <div class="footer-inst" style="margin-top:0.2rem; font-size:0.78rem;">
        Modelo entrenado con resultados ICFES del Departamento del Quindío · 2014 – 2022
    </div>
    <div class="footer-tag">UPB &nbsp;·&nbsp; APRENDIZAJE DE MÁQUINAS &nbsp;·&nbsp; 2026</div>
</div>
""", unsafe_allow_html=True)
