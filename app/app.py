import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
from PIL import Image, UnidentifiedImageError
from pathlib import Path

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "modelo" / "ufc_model_web.pkl"
CSV_PATH    = BASE_DIR / "peleadores_clean.csv"
FONDO_PATH  = BASE_DIR / "imagenes" / "aaaaufc_Octagono.jpg"

# -------------------------------
# Config — DEBE ir primero
# -------------------------------
st.set_page_config(page_title="UFC Predictor", layout="wide", page_icon="🥊")

# -------------------------------
# Fondo a Base64
# -------------------------------
def file_to_base64(path: Path) -> str:
    try:
        ext = path.suffix.lower().replace(".", "")
        if ext == "jpg": ext = "jpeg"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return ""

img_base64 = file_to_base64(FONDO_PATH)

# -------------------------------
# CSS estilo UFC
# -------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background-color: #0a0a0a;
        color: #ffffff;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #111111;
        border: 1px solid #D20A0A;
        border-radius: 8px;
        overflow: hidden;
    }}
    .main-title-container {{
        background-image: linear-gradient(rgba(0,0,0,0.1), rgba(0,0,0,0.1)),
                          url("{img_base64}");
        background-size: cover;
        background-position: center;
        padding: 40px 20px;
        border-radius: 10px;
        border: 2px solid #D20A0A;
        margin-bottom: 25px;
        text-align: center;
    }}
    .main-title-container h1 {{
        color: #ffffff !important;
        font-family: 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 8px;
        margin: 0;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.8);
        font-size: 50px !important;
    }}
    h2, h3 {{
        color: #ffffff !important;
        font-family: 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    label {{
        color: #D20A0A !important;
        font-weight: bold;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .stSelectbox > div > div {{
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #D20A0A !important;
        border-radius: 4px;
    }}
    .stButton > button:active {{
        transform: scale(0.96);
    }}
    .stButton > button {{
        background-color: #D20A0A !important;
        color: #ffffff !important;
        font-family: 'Arial Black', sans-serif;
        font-weight: bold;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 3px;
        border: none !important;
        border-radius: 4px;
        padding: 14px 75px !important;
        width: 100%;
        transition: background-color 0.2s;
        cursor: pointer;
        box-shadow: 0 0 15px rgba(210, 10, 10, 0.4);
    }}
    .stButton > button:hover {{
        background-color: #a00808 !important;
    }}
    hr {{ border: 1px solid #D20A0A; }}
    .result-card {{
        background: linear-gradient(to right, transparent, #D20A0A, transparent);
        border-left: 4px solid #D20A0A;
        border-radius: 8px;
        border: none;
        padding: 16px 20px;
        margin: 25px 0;
        height: 2px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        transition: 0.2s;
    }}
    .result-card:hover {{
        transform: translateY(-3px);
    }}
    .result-card .label {{
        color: #888888;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .result-card .value {{
        color: #ffffff;
        font-size: 28px;
        font-weight: bold;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 0 0 10px rgba(210, 10, 10, 0.5);
    }}
    .fighter-name {{
        font-family: 'Arial Black', sans-serif;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin-top: 10px;
    }}
    .fighter-name-red  {{     
        color: #D20A0A;
        text-shadow: 0 0 10px rgba(210, 10, 10, 0.6); 
    }}
    .fighter-name-blue {{     
        color: #4a9eff;
        text-shadow: 0 0 10px rgba(74, 158, 255, 0.6); 
    }}
    .vs-badge {{
        color: #ffffff;
        font-family: 'Arial Black', sans-serif;
        font-size: 70px;
        font-weight: 900;
        text-align: center;
        padding: 100px 0 0 0;
        text-shadow: 
            0 0 10px #D20A0A,
            0 0 20px #D20A0A;
        animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); }}
    }}
    .alerta-fire {{
        background: linear-gradient(90deg, #D20A0A, #ff4444);
        color: white;
        border-radius: 4px;
        padding: 10px 20px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        letter-spacing: 2px;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Cargar modelo y datos (cacheado)
# -------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(CSV_PATH)
    df['imagen_url'] = df['imagen_url'].astype(str).str.replace("\\", "/")
    return model, df

model, df = load_assets()

# -------------------------------
# Funciones de lógica
# -------------------------------
def get_signal(prob, upper=0.65, lower=0.35):
    if prob > upper:   return '🔴 ROJO FAVORITO'
    elif prob < lower: return '🔵 AZUL FAVORITO'
    else:              return 'IGUALADO'

def get_alert(prob, upper=0.75, lower=0.25):
    return '🔥 MUY SEGURO' if (prob > upper or prob < lower) else ''

def calcular_diferencias(r, b):
    """
    Calcula las 12 features que espera el modelo.
    r, b = filas del DataFrame peleadores_clean.csv
    """
    def winrate(row):
        total = row['wins'] + row['losses']
        return row['wins'] / total if total > 0 else 0.0

    return {
        'winrate_diff':  winrate(r)    - winrate(b),
        'striking_diff': r['splm']     - b['splm'],
        'str_acc_diff':  r['str_acc']  - b['str_acc'],
        'sapm_diff':     r['sapm']     - b['sapm'],
        'str_def_diff':  r['str_def']  - b['str_def'],
        'td_diff':       r['td_avg']   - b['td_avg'],
        'td_acc_diff':   r['td_acc']   - b['td_acc'],
        'td_def_diff':   r['td_def']   - b['td_def'],
        'sub_diff':      r['sub_avg']  - b['sub_avg'],
        'age_diff':      r['edad']     - b['edad'],
        'exp_diff':      (r['wins'] + r['losses']) - (b['wins'] + b['losses']),
        'losses_diff':   r['losses']   - b['losses'],
    }

def mostrar_foto(ruta: str):
    placeholder = "https://via.placeholder.com/300x350/1a1a1a/D20A0A?text=NO+IMG"
    try:
        if not isinstance(ruta, str) or ruta.strip() in ('', 'nan', 'None'):
            img = placeholder
        else:
            path = Path(ruta.strip())
            if not path.is_absolute():
                path = BASE_DIR / path
            path = path.resolve()
            img = Image.open(path) if path.exists() else placeholder
    except Exception:
        img = placeholder

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, use_container_width=True)

def grafico_probabilidad(prob, red_name, blue_name):
    fig, ax = plt.subplots(figsize=(5, 1.2))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.barh(0, prob,        color='#D20A0A', height=0.5, left=0)
    ax.barh(0, 1 - prob,    color='#4a9eff', height=0.5, left=prob)
    ax.axvline(0.5, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4)
    ax.text(prob / 2,          0, f'{prob*100:.0f}%', ha='center', va='center',
            color='white', fontsize=14, fontweight='bold', fontfamily='Arial Black')
    ax.text(prob + (1-prob)/2, 0, f'{(1-prob)*100:.0f}%', ha='center', va='center',
            color='white', fontsize=14, fontweight='bold', fontfamily='Arial Black')
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.4); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# ================================
# INTERFAZ
# ================================
st.markdown("""
    <div class="main-title-container">
        <h1>UFC FIGHT PREDICTOR</h1>
    </div>
""", unsafe_allow_html=True)

nombres = sorted(df['nombre'].unique())
col1, colVS, col2 = st.columns([5, 2, 5])

with col1:
    red_name = st.selectbox("🔴 ESQUINA ROJA", nombres, key="red")
    st.markdown(f'<p class="fighter-name fighter-name-red">{red_name}</p>', unsafe_allow_html=True)
    mostrar_foto(df[df['nombre'] == red_name]['imagen_url'].values[0])

with colVS:
    st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)

with col2:
    blue_name = st.selectbox("🔵 ESQUINA AZUL", nombres, key="blue")
    st.markdown(f'<p class="fighter-name fighter-name-blue">{blue_name}</p>', unsafe_allow_html=True)
    mostrar_foto(df[df['nombre'] == blue_name]['imagen_url'].values[0])

st.markdown("<hr>", unsafe_allow_html=True)

_, bcol, _ = st.columns([2, 3, 2])
with bcol:
    predecir = st.button("PREDECIR GANADOR")

# ================================
# RESULTADO
# ================================
if predecir:
    if red_name == blue_name:
        st.warning("⚠️ Selecciona peleadores distintos")
    else:
        r = df[df['nombre'] == red_name].iloc[0]
        b = df[df['nombre'] == blue_name].iloc[0]

        stats  = calcular_diferencias(r, b)
        X      = pd.DataFrame([stats])
        prob   = model.predict_proba(X)[0][1]
        signal = get_signal(prob)
        alert  = get_alert(prob)

        st.markdown("<br>", unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f'<div class="result-card"><div class="label">Probabilidad Rojo</div><div class="value">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
        with rc2:
            st.markdown(f'<div class="result-card"><div class="label">Probabilidad Azul</div><div class="value">{(1-prob)*100:.1f}%</div></div>', unsafe_allow_html=True)
        with rc3:
            st.markdown(f'<div class="result-card"><div class="label">Predicción</div><div class="value" style="font-size:18px;">{signal}</div></div>', unsafe_allow_html=True)

        if alert:
            st.markdown(f'<div class="alerta-fire">{alert}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        _, gc, _ = st.columns([1, 4, 1])
        with gc:
            fig = grafico_probabilidad(prob, red_name, blue_name)
            st.pyplot(fig)
        plt.close(fig)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>ESTADÍSTICAS COMPARATIVAS</h3>", unsafe_allow_html=True)

        def fmt_pct(v):
            return f"{v*100:.0f}%" if v <= 1 else f"{v:.0f}%"

        stats_df = pd.DataFrame({
            'Estadística': [
                'Edad', 'Altura', 'Peso', 'Victorias', 'Derrotas',
                'Golpes / min', 'Golpes recibidos / min',
                'Precisión golpeo', 'Defensa golpeo',
                'Derribos / 15min', 'Precisión derribos', 'Defensa derribos',
                'Sumisiones / 15min',
            ],
            f'🔴 {red_name}': [
                f"{int(r['edad'])} años", f"{int(r['altura_cm'])} cm", f"{int(r['peso_kg'])} kg",
                int(r['wins']), int(r['losses']),
                f"{r['splm']:.2f}", f"{r['sapm']:.2f}",
                fmt_pct(r['str_acc']), fmt_pct(r['str_def']),
                f"{r['td_avg']:.2f}", fmt_pct(r['td_acc']), fmt_pct(r['td_def']),
                f"{r['sub_avg']:.2f}",
            ],
            f'🔵 {blue_name}': [
                f"{int(b['edad'])} años", f"{int(b['altura_cm'])} cm", f"{int(b['peso_kg'])} kg",
                int(b['wins']), int(b['losses']),
                f"{b['splm']:.2f}", f"{b['sapm']:.2f}",
                fmt_pct(b['str_acc']), fmt_pct(b['str_def']),
                f"{b['td_avg']:.2f}", fmt_pct(b['td_acc']), fmt_pct(b['td_def']),
                f"{b['sub_avg']:.2f}",
            ],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)