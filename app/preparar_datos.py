# ---------------------------------------
# preparar_datos.py
# Fuente: ufc_fighters_final.csv
# Genera: peleadores_clean.csv
# Portable: rutas relativas al script
# ---------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

CSV_RAW       = BASE_DIR / "ufc_fighters_final.csv"
CSV_CLEAN     = BASE_DIR / "peleadores_clean.csv"
IMAGEN_FOLDER = BASE_DIR / "imagenes"

print(f"📁 Base: {BASE_DIR}")

# ================================================
# 1. CARGAR
# ================================================
df = pd.read_csv(CSV_RAW)
print(f"Peleadores cargados: {len(df)}")

# ================================================
# 2. TRANSFORMAR COLUMNAS
# ================================================
def pct(s):
    try:    return float(str(s).rstrip('%')) / 100
    except: return np.nan

def altura_cm(h):
    try:
        f, i = h.split("'")
        return float(f) * 30.48 + float(i.replace('"', '').strip()) * 2.54
    except: return np.nan

def reach_cm(r):
    try:    return float(str(r).replace('"', '').strip()) * 2.54
    except: return np.nan

def peso_kg(w):
    try:    return float(str(w).replace(' lbs.', '').strip()) * 0.453592
    except: return np.nan

df['altura_cm']  = df['Height'].apply(altura_cm)
df['peso_kg']    = df['Weight'].apply(peso_kg)
df['alcance_cm'] = df['Reach'].apply(reach_cm)
df['str_acc']    = df['Str_Acc'].apply(pct)
df['str_def']    = df['Str_Def'].apply(pct)
df['td_acc']     = df['TD_Acc'].apply(pct)
df['td_def']     = df['TD_Def'].apply(pct)
df['DOB']        = pd.to_datetime(df['DOB'], errors='coerce')
df['edad']       = (datetime.now() - df['DOB']).dt.days / 365

df = df.rename(columns={
    'Fighter_Name': 'nombre',
    'Wins':         'wins',
    'Losses':       'losses',
    'SLpM':         'splm',
    'SApM':         'sapm',
    'TD_Avg':       'td_avg',
    'Sub_Avg':      'sub_avg',
})

# ================================================
# 3. SELECCIONAR Y LIMPIAR COLUMNAS FINALES
# ================================================
COLS = ['nombre', 'edad', 'altura_cm', 'peso_kg', 'alcance_cm',
        'splm', 'sapm', 'str_acc', 'str_def',
        'td_avg', 'td_acc', 'td_def', 'sub_avg',
        'wins', 'losses']

df = df[COLS].copy()
df.fillna({
    'edad':      0, 'altura_cm': 0, 'peso_kg':   0,
    'alcance_cm':0, 'splm':      0, 'sapm':       0,
    'str_acc':   0, 'str_def':   0, 'td_avg':     0,
    'td_acc':    0, 'td_def':    0, 'sub_avg':    0,
    'wins':      0, 'losses':    0,
}, inplace=True)

# ================================================
# 4. BUSCAR IMAGEN (ruta relativa portable)
# ================================================
def buscar_imagen(nombre: str) -> str:
    nombre_clean = nombre.strip()
    for ext in ['png', 'jpg', 'jpeg', 'avif']:
        ruta = IMAGEN_FOLDER / f"{nombre_clean}.{ext}"
        if ruta.exists():
            return str(ruta.relative_to(BASE_DIR)).replace("\\", "/")
    return ""

df['imagen_url'] = df['nombre'].apply(buscar_imagen)

# ================================================
# 5. GUARDAR
# ================================================
df.to_csv(CSV_CLEAN, index=False)
print(f"✅ Guardado en: {CSV_CLEAN}")

con_imagen = (df['imagen_url'] != '').sum()
sin_imagen = (df['imagen_url'] == '').sum()
print(f"   Con imagen : {con_imagen}")
print(f"   Sin imagen : {sin_imagen}")
if con_imagen:
    print("\nEjemplos:")
    print(df[df['imagen_url'] != ''][['nombre','imagen_url']].head(5).to_string(index=False))