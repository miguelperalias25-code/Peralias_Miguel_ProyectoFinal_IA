import pandas as pd
from datetime import datetime

df = pd.read_csv("raw_fighter_details.csv")

# -------------------------------
# LIMPIEZA
# -------------------------------

# Altura → cm
def altura_to_cm(h):
    if pd.isna(h):
        return None
    try:
        feet, inches = h.split("'")
        inches = inches.replace('"','').strip()
        return int(feet)*30.48 + int(inches)*2.54
    except:
        return None

df['altura_cm'] = df['Altura'].apply(altura_to_cm)

# Peso → kg
df['peso_kg'] = df['Peso'].str.replace(' libras.','', regex=False)
df['peso_kg'] = pd.to_numeric(df['peso_kg'], errors='coerce') * 0.453592

# Alcance → cm
df['alcance_cm'] = pd.to_numeric(df['Alcanzar'], errors='coerce') * 2.54

# Porcentajes → decimal
for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    df[col] = df[col].str.rstrip('%')
    df[col] = pd.to_numeric(df[col], errors='coerce') / 100

# Edad
df['Fecha_nac'] = pd.to_datetime(df['Fecha de nacimiento'], errors='coerce', dayfirst=True)
df['edad'] = (datetime.now() - df['Fecha_nac']).dt.days / 365

# Renombrar columnas importantes
df = df.rename(columns={
    'nombre_del_luchador': 'nombre',
    'SLpM': 'splm',
    'TD_Avg': 'td_avg',
    'Subpromedio': 'sub_avg'
})

# Eliminar filas incompletas
df = df[['nombre','edad','altura_cm','peso_kg','alcance_cm','splm','td_avg','sub_avg']].dropna()

# Guardar limpio
df.to_csv("peleadores_clean.csv", index=False)

print("✅ Datos limpiados y guardados en peleadores_clean.csv")