import pandas as pd
import numpy as np
import os

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("CMC_BASE", r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2 copia")
SUJETO   = os.environ.get("CMC_SUJETO", "S6")
RUTA_CARPETA = os.path.join(BASE_DIR, SUJETO, "EEG")

COLUMNAS     = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
FS           = 250
SEG_INICIO = int(os.environ.get("CMC_SEG_INICIO", 1))
SEG_UTILES = int(os.environ.get("CMC_SEG_UTILES", 10))

# ── Cálculo de muestras ───────────────────────────────────────────────────────
M_INICIO  = FS * SEG_INICIO
M_FIN     = M_INICIO + FS * SEG_UTILES
M_MINIMO  = M_FIN

# ── Procesamiento ─────────────────────────────────────────────────────────────
nombres_col = ['Timestamp', 'Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'NC', 'C4',
               'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17']

archivos_csv = [f for f in os.listdir(RUTA_CARPETA) if f.endswith('.csv')]
print(f"Archivos encontrados: {len(archivos_csv)}\n")

for archivo in archivos_csv:
    ruta_csv    = os.path.join(RUTA_CARPETA, archivo)
    ruta_salida = os.path.join(RUTA_CARPETA, os.path.splitext(archivo)[0])

    datos  = pd.read_csv(ruta_csv, header=0, names=nombres_col)
    matriz = datos[COLUMNAS].astype(float).to_numpy().T

    if matriz.shape[1] < M_MINIMO:
        print(f"[OMITIDO]  {archivo} — solo tiene {matriz.shape[1]} muestras (mínimo {M_MINIMO})")
        continue

    matriz = matriz[:, M_INICIO:M_FIN]

    np.save(ruta_salida + ".npy", matriz)
    print(f"[OK]  {archivo} → {os.path.splitext(archivo)[0]}.npy  {matriz.shape}")

print("\nProceso completado.")
