import numpy as np
import os

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("CMC_BASE", r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2 copia")
SUJETO   = os.environ.get("CMC_SUJETO", "S6")
RUTA_CARPETA = os.path.join(BASE_DIR, SUJETO, "EMG")

FS           = 1000
SEG_INICIO = int(os.environ.get("CMC_SEG_INICIO", 1))
SEG_UTILES = int(os.environ.get("CMC_SEG_UTILES", 10))

# ── Cálculo de muestras ───────────────────────────────────────────────────────
M_INICIO  = FS * SEG_INICIO
M_FIN     = M_INICIO + FS * SEG_UTILES
M_MINIMO  = M_FIN

# ── Procesamiento ─────────────────────────────────────────────────────────────
archivos_emt = [f for f in os.listdir(RUTA_CARPETA) if f.endswith('.emt')]
print(f"Archivos encontrados: {len(archivos_emt)}\n")

for archivo in archivos_emt:
    ruta_emt    = os.path.join(RUTA_CARPETA, archivo)
    ruta_salida = os.path.join(RUTA_CARPETA, os.path.splitext(archivo)[0])

    header_info = {}
    with open(ruta_emt, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Frequency' in line:
            header_info['fs'] = float(line.split(':')[1].replace('Hz','').strip())
        if line.strip().startswith('Tracks'):
            header_info['tracks'] = int(line.split(':')[1].strip())
        if 'Frames' in line:
            header_info['frames'] = int(line.split(':')[1].strip())
        if line.strip().startswith('Frame\t') or line.strip().startswith('Frame '):
            for j in range(i+1, len(lines)):
                stripped = lines[j].strip()
                if stripped == '' or stripped.startswith('Start'):
                    continue
                try:
                    float(stripped.split()[0])
                    header_info['header_lines'] = j
                    break
                except ValueError:
                    continue
            break

    data   = np.loadtxt(ruta_emt, skiprows=header_info['header_lines'])
    matriz = data[:, 2:].T

    if matriz.shape[1] < M_MINIMO:
        print(f"[OMITIDO]  {archivo} — solo tiene {matriz.shape[1]} muestras (mínimo {M_MINIMO})")
        continue

    matriz = matriz[:, M_INICIO:M_FIN]

    np.save(ruta_salida + ".npy", matriz)
    print(f"[OK]  {archivo} → {os.path.splitext(archivo)[0]}.npy  {matriz.shape}")

print("\nProceso completado.")
