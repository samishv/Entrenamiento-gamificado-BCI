import os
import runpy
from pathlib import Path

# ====== CONFIGURAME ESTA ======

# BASE_DIR    = Path(r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2")
BASE_DIR    = Path(r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2 copia")
SUJETO      = "S6"
SEG_INICIO  = 1  
SEG_UTILES  = 10    
EJECUTAR    = ["EEG", "EMG", "DINAM"]

SCRIPTS = {
    "EEG":  "Cortar_Datos_EEG.py",
    "EMG":  "Cortar_Datos_EMG.py",
    "DINAM":"Cortar_Datos_DINAM.py",
}

def validar_estructura(base: Path, sujeto: str):
    sujeto_dir = base / sujeto
    if not sujeto_dir.exists():
        disponibles = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("S")]
        raise FileNotFoundError(
            f"No existe {sujeto_dir}\nSujeto(s) disponibles: {disponibles}"
        )

for modo in EJECUTAR:
    validar_estructura(BASE_DIR, SUJETO)

    os.environ["CMC_BASE"]       = str(BASE_DIR)
    os.environ["CMC_SUJETO"]     = SUJETO
    os.environ["CMC_SEG_INICIO"] = str(SEG_INICIO)
    os.environ["CMC_SEG_UTILES"] = str(SEG_UTILES)

    script = SCRIPTS[modo]
    print(f"\n=== Ejecutando {modo} para {SUJETO} ===")
    runpy.run_path(script, run_name="__main__")

print("\nCortes Finalizados")
