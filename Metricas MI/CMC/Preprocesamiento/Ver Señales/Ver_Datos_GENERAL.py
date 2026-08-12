import re
import runpy
from pathlib import Path
import matplotlib.pyplot as plt

plt.close('all')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# BASE_DIR   = Path(r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2")
BASE_DIR   = Path(r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2")
SUJETO     = "S17"                                              # S17, S18
PRUEBA     = "OK"                                               # OV, OK, CV, CK
MOVIMIENTO = "ME"                                               # ME, MI, RE

PLOT_EEG_SCRIPT       = "Ver_Datos_EEG.py"
PLOT_EMG_DINAM_SCRIPT = "Ver_Datos_EMG.py"
PLOT_JUNTOS_SCRIPT      = "Ver_Datos_JUNTOS.py"

PATRON = re.compile(r"^(S\d+)_(\d{6})_(EEG|EMG|DINAM)_(OV|OK|CV|CK)_(\d+)_(ME|MI|RE)\.npy$")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def validar_inputs(prueba: str, mov: str):
    if prueba not in {"OV", "OK", "CV", "CK"}:
        raise ValueError(f"PRUEBA inválida: {prueba}. Usa OV, OK, CV o CK.")
    if mov not in {"ME", "MI", "RE"}:
        raise ValueError(f"MOVIMIENTO inválido: {mov}. Usa ME, MI o RE.")

def listar_eeg_matches(eeg_dir: Path, sujeto: str, prueba: str, mov: str):
    if not eeg_dir.exists():
        return []
    out = []
    for f in eeg_dir.glob("*.npy"):
        m = PATRON.match(f.name)
        if not m:
            continue
        sj, fecha, sig, pr, npr, mv = m.groups()
        if sj == sujeto and sig == "EEG" and pr == prueba and mv == mov:
            out.append((f, int(fecha), int(npr)))
    return out

def elegir_eeg_estricto(matches):
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        nombres = [m[0].name for m in matches]
        raise RuntimeError(
            "Ambigüedad: se encontró más de un EEG candidato.\n"
            "Candidatos:\n- " + "\n- ".join(nombres)
        )
    return matches[0][0]

def resolver_paths(base_dir: Path, sujeto: str, prueba: str, mov: str):
    sujeto_dir = base_dir / sujeto
    eeg_dir = sujeto_dir / "EEG"

    eeg_matches = listar_eeg_matches(eeg_dir, sujeto, prueba, mov)
    eeg_path = elegir_eeg_estricto(eeg_matches)

    if eeg_path is None:
        raise FileNotFoundError(
            f"No se encontró EEG para SUJETO={sujeto}, PRUEBA={prueba}, MOVIMIENTO={mov} en: {eeg_dir}"
        )

    m = PATRON.match(eeg_path.name)
    sj, fecha, sig, pr, npr, mv = m.groups()
    fecha_int = int(fecha)
    npr_int   = int(npr)

    emg_path = None
    dinam_path = None

    if mov == "ME":
        emg_path_cand   = sujeto_dir / "EMG"   / f"{sujeto}_{fecha_int:06d}_EMG_{prueba}_{npr_int}_ME.npy"
        dinam_path_cand = sujeto_dir / "DINAM" / f"{sujeto}_{fecha_int:06d}_DINAM_{prueba}_{npr_int}_ME.npy"

        if emg_path_cand.exists():
            emg_path = emg_path_cand
        else:
            print(f"[AVISO] No existe EMG esperado: {emg_path_cand}")

        if dinam_path_cand.exists():
            dinam_path = dinam_path_cand
        else:
            print(f"[AVISO] No existe DINAM esperado: {dinam_path_cand}")

    return eeg_path, emg_path, dinam_path

# ─────────────────────────────────────────────────────────────────────────────
# EJECUTABLE
# ─────────────────────────────────────────────────────────────────────────────

validar_inputs(PRUEBA, MOVIMIENTO)

eeg_path, emg_path, dinam_path = resolver_paths(BASE_DIR, SUJETO, PRUEBA, MOVIMIENTO)

print("\nEEG  :", eeg_path)

# EEG
runpy.run_path(
    PLOT_EEG_SCRIPT,
    run_name="__main__",
    init_globals={"RUTA_NPY": str(eeg_path)}
)

# EMG / DIM / JUNTOS
if MOVIMIENTO == "ME":
    if emg_path and dinam_path:
        print("EMG  :", emg_path)
        print("DINAM:", dinam_path)

        # EMG / DIM
        runpy.run_path(
            PLOT_EMG_DINAM_SCRIPT,
            run_name="__main__",
            init_globals={"RUTA_EMG": str(emg_path), "RUTA_DINAM": str(dinam_path)}
        )
        
        print("EEG  :", eeg_path)
        print("EMG  :", emg_path)
        print("DINAM:", dinam_path)
        
        # JUNTOS
        runpy.run_path(
            PLOT_JUNTOS_SCRIPT,
            run_name="__main__",
            init_globals={"RUTA_EEG": str(eeg_path), "RUTA_EMG": str(emg_path), "RUTA_DINAM": str(dinam_path)}
        )
        
    else:
        print("[INFO] MOVIMIENTO='ME' pero faltó EMG o DINAM → se graficó solo EEG.")
else:
    print("[INFO] MOVIMIENTO != 'ME' → solo se grafica EEG (MI/RE no tienen EMG/DINAM).")
