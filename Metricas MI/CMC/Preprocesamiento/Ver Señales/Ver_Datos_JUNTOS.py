import os
import re
import numpy as np
import matplotlib.pyplot as plt

from Procesamiento_Funciones import (
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    cargar_npy_emg,
    cargar_npy_dinam,
    resamplear_por_promedio,
    resamplear_profesional,
    aplicar_wavelet_dwt_multicanal,
    rectificar_senal,
    ica_eliminar_componente,
    emg_envelope,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_EEG" not in globals():
    RUTA_EEG = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\EEG\S17_260421_EEG_OV_2_ME.npy"

if "RUTA_EMG" not in globals():
    RUTA_EMG = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\EMG\S17_260421_EMG_OV_2_ME.npy"

if "RUTA_DINAM" not in globals():
    RUTA_DINAM = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\S17\DINAM\S17_260421_DINAM_OV_2_ME.npy"

FS_EEG      = 250
FS_EMG      = 1000
BP_ORDER    = 4

FREQS_NOTCH_EEG = (60.0, 120.0)
FREQS_NOTCH_EMG = [19.6, 60, 120, 180, 240, 300]
Q_NOTCH_EEG     = 5
Q_NOTCH_EMG     = 200

PAD_EEG = FS_EEG*5
PAD_EMG = FS_EMG*5

WAVELET =       'db4'
LEVEL =         6
NIVELES_CERO =  [1, 6]     # D6 y D1
REMOVE_APPROX = True       # A6
WAVE_MODE =     'reflect'

IC_TO_REMOVE =  0          # IC0
BP_ORDER =      3     

CANALES_EEG = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
MUSCULOS    = ["FCU", "ECRL", "ECU"]

COLORES_EEG = ['#ff4783', '#e96396', '#d27fab', '#bb9bc2', '#a4b7d5', '#8dd4ea', '#76f0ff']
COLORES_EMG = ["#a4f4d7", "#d2fab1", "#ffff89"]
COLOR_DINAM =  "#ffffff"

BG_FIG      = '#0a0a0a'
BG_AX       = '#0d0d0d'
COLOR_GRID  = '#404040'
COLOR_TICK  = '#a5a5a5'
COLOR_LABEL = '#d1d1d1'
COLOR_TITLE = '#ffffff'

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DE NOMBRE DE ARCHIVO
# ─────────────────────────────────────────────────────────────────────────────

CONDICION_MAP = {
    'OV': 'Ojos Abiertos - Movimiento Visual',
    'OK': 'Ojos Abiertos - Movimiento Cinestésico',
    'CK': 'Ojos Cerrados - Movimiento Cinestésico',
    'CV': 'Ojos Cerrados - Movimiento Visual',
}
ESTADO_MAP = {
    'ME': 'Movimiento Ejecutado',
    'MI': 'Movimiento Imaginado',
    'RE': 'Reposo',
}

def parsear_nombre_archivo(ruta_eeg):
    nombre = os.path.splitext(os.path.basename(ruta_eeg))[0]
    partes = nombre.split('_')
    sujeto, condicion, estado = partes[0], partes[3], partes[5]
    return (f"Sujeto {sujeto[1:]}  |  "
            f"{CONDICION_MAP.get(condicion, condicion)}  |  "
            f"{ESTADO_MAP.get(estado, estado)}")

# ─────────────────────────────────────────────────────────────────────────────
# CARGADERA DE COSAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_npy(ruta, nombre):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró {nombre}: {ruta}")
    return np.load(ruta)

eeg   = cargar_npy(RUTA_EEG,   "EEG")
emg   = cargar_npy(RUTA_EMG,   "EMG")
dinam = cargar_npy(RUTA_DINAM, "DINAM")

t_eeg = np.arange(eeg.shape[1])  / FS_EEG
t_emg = np.linspace(0, 10, int(FS_EMG*10 / 4))
t_din = np.linspace(0, emg.shape[1] / FS_EMG, dinam.shape[0])
tmax  = min(t_eeg[-1], t_emg[-1], t_din[-1])

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO EEG
# ─────────────────────────────────────────────────────────────────────────────

def procesar_eeg(
        eeg_raw,
        fs,
        ruta_npy=None,
        canales=None,
        freqs_notch=(60, 120),
        q_notch=5,
        pad=0,
        wavelet='db4',
        level=6,
        niveles_cero=(1, 6),
        remove_approx=True,
        wave_mode='symmetric',
        ic_to_remove=0,
        f_low=8,
        f_high=45,
        bp_order=4
    ):

    datos = eeg_raw.astype(np.float64).copy()

    # 1) NOTCH 60 y 120 Hz
    datos = aplicar_notch_multiple(datos,fs=fs,freqs=freqs_notch,Q=q_notch,pad=pad)

    # 2) WAVELET
    datos = aplicar_wavelet_dwt_multicanal(datos,wavelet=wavelet,level=level,niveles_cero=niveles_cero,remove_approx=remove_approx,pad=pad,mode=wave_mode)

    # 3) ICA SOLO S5
    if ruta_npy is not None:
        es_s5 = bool(
            re.search(r'(^|[\\/])S5([\\/]|_)',ruta_npy,flags=re.IGNORECASE))

        if es_s5:
            datos, _ = ica_eliminar_componente(datos,fs=fs,ch_names=canales,ic_to_remove=ic_to_remove)

    # 4) PASA BANDA
    datos = aplicar_pasabanda_butter(datos,fs=fs,f_low=f_low,f_high=f_high,orden=bp_order,pad=pad)

    return datos

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO EMG
# ─────────────────────────────────────────────────────────────────────────────

def procesar_emg(
        emg_raw,
        fs,
        freqs_notch=(60, 120),
        q_notch=5,
        pad=0,
        factor_resample=4,
        usar_resample_profesional=False,
        fs_original=1000,
        fs_nueva=250,
        wavelet='db4',
        level=6,
        niveles_cero=(1, 6),
        remove_approx=True,
        wave_mode='symmetric',
        f_low=20,
        f_high=150,
        orden_bp=4,
        tipo_rect='completa'
    ):

    emg_proc = emg_raw.astype(np.float64).copy()

    # 1) NOTCHS
    emg_proc = aplicar_notch_multiple(emg_proc,fs=fs,freqs=tuple(freqs_notch),Q=q_notch,pad=pad)

    # 2) RESAMPLEO
    if usar_resample_profesional:
        emg_proc = resamplear_profesional(emg_proc,fs_original,fs_nueva)
        fs_out = fs_nueva

    else:
        emg_proc = resamplear_por_promedio(emg_proc,factor_resample)
        fs_out = fs / factor_resample

    # 3) WAVELET
    emg_proc = aplicar_wavelet_dwt_multicanal(emg_proc,wavelet=wavelet,level=level,niveles_cero=niveles_cero,remove_approx=remove_approx,pad=pad,mode=wave_mode)

    # 4) PASA BANDA
    emg_proc = aplicar_pasabanda_butter(emg_proc,fs=fs_out,f_low=f_low,f_high=f_high,orden=orden_bp,pad=pad)

    # 5) RECTIFICACIÓN
    emg_final = rectificar_senal(emg_proc,tipo=tipo_rect)
    
    # 6) ENVOLVENTE
    emg_env = np.zeros_like(emg_proc)
    for i in range(emg_proc.shape[0]):
        emg_env[i, :] = emg_envelope(emg_proc[i, :], fs=FS_EMG, tc_ms=20, rectify="full")


    return emg_final, emg_env

print("\n  Procesando señales EEG...")
eeg_proc = procesar_eeg(
    eeg_raw=eeg,
    fs=FS_EEG,
    ruta_npy=RUTA_EEG,
    canales=CANALES_EEG,
    freqs_notch=FREQS_NOTCH_EEG,
    q_notch=Q_NOTCH_EEG,
    pad=PAD_EEG,
    wavelet=WAVELET,
    level=LEVEL,
    niveles_cero=NIVELES_CERO,
    remove_approx=REMOVE_APPROX,
    wave_mode=WAVE_MODE,
    ic_to_remove=IC_TO_REMOVE,
    f_low=8,
    f_high=45,
    bp_order=BP_ORDER
)

print("  Procesando señales EMG...")
emg_rect, emg_env = procesar_emg(
    emg_raw=emg,
    fs=FS_EMG,
    freqs_notch=FREQS_NOTCH_EMG,
    q_notch=Q_NOTCH_EMG,
    pad=PAD_EMG,
    factor_resample=4,
    usar_resample_profesional=False,
    wavelet=WAVELET,
    level=LEVEL,
    niveles_cero=NIVELES_CERO,
    remove_approx=REMOVE_APPROX,
    wave_mode=WAVE_MODE,
    f_low=5,
    f_high=45,
    orden_bp=BP_ORDER,
    tipo_rect='completa'
)

print("\n  Procesamiento completado.")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:    manager.window.state('zoomed');   return
        except: pass
        try:    manager.window.showMaximized();   return
        except: pass
        try:    manager.frame.Maximize(True);     return
        except: pass
        fig.set_size_inches(19, 10.5)
    except:
        fig.set_size_inches(19, 10.5)
        
# def nombre_salida(ruta, sufijo):
#     CARPETA_IMGS = r"D:\luiso\Documentos\Luis\UPIITA\TT\CMC\IMGS"
#     os.makedirs(CARPETA_IMGS, exist_ok=True)          # la crea si no existe
#     nombre = os.path.splitext(os.path.basename(ruta))[0]
#     return os.path.join(CARPETA_IMGS, f"{nombre}_{sufijo}.svg")

def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT MA-MA-MASIVO
# ─────────────────────────────────────────────────────────────────────────────

eeg_ymin = np.min(eeg_proc) * 1.1
eeg_ymax = np.max(eeg_proc) * 1.1
emg_ylim = np.max(np.abs(emg_rect)) * 1.1

fig, axes = plt.subplots(10, 1, sharex=True)
fig.patch.set_facecolor(BG_FIG)
fig.subplots_adjust(left=0.035, right=0.985, top=0.96, bottom=0.08)

fig.suptitle("EEG · EMG — Señales Procesadas",
             color=COLOR_TITLE, fontsize=11, fontweight='bold')
fig.text(0.5, 0.005, parsear_nombre_archivo(RUTA_EEG),
         ha='center', va='bottom',
         fontsize=8, color=COLOR_LABEL, style='italic', fontweight='bold')

# EEG (canales 0–6)
for i in range(7):
    ax = axes[i]
    estilo_ax(ax)
    ax.plot(t_eeg, eeg_proc[i], color=COLORES_EEG[i], linewidth=0.75)
    ax.set_ylabel(CANALES_EEG[i], color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, tmax)
    ax.set_ylim(eeg_ymin, eeg_ymax)

# EMG (canales 0–2, filas 7–9)
for j in range(min(3, emg_rect.shape[0])):
    ax = axes[7 + j]
    estilo_ax(ax)
    ax.plot(t_emg, emg_rect[j], color='#d7d7d7',
            linewidth=0.5, alpha=0.5)                          # señal rectificada
    ax.plot(t_emg, emg_env[j], color=COLORES_EMG[j] if j < len(COLORES_EMG) else "#99b2ff",
            linewidth=1.5)                                      # envolvente
    ax.set_ylabel(MUSCULOS[j] if j < len(MUSCULOS) else f"EMG{j+1}", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, emg_ylim)

# # Dinamómetro (fila 10, sin procesamiento)
# ax = axes[10]
# estilo_ax(ax)
# ax.plot(t_din, dinam, color=COLOR_DINAM, linewidth=1.5)
# ax.set_ylabel("DINAM", color=COLOR_LABEL, fontsize=8)

ax.set_xlabel("Tiempo [s]", color=COLOR_LABEL, fontsize=8)
ax.set_xlim(0, tmax)

fig.set_size_inches(19, 10.5) 

# Asegurar tick labels solo en la última fila
for i in range(9):
    plt.setp(axes[i].get_xticklabels(), visible=False)
plt.setp(axes[9].get_xticklabels(), visible=True, color=COLOR_TICK)

# === DESCOMENTAR PARA GUARDAR IMG EN RUTA

# ruta_svg = nombre_salida(RUTA_DINAM, "JUNTOS")
# fig.savefig(ruta_svg, format='svg', bbox_inches='tight',
#             facecolor=fig.get_facecolor())
# print(f"\n  Figura guardada en: {ruta_svg}")

pantalla_completa(fig)

plt.show()
