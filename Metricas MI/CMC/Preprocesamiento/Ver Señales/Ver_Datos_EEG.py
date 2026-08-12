import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_wavelet_dwt_multicanal,
    ica_eliminar_componente,
    aplicar_pasabanda_butter,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

if "RUTA_NPY" not in globals():
    RUTA_NPY = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S5\EEG\S5_260429_EEG_OK_1_ME.npy"

FS = 250

VENTANA_SEG = 1.0
NPERSEG = int(VENTANA_SEG * FS)

PAD_MUESTRAS = 5 * FS

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']

FIGURAS = [
    # ('EEG Completo',    0.5,  45.0),
    # ('Delta δ',         0.5,   4.0),
    # ('Theta θ',         4.0,   8.0),
    # ('Mu μ',            8.0,  13.0),
    # ('Beta β',         13.0,  30.0),
    # ('Gamma γ',        30.0,  45.0),
    # ('μ + β',           8.0,  30.0),
    # ('μ + β + γ',       8.0,  45.0),
    ('Preprocesamiento Coherencia',       5.0,  45.0),
]

BANDAS = {
    'δ' : (0.5,  4,  '#9B3461'),
    'θ' : (4,    8,  '#C15579'),
    'μ' : (8,   13,  '#E77692'),
    'β' : (13,  30,  '#f0B8AB'),
    'γ' : (30,  45,  '#FAFAC4'),
}

COLORES = ['#6FC7B7', '#A1DAD1', '#D0EDE8','#FFFFFF',  '#BFEEF6', '#81DCEE', '#41CCE3']

BG_FIG      = '#0a0a0a'
BG_AX       = '#0d0d0d'
COLOR_GRID  = '#404040'
COLOR_TICK  = '#a5a5a5'
COLOR_LABEL = '#d1d1d1'
COLOR_TITLE = '#ffffff'

FREQS_NOTCH =   (60.0, 120.0)
Q_NOTCH =       5.0

WAVELET =       'db4'
LEVEL =         6
NIVELES_CERO =  [1, 6]     # D6 y D1
REMOVE_APPROX = True       # A6
WAVE_MODE =     'reflect'

IC_TO_REMOVE =  0          # IC0
BP_ORDER =      3          

# ─────────────────────────────────────────────────────────────────────────────
# PARASEO DE LAS FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

CONDICION_MAP = {
    'OV': 'Ojos Abiertos - Movimiento Visual (OV)',
    'OK': 'Ojos Abiertos - Movimiento Cinestésico (OK)',
    'CK': 'Ojos Cerrados - Movimiento Cinestésico (CK)',
    'CV': 'Ojos Cerrados - Movimiento Visual (CV)',
}

ESTADO_MAP = {
    'ME': 'Movimiento Ejecutado (ME)',
    'MI': 'Movimiento Imaginado (MI)',
    'RE': 'Reposo (RE)',
}

def parsear_nombre_archivo(ruta):
    nombre = os.path.splitext(os.path.basename(ruta))[0]        # 'S17_260310_EEG_CK_3_RE'
    partes = nombre.split('_')                                  # ['S17','260310','EEG','CK','3','RE']

    sujeto    = partes[0]                                       # 'S17'
    condicion = partes[3]                                       # 'CK'
    estado    = partes[5]                                       # 'RE'

    num_sujeto   = sujeto[1:]                                   # '17'
    desc_cond    = CONDICION_MAP.get(condicion, condicion)
    desc_estado  = ESTADO_MAP.get(estado, estado)

    return f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"

PIE_FIGURA = parsear_nombre_archivo(RUTA_NPY)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:
            manager.window.state('zoomed'); return
        except:
            pass
        try:
            manager.window.showMaximized(); return
        except:
            pass
        try:
            manager.frame.Maximize(True); return
        except:
            pass
        fig.set_size_inches(19, 10.5)
    except:
        fig.set_size_inches(19, 10.5)

def calcular_psd(señal, fs, nperseg):
    freqs, psd = welch(señal, fs=fs, nperseg=nperseg)
    return freqs, psd

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

def plot_bandas_psd(ax, psd_max, f_low, f_high, etiquetas=False):
    for nombre, (b_ini, b_fin, bc) in BANDAS.items():
        if b_fin <= f_low or b_ini >= f_high:
            continue
        ax.axvspan(max(b_ini, f_low), min(b_fin, f_high),
                   alpha=0.15, color=bc, zorder=0)
        if etiquetas:
            ax.text((b_ini + b_fin) / 2, psd_max * 1.02,
                    nombre, color=bc, fontsize=6.5,
                    ha='center', va='bottom')

def plot_figura(nombre_fig, datos_filt, f_low, f_high, tiempo, fs, nperseg, colores, canales, pie):
    fig, axes = plt.subplots(
        7, 2, sharex='col', sharey='col', 
        gridspec_kw={'width_ratios': [3,1], 'hspace': 0.15, 'wspace': 0.08}
    )
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.96, bottom=0.08)
    
    fig.set_size_inches(19, 10.5) 

    fig.suptitle(
        f'{nombre_fig} — Butterworth ord. {BP_ORDER} | {f_low}–{f_high} Hz | fs = {fs} Hz', 
        color=COLOR_TITLE, fontsize=11, fontweight='bold'
    )
    
    y_min_fig = datos_filt.min()
    y_max_fig = datos_filt.max()
    margen_fig = (y_max_fig - y_min_fig) * 0.08

    psd_max_global = 0.0
    psds = []
    for i in range(7):
        freqs, psd = calcular_psd(datos_filt[i], fs, nperseg)
        psds.append((freqs, psd))
        psd_max_global = max(psd_max_global, psd.max())

    fig.text(0.5, 0.01, pie, ha='center', va='bottom', fontsize=8, color= COLOR_LABEL, style='italic', fontweight='bold')

    for i in range(7):
        ax_t = axes[i, 0]
        ax_p = axes[i, 1]
        estilo_ax(ax_t); estilo_ax(ax_p)
        
        # Tiempo
        ax_t.plot(tiempo, datos_filt[i], color=colores[i], linewidth=0.8)
        ax_t.set_ylabel(canales[i], color=COLOR_LABEL, fontsize=8)
        ax_t.set_xlim(tiempo[0], tiempo[-1])
        ax_t.set_ylim(y_min_fig - margen_fig, y_max_fig + margen_fig)
        
        if i == 6:
            ax_t.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
        
        freqs, psd = psds[i]
        ax_p.plot(freqs, psd, color=colores[i], linewidth=0.9)
        ax_p.set_xlim(0, max(45, f_high) + 8)
        ax_p.set_ylim(0, psd_max_global * 1.15)
        
        if i == 6:
            ax_p.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    
    # Ocultar tick labels en filas intermedias DESPUÉS de dibujar todo
    for i in range(6):           # filas 0–5
        plt.setp(axes[i, 0].get_xticklabels(), visible=False)
        plt.setp(axes[i, 1].get_xticklabels(), visible=False)
    
    # Asegurar que la fila 6 sí los muestre
    plt.setp(axes[6, 0].get_xticklabels(), visible=True, color=COLOR_TICK)
    plt.setp(axes[6, 1].get_xticklabels(), visible=True, color=COLOR_TICK)

    for i in range(7):
        ax_p = axes[i, 1]
        plot_bandas_psd(ax_p, psd_max_global, f_low, f_high, etiquetas=(i == 0))
        
    # === DESCOMENTAR PARA GUARDAR IMG EN RUTA
    
    # ruta_svg = nombre_salida(RUTA_EMG, "EMG")
    # fig.savefig(ruta_svg, format='svg', bbox_inches='tight',
    #             facecolor=fig.get_facecolor())
    # print(f"\n  Figura guardada en: {ruta_svg}")

    pantalla_completa(fig)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAR SEÑALES
# ─────────────────────────────────────────────────────────────────────────────

# 1) Cargar
datos = cargar_npy_eeg(RUTA_NPY, n_canales_esperados=len(CANALES))
n_canales, n_muestras = datos.shape
tiempo = np.arange(n_muestras) / FS

# 2) Notch 60 y 120 Hz (Q=5)
datos = aplicar_notch_multiple(datos, fs=FS, freqs=FREQS_NOTCH, Q=Q_NOTCH, pad=PAD_MUESTRAS)
print(f"\n  [OK] Notch aplicado: {FREQS_NOTCH} Hz | Q={Q_NOTCH}")

# 3) Wavelet db4 nivel 6, eliminar D1 D6 y A6
datos = aplicar_wavelet_dwt_multicanal(
    datos,
    wavelet=WAVELET,
    level=LEVEL,
    niveles_cero=NIVELES_CERO,
    remove_approx=REMOVE_APPROX,
    pad=PAD_MUESTRAS,
    mode=WAVE_MODE
)
print(f"  [OK] Wavelet aplicado: {WAVELET} | level={LEVEL} | cero={NIVELES_CERO} | remove A{LEVEL}={REMOVE_APPROX}")

# 4) ICA solo si el sujeto es S5 (remover IC0)
es_s5 = bool(re.search(r'(^|[\\/])S5([\\/]|_)', RUTA_NPY, flags=re.IGNORECASE))
if es_s5:
    datos, _ica = ica_eliminar_componente(datos, fs=FS, ch_names=CANALES, ic_to_remove=IC_TO_REMOVE)
    print(f"  [OK] ICA aplicada (solo S5): removida IC{IC_TO_REMOVE}")
else:
    print("  [SKIP] ICA omitida (sujeto distinto de S5)")

# 5) Pasa Banda y Plot
for (nombre, f_low, f_high) in FIGURAS:
    print(f"\n  → Generando: {nombre} ({f_low}-{f_high} Hz)")
    datos_filt = aplicar_pasabanda_butter(
        datos,
        fs=FS,
        f_low=f_low,
        f_high=f_high,
        orden=BP_ORDER,
        pad=PAD_MUESTRAS
    )
    plot_figura(nombre, datos_filt, f_low, f_high, tiempo, FS, NPERSEG, COLORES, CANALES, PIE_FIGURA)

print("\n  [FIN] Todas las figuras EEG generadas.\n")

plt.show()