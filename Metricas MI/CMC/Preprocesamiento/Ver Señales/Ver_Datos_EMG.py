import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Procesamiento_Funciones import (
    aplicar_notch_multiple,
    aplicar_pasabanda_butter,
    cargar_npy_emg,
    cargar_npy_dinam,
    resamplear_por_promedio,
    resamplear_profesional,
    aplicar_wavelet_dwt_multicanal,
    rectificar_senal,
    emg_envelope,
)

if "RUTA_EMG" not in globals():
    RUTA_EMG = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S5\EMG\S5_260429_EMG_OV_1_ME.npy"

if "RUTA_DINAM" not in globals():
    RUTA_DINAM = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2\S5\DINAM\S5_260429_DINAM_OV_1_ME.npy"

FS = 1000                
ORDEN_BP = 3              
PAD_MUESTRAS = 5 * FS
FREQS_NOTCH = [19.6, 60, 120, 180, 240, 300]
Q_NOTCH = 200

FIGURAS = [
    ("EMG Completo", 10, 450.0),
    ("EMG Completo RESAMPLED", 2.0, 113.0),
    ("EMG Completo COHERENCIA", 5.0, 45.0),
]

MUSCULOS = ["FCU", "ECRL", "ECU"]
COLORES_EMG = ["#99b2ff", "#90c7db", "#b0dfa9"]
COLOR_DINAM = "#fefb6e"

BG_FIG = "#0a0a0a"
BG_AX = "#0d0d0d"
COLOR_GRID = "#404040"
COLOR_TICK = "#a5a5a5"
COLOR_LABEL = "#d1d1d1"
COLOR_TITLE = "#ffffff"

WAVELET =       'db4'
LEVEL =         6
NIVELES_CERO =  [1, 6]    
REMOVE_APPROX = True       
WAVE_MODE =     'reflect'

CONDICION_MAP = {
    "OV": "Ojos Abiertos - Movimiento Visual",
    "OK": "Ojos Abiertos - Movimiento Cinestésico",
    "CK": "Ojos Cerrados - Movimiento Cinestésico",
    "CV": "Ojos Cerrados - Movimiento Visual",
}

ESTADO_MAP = {
    "ME": "Movimiento Ejecutado",
    "MI": "Movimiento Imaginado",
    "RE": "Reposo",
}


def parsear_nombre_archivo(ruta):
    nombre = os.path.splitext(os.path.basename(ruta))[0]        
    partes = nombre.split('_')                                

    sujeto    = partes[0]                                    
    condicion = partes[3]                                  
    estado    = partes[5]                                      

    num_sujeto   = sujeto[1:]                                  
    desc_cond    = CONDICION_MAP.get(condicion, condicion)
    desc_estado  = ESTADO_MAP.get(estado, estado)

    return f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"

PIE_FIGURA = parsear_nombre_archivo(RUTA_DINAM)

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

def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--')

def calcular_fft(senal: np.ndarray, fs: float):
    N = len(senal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_vals = np.fft.rfft(senal)
    mag = np.abs(fft_vals) / N
    if N > 2:
        mag[1:-1] *= 2
    return freqs, mag

def plot_figura(nombre, emg_filt, dinam, t_e, t_d, f_low, f_high, fs, pie):

    fig = plt.figure()
    fig.patch.set_facecolor(BG_FIG)

    gs = gridspec.GridSpec(
        4, 2,
        width_ratios=[1, 4],
        hspace=0.15, wspace=0.08,
        left=0.035, right=0.985, top=0.96, bottom=0.08
    )

    ax_fft0 = fig.add_subplot(gs[0, 0])
    ax_fft1 = fig.add_subplot(gs[1, 0], sharex=ax_fft0, sharey=ax_fft0)
    ax_fft2 = fig.add_subplot(gs[2, 0], sharex=ax_fft0, sharey=ax_fft0)
    ax_fft3 = fig.add_subplot(gs[3, 0]) 
    ax_sig0 = fig.add_subplot(gs[0, 1])
    ax_sig1 = fig.add_subplot(gs[1, 1], sharex=ax_sig0, sharey=ax_sig0)
    ax_sig2 = fig.add_subplot(gs[2, 1], sharex=ax_sig0, sharey=ax_sig0)
    ax_din  = fig.add_subplot(gs[3, 1], sharex=ax_sig0)  

    axes = np.array([
        [ax_fft0, ax_sig0],
        [ax_fft1, ax_sig1],
        [ax_fft2, ax_sig2],
        [ax_fft3, ax_din ],
    ])

    fig.set_size_inches(19, 10.5)

    for i in range(n_canales):
        ax_fft = axes[i, 0]
        ax_sig = axes[i, 1]
        color  = COLORES_EMG[i]

        ax_fft.set_ylabel(MUSCULOS[i], color=color, fontsize=9,
                          rotation=90, labelpad=12, va='center')
        freqs, mag = calcular_fft(emg_filt[i], fs)
        ax_fft.plot(freqs, mag, color=color, linewidth=0.75, alpha=0.92)
        ax_fft.set_xlim(0, fs / 2)
        estilo_ax(ax_fft)
        if i == 0:
            ax_fft.set_title("FFT  [|X(f)|]", color=COLOR_LABEL, fontsize=8, pad=3)

        ax_sig.plot(t_e, emg_filt[i], color=color, linewidth=0.75, alpha=0.92)
        ax_sig.set_xlim(t_e[0], t_e[-1])
        estilo_ax(ax_sig)
        if i == 0:
            ax_sig.set_title("Señal filtrada  [µV]", color=COLOR_LABEL, fontsize=8, pad=3)

    for i in range(n_canales - 1):
        plt.setp(axes[i, 0].get_xticklabels(), visible=False)
        plt.setp(axes[i, 1].get_xticklabels(), visible=False)

    plt.setp(axes[n_canales - 1, 0].get_xticklabels(), visible=True, color=COLOR_TICK)
    plt.setp(axes[n_canales - 1, 1].get_xticklabels(), visible=True, color=COLOR_TICK)
    axes[n_canales - 1, 0].set_xlabel("Frecuencia [Hz]", color=COLOR_LABEL, fontsize=8)

    axes[3, 0].set_visible(False)

    ax_din = axes[3, 1]
    ax_din.plot(t_d, dinam, color=COLOR_DINAM, linewidth=1.5, alpha=0.95)
    ax_din.set_ylabel("DINAMÓMETRO\n[N]", color=COLOR_DINAM,
                      fontsize=8, rotation=90, labelpad=20, va='center')
    ax_din.tick_params(axis='y', colors=COLOR_DINAM)
    ax_din.set_xlabel("Tiempo [s]", color=COLOR_LABEL, fontsize=8)
    ax_din.set_xlim(t_d[0], t_d[-1])
    estilo_ax(ax_din)
    plt.setp(ax_din.get_xticklabels(), visible=True, color=COLOR_TICK)

    fig.suptitle(
        f"{nombre}  —  Butterworth ord. {ORDEN_BP}  |  {f_low}–{f_high} Hz  |  Fs = {fs} Hz",
        color=COLOR_TITLE, fontsize=11, fontweight='bold'
    )
    fig.text(0.5, 0.005, pie, ha='center', va='bottom',
             fontsize=8, color=COLOR_LABEL, style='italic', fontweight='bold')

    pantalla_completa(fig)
    return fig

emg_data   = cargar_npy_emg(RUTA_EMG)         
dinam_data = cargar_npy_dinam(RUTA_DINAM)   

n_canales  = emg_data.shape[0]
n_muestras = emg_data.shape[1]
t_emg      = np.linspace(0, emg_data.shape[1]/FS, n_muestras)

n_dinam  = len(dinam_data)
t_dinam = np.linspace(0, emg_data.shape[1]/FS, n_dinam)

emg_proc = aplicar_notch_multiple(
    emg_data,
    fs=FS,
    freqs=tuple(FREQS_NOTCH),
    Q=Q_NOTCH,
    pad=PAD_MUESTRAS,
)
print(f"\n  [OK] Notch aplicado: {FREQS_NOTCH} Hz  |  Q={Q_NOTCH}  |  pad={PAD_MUESTRAS} muestras")

emg_proc = resamplear_por_promedio(emg_proc,4)

FS = FS / 4  
n_muestras = emg_proc.shape[1]
t_emg = np.linspace(0, n_muestras / FS, n_muestras)

print(f"\n  [OK] Resampleo aplicado. Nueva FS: {FS} Hz | Muestras: {n_muestras}")

emg_proc = aplicar_wavelet_dwt_multicanal(
    emg_proc,
    wavelet=WAVELET,
    level=LEVEL,
    niveles_cero=NIVELES_CERO,
    remove_approx=REMOVE_APPROX,
    pad=PAD_MUESTRAS,
    mode=WAVE_MODE
)
print(f"  [OK] Wavelet aplicado: {WAVELET} | level={LEVEL} | cero={NIVELES_CERO} | remove A{LEVEL}={REMOVE_APPROX}")

for nombre, f_low, f_high in FIGURAS:
    print(f"\n → Generando: {nombre} ({f_low:.1f}–{f_high:.1f} Hz)")

    emg_final = aplicar_pasabanda_butter(
        emg_proc,
        fs=FS,
        f_low=f_low,
        f_high=f_high,
        orden=ORDEN_BP,
        pad=PAD_MUESTRAS,
    )

    for i in range(emg_final.shape[0]):
        emg_final[i, :] = emg_envelope(emg_final[i, :], fs=FS, tc_ms=20, rectify="full")
    
    plot_figura(nombre, emg_final, dinam_data, t_emg, t_dinam, f_low, f_high, FS, PIE_FIGURA)

print("\n  [FIN] Todas las figuras EMG/DINAM generadas.\n")
plt.show()