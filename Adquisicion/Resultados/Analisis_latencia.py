"""
Análisis de latencia EEG / EMG vs. Dinamómetro
================================================
Señales en formato .npy, arreglo 2D (canales x muestras).
Tres frecuencias de muestreo independientes.

Métodos de detección de onset disponibles:
  1. Umbral fijo      - supera N% del valor máximo de la señal
  2. Media + k·std    - supera la media del reposo más k desviaciones estándar
  3. Cambio de pendiente (Derivada) - primera Derivada supera un umbral
  4. Energía (RMS)    - envolvente RMS supera umbral (muy útil en EMG)

Uso rápido
----------
Edita el bloque "CONFIGURACIÓN" y ejecuta:
    python latency_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, butter, filtfilt
from math import gcd


# ─────────────────────────────────────────────
# CONFIGURACIÓN  
# ─────────────────────────────────────────────
    # (7 canales × muestras)
EMG_BASE = r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2 copia"    
DINO_BASE = r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-BCI\DataCMC\Registros CMC 2 copia" 
archivos = {"EMG": ["\S4\EMG\S4_260507_EMG_CK_3_ME.npy", "\S5\EMG\S5_260429_EMG_CK_3_ME.npy","\S7\EMG\S7_260422_EMG_CV_3_ME.npy","\S17\EMG\S17_260421_EMG_CK_2_ME.npy","\S18\EMG\S18_260429_EMG_CK_1_ME.npy"], 
            "DINO": ["\S4\DINAM\S4_260507_DINAM_CK_3_ME.npy", "\S5\DINAM\S5_260429_DINAM_CK_3_ME.npy","\S7\DINAM\S7_260422_DINAM_CV_3_ME.npy","\S17\DINAM\S17_260421_DINAM_CK_2_ME.npy","\S18\DINAM\S18_260429_DINAM_CK_1_ME.npy"]}
canales_emg = ["FCU", "ECRL", "ECU"]
FS_EMG   = 1000   # Hz
FS_DINO  = 60   # Hz

# fs_target: todos los canales se remuestrean a esta frecuencia.
FS_TARGET = 1000  # Hz — fs común para el análisis
 
REPOSO_SEG = 2.0  # segundos de reposo al inicio de CADA señal
DINO_CANAL = 0    # índice del canal del dinamómetro a usar
 
METODOS = ["Derivada"]
 
# ── Parámetros de detección ──
UMBRAL_FIJO_PCT  = 0.10    # fracción del máximo global
MEDIA_STD_K      = 5.0     # media_reposo + k·std
DERIVADA_PCT     = 0.10    # fracción del máximo de |diff| (más alto = menos falsos positivos)
RMS_VENTANA_MS   = 100     # ventana RMS en ms
RMS_K            = 5.0     # media_rms_reposo + k·std
 
# ── Ventana de búsqueda ──
# Solo busca onset dentro de este margen DESPUÉS del reposo.
# Protege contra falsos positivos al final de la señal.
BUSQUEDA_MAX_SEG = 4.0     # segundos máximos de búsqueda tras el reposo
 
# ── Suavizado del dinamómetro (media móvil) ──
# Recomendado cuando la fs del dino es baja (≤100 Hz).
# Reduce falsas detecciones por variaciones pequeñas al final.
DINO_SMOOTH_MS   = 200     # ms  (0 = sin suavizado)
 
# ── Filtros ──
EMG_BANDPASS  = (5.0, 450.0)
DINO_BANDPASS = None

# ─────────────────────────────────────────────
# FIN CONFIGURACIÓN
# ─────────────────────────────────────────────
 
METHOD_COLOR = {"umbral_fijo": "#E24B4A", "media_std": "#7F77DD",
                "Derivada":    "#BA7517", "energia_rms": "#185FA5"}
METHOD_LS    = {"umbral_fijo": "-",       "media_std": "--",
                "Derivada":    "-.",      "energia_rms": ":"}
 
 
# ── Utilidades ────────────────────────────────
 
def bandpass(sig, fs, lo, hi, order=4):
    nyq = fs / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    lo_n = max(1e-4, min(lo_n, 0.999))
    hi_n = max(1e-4, min(hi_n, 0.999))
    if lo_n >= hi_n:
        return sig
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig)
 
 
def apply_filter(data_2d, fs, bp):
    if bp is None:
        return data_2d
    return np.array([bandpass(ch, fs, bp[0], bp[1]) for ch in data_2d])
 
 
def resample_2d(data_2d, fs_orig, fs_target):
    if fs_orig == fs_target:
        return data_2d
    g = gcd(int(fs_target), int(fs_orig))
    up, down = int(fs_target) // g, int(fs_orig) // g
    return np.array([resample_poly(ch, up, down) for ch in data_2d])
 
 
def smooth(sig, fs, window_ms):
    """Media móvil simple."""
    w = max(1, int(fs * window_ms / 1000))
    if w <= 1:
        return sig
    kernel = np.ones(w) / w
    return np.convolve(sig, kernel, mode="same")
 
 
def rms_envelope(sig, fs, window_ms=100):
    w = max(1, int(fs * window_ms / 1000))
    return np.sqrt(np.convolve(sig**2, np.ones(w) / w, mode="same"))
 
 
def detect_onset(sig, fs, reposo_seg, method, apply_smooth=False):
    """
    Devuelve índice (muestra) del onset o None.
    La búsqueda se limita a BUSQUEDA_MAX_SEG segundos después del reposo.
    """
    n_rep      = int(reposo_seg * fs)
    n_busqueda = int(BUSQUEDA_MAX_SEG * fs)
 
    # Suavizado opcional (solo para el dinamómetro)
    sig_work = smooth(sig, fs, DINO_SMOOTH_MS) if apply_smooth else sig
 
    reposo = sig_work[:n_rep]
    # Ventana de búsqueda acotada
    activo = sig_work[n_rep : n_rep + n_busqueda]
 
    if len(activo) == 0:
        return None
 
    if method == "umbral_fijo":
        thr = UMBRAL_FIJO_PCT * np.max(np.abs(sig_work))
        idx = np.where(np.abs(activo) > thr)[0]
 
    elif method == "media_std":
        mu, sd = np.mean(reposo), np.std(reposo)
        thr = mu + MEDIA_STD_K * sd
        idx = np.where(activo > thr)[0]
 
    elif method == "Derivada":
        d   = np.abs(np.diff(sig_work, prepend=sig_work[0]))
        # Calcular umbral solo sobre la región de reposo+búsqueda
        thr = DERIVADA_PCT * np.max(d[:n_rep + n_busqueda])
        idx = np.where(d[n_rep : n_rep + n_busqueda] > thr)[0]
 
    elif method == "energia_rms":
        env    = rms_envelope(sig_work, fs, RMS_VENTANA_MS)
        mu, sd = np.mean(env[:n_rep]), np.std(env[:n_rep])
        thr    = mu + RMS_K * sd
        idx    = np.where(env[n_rep : n_rep + n_busqueda] > thr)[0]
 
    else:
        raise ValueError(f"Método desconocido: {method}")
 
    return (n_rep + idx[0]) if len(idx) > 0 else None
 
 
# ── Carga ─────────────────────────────────────
 
def load_signals():
    print("Cargando archivos...")
    emg_raw  = np.load(EMG_FILE)
    dino_raw = np.load(DINO_FILE)
 
    if dino_raw.ndim == 1:
        dino_raw = dino_raw[np.newaxis, :]
 
    print(f"  EMG  shape: {emg_raw.shape}  @ {FS_EMG} Hz")
    print(f"  DINO shape: {dino_raw.shape} @ {FS_DINO} Hz")
 
    emg_f  = apply_filter(emg_raw,  FS_EMG,  EMG_BANDPASS)
    dino_f = apply_filter(dino_raw, FS_DINO, DINO_BANDPASS)
 
    # Remuestrear a FS_TARGET — relojes independientes, NO recortar
    emg_rs  = resample_2d(emg_f,  FS_EMG,  FS_TARGET)
    dino_rs = resample_2d(dino_f, FS_DINO, FS_TARGET)
 
    print(f"  EMG  remuestrado: {emg_rs.shape[1]}  muestras "
          f"({emg_rs.shape[1]/FS_TARGET:.2f} s)")
    print(f"  DINO remuestrado: {dino_rs.shape[1]} muestras "
          f"({dino_rs.shape[1]/FS_TARGET:.2f} s)\n")
    return emg_rs, dino_rs
 
 
# ── Análisis ──────────────────────────────────
 
def analyze(emg_rs, dino_rs):
    dino_ch = dino_rs[DINO_CANAL]
    fs      = FS_TARGET
    resultados = {}
 
    for method in METODOS:
        print(f"── Método: {method} ──")
 
        # Dinamómetro: se suaviza antes de detectar onset
        onset_dino = detect_onset(dino_ch, fs, REPOSO_SEG, method, apply_smooth=True)
        if onset_dino is None:
            print("  [!] No se detectó onset en el dinamómetro.\n")
            resultados[method] = None
            continue
 
        t_dino = onset_dino / fs
        print(f"  Onset DINO : muestra {onset_dino:6d}  →  t = {t_dino:.4f} s")
 
        lags, onsets_emg = [], []
        for i, ch in enumerate(emg_rs):
            o = detect_onset(ch, fs, REPOSO_SEG, method, apply_smooth=False)
            if o is None:
                print(f"  EMG canal {i}: onset no detectado")
                continue
            t_emg  = o / fs
            lat_ms = (t_dino - t_emg) * 1000
            lags.append(lat_ms)
            onsets_emg.append(o)
            print(f"  EMG canal {i}: muestra {o:6d}  →  t = {t_emg:.4f} s  "
                  f"| latencia click = {lat_ms:+.1f} ms")
 
        if lags:
            media, std = float(np.mean(lags)), float(np.std(lags))
            print(f"  → Latencia click promedio: {media:+.2f} ms  (±{std:.2f} ms)\n")
            resultados[method] = dict(
                t_dino=t_dino, onset_dino=onset_dino,
                onsets_emg=onsets_emg, lags=lags,
                media_ms=media, std_ms=std
            )
        else:
            resultados[method] = None
            print()
 
    return resultados
 
 
# ── Tabla resumen ─────────────────────────────
 
def print_summary(resultados):
    print("═" * 54)
    print(f"{'MÉTODO':<20} {'LATENCIA CLICK (ms)':>22}")
    print("─" * 54)
    for m in METODOS:
        r = resultados.get(m)
        if r is None:
            print(f"{m:<20} {'N/D':>22}")
        else:
            print(f"{m:<20} {r['media_ms']:>+12.2f}  ±{r['std_ms']:.2f}")
    print("═" * 54)
    print("Positivo → EMG arrancó DESPUÉS del dinamómetro.")
    print("Negativo → EMG arrancó ANTES  del dinamómetro.\n")
 
 
# ── Gráfica ───────────────────────────────────
 
def plot_results(emg_rs, dino_rs, resultados, sujeto="S5"):
    fs     = FS_TARGET
    t_dino = np.arange(dino_rs.shape[1]) / fs
    t_emg  = np.arange(emg_rs.shape[1])  / fs
 
    # Señal del dino suavizada (para mostrar lo que ve el detector)
    dino_ch      = dino_rs[DINO_CANAL]
    dino_smooth  = smooth(dino_ch, fs, DINO_SMOOTH_MS)
 
    fig, (ax_dino, ax_emg) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    fig.subplots_adjust(hspace=0.6)
 
    # ── Panel dinamómetro ──
    ax_dino.plot(t_dino, dino_ch,     color="#1D9E75", lw=1.2, alpha=0.4, label="Dinamómetro (crudo)")
    ax_dino.plot(t_dino, dino_smooth, color="#1D9E75", lw=2.0, label=f"Dinamómetro (suavizado)")
    ax_dino.set_ylabel("Fuerza (N)", fontsize=13)
    ax_dino.set_xlabel("Tiempo desde inicio dinamómetro (s)", fontsize=13)
    ax_dino.set_title("Dinamómetro", fontsize=15)
    ax_dino.tick_params(axis='both', labelsize=13)
 
    for method, res in resultados.items():
        if res is None:
            continue
        ax_dino.axvline(res["t_dino"], color=METHOD_COLOR[method],
                        lw=1.4, ls=METHOD_LS[method], label=method)
    ax_dino.legend(loc="upper left", fontsize=12)
 
    # ── Panel EMG ──
    orgs = plt.cm.Oranges(np.linspace(0.45, 0.9, emg_rs.shape[0]))
    sep  = 500
    for i, ch in enumerate(emg_rs):
        ax_emg.plot(t_emg, np.abs(ch) + i * sep,
                    color=orgs[i], lw=0.8, label=f"{canales_emg[i]} (+{i*sep} µV)")
 
    for method, res in resultados.items():
        if res is None:
            continue
        for o in res["onsets_emg"]:
            ax_emg.axvline(o / fs, color=METHOD_COLOR[method],
                           lw=1.0, ls=METHOD_LS[method], alpha=0.75)

    ax_emg.set_title("EMG (Offset visual)", fontsize=15)
    ax_emg.set_ylabel("µV", fontsize=13)
    ax_emg.set_xlabel("Tiempo desde inicio EMG (s)", fontsize=13)
    ax_emg.tick_params(axis='both', labelsize=13)
    ax_emg.legend(loc="upper left", fontsize=12)
 
    validos = {m: r for m, r in resultados.items() if r is not None}
    if validos:
        mejor = min(validos, key=lambda m: validos[m]["std_ms"])
        lat   = validos[mejor]["media_ms"]
        #ax_emg.set_title(
        #    f"Latencia click promedio: {lat:.2f} ms ± {validos[mejor]['std_ms']:.2f} ms\n",
        #    fontsize=11
        #)
 
    #plt.suptitle("Latencia de incialización manual: EMG vs Dinamómetro\n"
    #             "(cada señal en su propio eje de tiempo)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"click_latency_{sujeto}.png", dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada: click_latency_{sujeto}.png")
 
    # ── Barras de resumen ──
    metodos_v = [m for m in METODOS if resultados.get(m)]
    if not metodos_v:
        print("[!] Sin resultados para graficar barras.")
        plt.show()
        return
 
    lags_med = [resultados[m]["media_ms"] for m in metodos_v]
    lags_std = [resultados[m]["std_ms"]   for m in metodos_v]
 
    fig2, ax = plt.subplots(figsize=(8, 4))
    x    = np.arange(len(metodos_v))
    bars = ax.bar(x, lags_med, 0.5, yerr=lags_std,
                  color=[METHOD_COLOR[m] for m in metodos_v],
                  alpha=0.85, error_kw=dict(capsize=6, ecolor="black"))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metodos_v, fontsize=11)
    ax.set_ylabel("Latencia click (ms)", fontsize=11)
    ax.set_title("Latencia del click por método de detección de onset", fontsize=12)
 
    for bar, val in zip(bars, lags_med):
        offset = 3 if val >= 0 else -14
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.1f}", ha="center", fontsize=10)
 
    plt.tight_layout()
    plt.savefig(f"click_latency_summary_{sujeto}.png", dpi=150, bbox_inches="tight")
    print(f"Gráfica de resumen guardada: click_latency_summary_{sujeto}.png")
    plt.show()
 
 
# ── Resumen global (todas las iteraciones) ────
 
def print_global_summary(todas_iteraciones):
    """Calcula y muestra estadísticas globales de todas las iteraciones."""
    print("\n" + "═" * 70)
    print(" RESUMEN GLOBAL DE TODAS LAS ITERACIONES")
    print("═" * 70)
    
    for method in METODOS:
        lags_globales = []
        # Acumular todos los lags de este método de todas las iteraciones
        for iter_data in todas_iteraciones:
            if iter_data.get(method) and iter_data[method]["lags"]:
                lags_globales.extend(iter_data[method]["lags"])
        
        if lags_globales:
            media_global = float(np.mean(lags_globales))
            std_global   = float(np.std(lags_globales))
            n_canales    = len(lags_globales)
            print(f"\n{method.upper()}")
            print(f"  Total latencias: {n_canales}")
            print(f"  Latencia promedio: {media_global:+.2f} ms")
            print(f"  Desviación estándar: {std_global:.2f} ms")
            print(f"  Rango: [{min(lags_globales):.2f}, {max(lags_globales):.2f}] ms")
        else:
            print(f"\n{method.upper()}")
            print(f"  Sin datos disponibles")
    
    print("═" * 70 + "\n")


def plot_global_results(todas_iteraciones):
    """Genera gráficas consolidadas de todas las iteraciones."""
    # Recolectar datos globales por método
    datos_por_metodo = {}
    for method in METODOS:
        lags_globales = []
        for iter_data in todas_iteraciones:
            if iter_data.get(method) and iter_data[method]["lags"]:
                lags_globales.extend(iter_data[method]["lags"])
        datos_por_metodo[method] = lags_globales
    
    # Filtrar métodos con datos
    metodos_con_datos = [m for m in METODOS if datos_por_metodo[m]]
    
    if not metodos_con_datos:
        print("[!] Sin datos para graficar resultados globales.")
        return
    
    # ─── Figura 1: Boxplot y distribución ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot
    lags_list = [datos_por_metodo[m] for m in metodos_con_datos]
    ax = axes[0]
    bp = ax.boxplot(lags_list, labels=metodos_con_datos, patch_artist=True)
    for patch, method in zip(bp['boxes'], metodos_con_datos):
        patch.set_facecolor(METHOD_COLOR[method])
        patch.set_alpha(0.7)
    ax.axhline(0, color="red", lw=1, ls="--", alpha=0.5)
    ax.set_ylabel("Latencia (ms)", fontsize=11)
    ax.set_title("Distribución de latencias (Boxplot)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Histograma
    ax = axes[1]
    for method in metodos_con_datos:
        lags = datos_por_metodo[method]
        ax.hist(lags, bins=8, alpha=0.6, label=method,
                color=METHOD_COLOR[method], edgecolor="black")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7, label="t=0")
    ax.set_xlabel("Latencia (ms)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.set_title("Histograma de latencias (todas las iteraciones)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("latency_global_distribution.png", dpi=150, bbox_inches="tight")
    print("Gráfica de distribución global guardada: latency_global_distribution.png")
    
    # ─── Figura 2: Barras con estadísticas globales ───
    fig2, ax = plt.subplots(figsize=(10, 5))
    
    medias = [float(np.mean(datos_por_metodo[m])) for m in metodos_con_datos]
    stds = [float(np.std(datos_por_metodo[m])) for m in metodos_con_datos]
    
    x = np.arange(len(metodos_con_datos))
    bars = ax.bar(x, medias, 0.6, yerr=stds,
                  color=[METHOD_COLOR[m] for m in metodos_con_datos],
                  alpha=0.8, error_kw=dict(capsize=8, elinewidth=2, ecolor="black"))
    
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metodos_con_datos, fontsize=11)
    ax.set_ylabel("Latencia promedio (ms)", fontsize=11)
    ax.set_title("Latencia promedio global ± desviación estándar\n(TODAS las iteraciones)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Etiquetas en barras
    for bar, media, std in zip(bars, medias, stds):
        offset = 5 if media >= 0 else -15
        ax.text(bar.get_x() + bar.get_width() / 2, media + offset,
                f"{media:+.1f}\n±{std:.1f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("latency_global_summary.png", dpi=150, bbox_inches="tight")
    print("Gráfica de resumen global guardada: latency_global_summary.png\n")
    
    plt.show()


# ── Main ──────────────────────────────────────
 
if __name__ == "__main__":
    todas_iteraciones = []
    
    for i in range(len(archivos["EMG"])):
        EMG_FILE  = f"{EMG_BASE}\\{archivos['EMG'][i]}"
        DINO_FILE = f"{DINO_BASE}\\{archivos['DINO'][i]}"
        sujeto = f"S{i+1}"  
        print(f"\nProcesando sujeto: {sujeto}")
        print(f"\n{'─' * 70}")
        print(f"ITERACIÓN {i+1}/{len(archivos['EMG'])}")
        print(f"{'─' * 70}")
        
        emg_rs, dino_rs = load_signals()
        resultados = analyze(emg_rs, dino_rs)
        print_summary(resultados)
        plot_results(emg_rs, dino_rs, resultados, sujeto)
        
        # Guardar resultados de esta iteración
        todas_iteraciones.append(resultados)
    
    # Mostrar resumen global
    print_global_summary(todas_iteraciones)
    
    # Graficar resultados consolidados
    plot_global_results(todas_iteraciones)
 
